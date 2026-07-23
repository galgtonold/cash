# Smart persistence — when caching to disk would cost more than it saves

Caching to disk costs I/O. For very cheap computations, the disk-write itself takes longer than rerunning the function. Cash's smart-persistence policy decides per call whether a result is worth promoting past RAM — using the call's measured execution time and the result's size as inputs.

## Why this exists

A naive "cache everything to disk" backend hurts when most calls take 5 ms. The serialize-and-write step alone can take 10 ms, so every "hit" on disk is slower than just rerunning the function. Smart persistence keeps cheap results in the in-memory tier and only writes to disk (or Redis, or S3) when the compute was expensive enough that re-reading is genuinely faster than recomputing.

The decision applies to **multi-tier backends only**. A single-tier `FileBackend` or `RedisBackend` writes every entry — there is no tier 0 to fall back to. Smart persistence is what makes the default `TieredBackend([RAM, FileBackend])` produce sensible disk usage instead of a flat "save everything that ran" pile.

## The decision in one sentence

If the promotion policy returns `False` for a given call, the result lands in RAM only — the next disk tier (and every tier after) is skipped (`src/cash/backends/tiered_backend.py`). Promote when recomputing the value would cost more than restoring it: `execution_time - est_restore_time > min_cache_savings_pct × execution_time`, where `est_restore_time` is the fitted cost model's end-to-end serialize-write / read-deserialize prediction.

## Quick start

Smart persistence is on by default. Nothing to configure:

```python
import cash

@cash.cache
def fast(x):
    return x + 1            # ~1 µs

@cash.cache
def slow(x):
    time.sleep(2)
    return heavy_thing(x)   # ~2 s

fast(1); fast(1)            # MISS, HIT — both served from RAM, no disk write
slow(1); slow(1)            # MISS, HIT — second call survives kernel restart
```

After both calls, peek at the cache directory: only `slow`'s entry is on disk. `fast` lives in RAM and disappears when the process exits.

## The promotion policy

The active policy is built by `_build_smart_persistence_policy` (`src/cash/backends/factory.py`) and handed to the `TieredBackend` constructor at startup. Its body:

```python
# test:inject: min_persist_compute_s = 0.1
# test:inject: min_savings = 0.20
# test:inject: from cash.notebook import cost_model
def policy(execution_time: float, size_bytes: int) -> bool:
    if execution_time < min_persist_compute_s:        # 0.1 s compute floor
        return False
    # Fitted cost model: predicted seconds to read + deserialize the value.
    # (End-to-end serialize/deserialize, NOT a raw byte-per-second guess.)
    est_restore = cost_model.estimated_restore_time("", size_bytes, "disk")
    # Promote only when recomputing costs more than restoring, by min_savings.
    return execution_time - est_restore > min_savings * execution_time
```

Two things gate the promotion:

1. **Hard floor at 100 ms.** Anything that ran faster than `0.1 s` never reaches disk — the I/O alone would cost more than recomputing.
2. **Restore-vs-recompute check.** Above the floor, Cash predicts how long the value would take to *restore* (`cost_model.estimated_restore_time`, the fitted serialize-write / read-deserialize model) and promotes only when recomputing would cost more, by at least `min_cache_savings_pct` of the compute time. Because the prediction is per-object-size, a **bigger** result that is **expensive** to recompute is now *more* likely to persist — the opposite of the old raw-bandwidth model, which scaled a fake `io_time` with size and left large frames RAM-only.

This 2-argument closure carries no *type*, so it assumes the slowest (`_GENERIC`) family as a conservative floor. When the entry does know its type — every notebook-cached value records its `cost_model_family` on the metadata — `TieredBackend.set` recomputes the same decision with the *real* family, so the two persistence gates (this one and the statement processor's Gate A) agree instead of contradicting each other.

The `100 ms` floor is hardcoded in `factory.py`; the savings fraction is `min_cache_savings_pct` (default `0.20`).

> **Restart implication.** The corollary of the 100 ms floor is that a *fast but
> important* computation on the default `TieredBackend` stays RAM-only and does
> **not** survive a process restart — by design, since re-running it is cheaper
> than the disk round-trip. If you need a sub-100 ms result to persist across
> restarts (e.g. a thin reader process that should always restore), use a
> single-tier persistent backend (`Cash(backend=FileBackend(...))` or
> `SQLiteBackend`), which writes every entry regardless of compute time.

### Worked examples

Walking the policy with concrete values clarifies why each is promoted or kept in RAM. The predicted restore times below come from the conservative `_GENERIC` family the 2-argument closure assumes; a known type (e.g. a numeric DataFrame) predicts lower, so it is even more likely to persist.

| `execution_time` | `size_bytes` | Predicted restore | Outcome |
|---|---|---|---|
| 50 ms | 1 KiB | — (below floor) | RAM only — under the 0.1 s compute floor |
| 50 ms | 100 MiB | — (below floor) | RAM only — same gate; size is irrelevant below the floor |
| 200 ms | 1 MiB | ~0.01 s | Promoted — restoring costs a fraction of the 0.2 s recompute |
| 2 s | 50 MiB | ~0.11 s | Promoted — restore ≪ recompute |
| 2 s | 500 MiB | ~1.05 s | Promoted — restore (1.05 s) still saves well over 20% of 2 s |
| 1 s | 1 GiB | ~2.13 s | RAM only — restoring would cost *more* than recomputing |
| 0.5 s | 2 GiB | ~4.25 s | RAM only — huge but cheap to recompute; recompute wins |

The comparison is *write-now-read-later* vs *recompute-now*: `est_restore_time` is the fitted read-plus-deserialize cost, and a value is promoted only when skipping the recompute saves more than that. Crucially, a large result at high compute (row 5) now persists, where the old `2 × size / 100 MB/s` bandwidth model wrongly refused it.

## Configuration

The policy knobs exposed via `CashConfig` (`src/cash/config.py`):

| Field | Default | Effect |
|---|---|---|
| `smart_persistence` | `True` | Master toggle. When `False`, the `TieredBackend` is built without the cost-model policy and falls back to `_default_promotion_policy` (still serialization-aware, but with a 1.0 s floor). |
| `min_cache_savings_pct` | `0.20` | Required time-savings fraction. A cache hit must save at least this fraction of the compute cost to be worth promoting past RAM. |

Set them via any layer (`pyproject.toml [tool.cash]`, `CASH_*` env vars, or kwargs):

```python
import cash

cash.configure(min_cache_savings_pct=0.10)        # promote when a hit saves >10%
cash.configure(smart_persistence=False)           # fall back to the default policy
```

`cash.configure(smart_persistence=...)` is **not** in the BACKEND_AFFECTING set (`src/cash/__init__.py`), so changing it at runtime updates the dataclass but does not rebuild the active backend's policy closure. To make a runtime change stick, restart the process or reconstruct the `Cash` instance.

## Inspecting where a value actually landed

The `TieredBackend.set` path records which tiers accepted the write in `metadata['storage']` (`src/cash/backends/tiered_backend.py`). This is a list of source labels — `"RAM"`, the file backend's `source_label`, etc. On a hit, `metadata['source']` records which tier served the read (`tiered_backend.py:72-73`).

For debugging, enable verbose logging:

```python
import cash
import logging
logging.basicConfig(level=logging.DEBUG)
cash.configure(debug=True)
```

The TieredBackend logs `[STORAGE] Stored in: RAM` for skipped-disk entries and `[STORAGE] Stored in: RAM, FileBackend` for promoted ones (`tiered_backend.py:139-140`). For per-call introspection use `f.explain(*args, **kwargs)` — it tells you whether the next call would hit and which tier the entry currently lives in (`src/cash/core.py`):

```python
@cash.cache
def slow(x):
    time.sleep(2)
    return x

slow(1)
print(slow.explain(1))
# [HIT] __main__.slow — hit
#   cache_key: slow:...
#   cached_at: 1742813001.4
#   execution_time_saved: 2.001
#   cache_age_seconds: 0.012
```

`explain` returns the `reason` string, the `cache_key`, and a `details` dict (`src/cash/core.py`). To see *which tier* a hit came from, read the metadata directly from the backend or watch the `[STORAGE]` log lines.

> **Note.** The decorator's `f.cache_info()` returns aggregate hit/miss/savings stats and a recent-warnings log — it does *not* expose persist/skip context per call. Inspecting promotion decisions is a debug-log job, not a `cache_info` field.

## The notebook path — the same cost model, one gate earlier

The notebook integration (`%%cash` cells, `%cash_on` magic) applies the **same** fitted cost model, but one step earlier: its Gate A decides whether a statement's output is worth caching *at all* before the value ever reaches the backend. The tier promotion policy uses that same model, so the two gates agree. Gate A lives in `statement/processor.py`; the fitted coefficients live in `src/cash/notebook/cost_model.py` and predict serialize / deserialize wall-time per `(type_family, backend_kind, size_bytes)`. They are re-fittable via:

1. `benchmarks/measure_ser_deser.py` — runs a measurement campaign across families and sizes, writing the matrix to `benchmarks/results/ser_deser_matrix.csv`.
2. `benchmarks/fit_cost_model.py` — fits per-(family, backend, op) `cost = a + b · size_bytes` lines and prints constants ready to paste into the module.

Two further `CashConfig` fields apply **only** to the notebook Gate A (`src/cash/config.py`):

| Field | Default | Effect |
|---|---|---|
| `min_execution_time_to_cache_seconds` | `0.01` | Statements faster than this are not cached at all — no metadata entry is written. |
| `min_cache_fixed_budget_seconds` | `0.05` | Flat floor on restore-time budget. Trivial cells get this much budget regardless of compute, so tiny results aren't refused over a few-ms ratio. |

`min_cache_savings_pct` (above) is shared by Gate A and the tier promotion policy; these two are notebook-only. The plain `@cash.cache` decorator path caches every returned value and relies solely on the `TieredBackend` promotion policy for the RAM-vs-disk decision.

For a deep dive into the notebook filter and its skip-reason taxonomy, see [Cost Model](../../cost-model.md).

## Overriding the decision

Two override mechanisms exist, and they apply to different paths:

- **Notebook `# @cash:persist` annotation.** When a `%%cash` cell carries a `# @cash:persist` comment, the parser sets `force_persist=True` on the entry's metadata (`src/cash/notebook/annotations.py`, `src/cash/notebook/statement/processor.py`). The notebook filter then bypasses its skip checks (`statement/processor.py:1127`), and the `TieredBackend` also reads `metadata['force_persist']` and bypasses its promotion policy (`tiered_backend.py:105-109`). The annotation is the only way to force a single statement past both filters.
- **`smart_persistence=False`.** Disables the policy for every call. Useful for benchmarking, debugging, or workloads where you've measured that the heuristic is wrong on your data.
- **`%cash_persist on` / `cash.configure(persist_all=True)`.** Force-caches *every* statement, bypassing the cost-aware floors globally — the blanket equivalent of putting `# @cash:persist` on all of them. Good for reproducibility and benchmarking; wasteful for trivial statements in normal use.

There is **no** `@cash.cache(persist=True)` decorator parameter. To force persistence of a specific function's results, the available options are: switch to a non-tiered backend (`Cash(backend=FileBackend(...))` writes everything), or lower `min_cache_savings_pct` toward `0` so almost any hit clears the promotion bar. See [Controlling Cache Behavior](controlling-cache-behavior.md) for the full list of decorator knobs.

## Tuning for your workload

- **Many small fast computations.** Defaults are fine — the 100 ms floor drops them from disk before the cost model even runs.
- **Big slow rare computations.** Defaults are fine. Their recompute cost dwarfs the predicted restore, so the cost model almost always promotes them.
- **Medium-cost, medium-size workloads that should land on disk.** If your typical "worth caching" call runs 200–500 ms and you want more of them on disk, lower the savings bar: `cash.configure(min_cache_savings_pct=0.1)`.
- **Notebook cells skipped by the fitted cost model.** That's the statement processor's Gate A, which shares the same rule. Tune `min_cache_savings_pct` (raise to skip more, lower to skip less) or use `# @cash:persist` on the cell. See [Cost Model](../../cost-model.md).
- **All results must land on disk regardless.** Build the backend directly: `Cash(backend=FileBackend(...))` writes every entry.

## TieredBackend interaction

Two decisions are made when a value is set on a `TieredBackend`:

1. **Tier 0 always writes.** Memory tier 0 takes every entry, no policy consulted (`tiered_backend.py:87-95`).
2. **Tiers 1..N consult the promotion policy.** If the policy returns `True` *and* the entry fits under each tier's `max_size_bytes` cap, the entry is set on that tier. A 20 MB entry might land in RAM + DISK but skip a Redis tier with a 10 MB cap (`tiered_backend.py:115-122`).

The promotion policy decides *whether to write past tier 0*. Each tier's `max_size_bytes` then decides *which tiers* among the eligible ones get a copy. The two checks are independent — a `True` from the policy is necessary, not sufficient.

See [Choosing a Backend](choosing-a-backend.md) for how to wire `TieredBackend` stacks and what each tier's cap means in practice.

## Built-in `_default_promotion_policy` fallback

When `smart_persistence=False` (so the factory wires in no cost-model closure), or when a user constructs `TieredBackend(..., promotion_policy=None)` directly, the backend falls back to its own bound method `_default_promotion_policy` (`src/cash/backends/tiered_backend.py`):

```python
def _default_promotion_policy(self, execution_time, size_bytes):
    # Serialization-aware like the smart policy, but with only size_bytes to go
    # on it assumes the slowest (_GENERIC) family, and keeps a 1.0 s floor.
    return self._cost_model_promote(
        "", size_bytes, execution_time, self._promotion_backend_kind()
    )
```

It applies the same restore-vs-recompute rule as the smart policy, just with a 1.0 s compute floor instead of 0.1 s. Whenever an entry's metadata carries a `cost_model_family`, `TieredBackend.set` bypasses both closures and predicts with the real type instead.

## Caveats

- **Heuristic, not optimal.** The policy uses a hardcoded `100 ms` floor and the fitted cost model's coefficients, which were measured on one dev machine's NVMe. On very different storage (a slow network share, a RAM disk) the predictions drift; per-machine recalibration is a planned follow-up. Measure before tuning.
- **First call to a new function.** There's no history-tracking — every call's policy is decided from that call's own `execution_time` and `size_bytes`. Cold-start times that happen to be slow get promoted; cold-start times that happen to be fast (e.g. JIT not warmed up) skip disk and are recomputed on the next process.
- **`size_bytes` comes from the backend's serializer.** A pre-serialization size estimate isn't always accurate for objects that pickle to dramatically different sizes than their in-memory footprint (compressed numpy arrays, sparse matrices, dicts of small primitives).
- **No `@cash.cache(persist=True)` knob.** The only force-persist mechanism is the `# @cash:persist` notebook annotation. If you need to guarantee persistence for a decorator-wrapped function, either disable smart persistence globally or pick a single-tier backend.
- **Runtime `cash.configure(smart_persistence=...)` doesn't rebuild the backend.** It updates the dataclass field but the active `TieredBackend`'s `promotion_policy` closure is set at build time and not reread. Restart the process or reconstruct the `Cash` instance to apply.
- **Don't trust the heuristic blindly for production caches.** If a specific entry's freshness is critical (a tier 0 RAM-only entry disappears on restart), use a single-tier persistent backend or a `# @cash:persist` annotation.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `smart_persistence` | `CashConfig` field / `CASH_SMART_PERSISTENCE` | Master toggle. Default `True`. False falls back to `_default_promotion_policy`. |
| `min_cache_savings_pct` | `CashConfig` field | Required savings fraction for promotion — used by **both** the tier policy and the notebook Gate A. Default `0.20`. |
| `min_execution_time_to_cache_seconds` | `CashConfig` field | **Notebook path only.** Per-statement floor. Default `0.01 s`. |
| `min_cache_fixed_budget_seconds` | `CashConfig` field | **Notebook path only.** Flat restore-time budget floor. Default `0.05 s`. |
| `_build_smart_persistence_policy` | Internal (`src/cash/backends/factory.py`) | Returns the cost-model closure that `TieredBackend` calls on each set. |
| `TieredBackend.promotion_policy` | `Callable[(float, int), bool]` | Per-`set` gate that decides whether to write past tier 0. Replaceable via constructor `promotion_policy=` kwarg. |
| `TieredBackend._default_promotion_policy` | Internal (`src/cash/backends/tiered_backend.py`) | Fallback policy when `promotion_policy=None` is passed to the constructor. |
| `metadata['force_persist']` | Backend metadata | Set by `# @cash:persist` notebook annotation. Bypasses the policy. |
| `metadata['cost_model_family']` / `['cost_model_size_bytes']` | Backend metadata | Written by the statement processor; let `TieredBackend.set` predict restore time with the real type. |
| `metadata['storage']` | Backend metadata (list[str]) | Records which tiers accepted the write — `["RAM"]`, `["RAM", "FileBackend"]`, etc. |
| `cost_model.estimated_serialize_time` / `estimated_restore_time` | `src/cash/notebook/cost_model.py` | Fitted predictions used by **both** the notebook Gate A and the tier promotion policy. |

## Related

- [Choosing a Backend](choosing-a-backend.md) — what a `TieredBackend` is, what each tier's `max_size_bytes` means, and when to skip tiered entirely.
- [Cost Model](../../cost-model.md) — the notebook-only second filter, its fitted coefficients, and the skip-reason taxonomy you see in cell badges.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — every `@cash.cache` knob (TTL, `cache_if`, `file_depends_on`, etc.) and how they interact with the promotion policy.
- [Configuration](../../getting-started/configuration.md) — the full `CashConfig` field table and env-var bindings.
