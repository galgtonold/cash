# Smart persistence — when caching to disk would cost more than it saves

Caching to disk costs I/O. For very cheap computations, the disk-write itself takes longer than rerunning the function. Cash's smart-persistence policy decides per call whether a result is worth promoting past RAM — using the call's measured execution time and the result's size as inputs.

## Why this exists

A naive "cache everything to disk" backend hurts when most calls take 5 ms. The serialize-and-write step alone can take 10 ms, so every "hit" on disk is slower than just rerunning the function. Smart persistence keeps cheap results in the in-memory tier and only writes to disk (or Redis, or S3) when the compute was expensive enough that re-reading is genuinely faster than recomputing.

The decision applies to **multi-tier backends only**. A single-tier `FileBackend` or `RedisBackend` writes every entry — there is no tier 0 to fall back to. Smart persistence is what makes the default `TieredBackend([RAM, FileBackend])` produce sensible disk usage instead of a flat "save everything that ran" pile.

## The decision in one sentence

If the promotion policy returns `False` for a given call, the result lands in RAM only — the next disk tier (and every tier after) is skipped (`src/cash/backends/tiered_backend.py:97-114`). Promote when `execution_time > read_time` *and* the call is heavy enough that the disk round-trip is worth saving.

## Quick start

Smart persistence is on by default. Nothing to configure:

<!-- test:skip reason="time.sleep(2) and undefined heavy_thing() make this unsuitable for automated testing" -->
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

The active policy is built by `_build_smart_persistence_policy` (`src/cash/backends/factory.py:109-125`) and handed to the `TieredBackend` constructor at startup. Its body:

<!-- test:skip reason="references undefined local variables min_persist_compute_s, small_result_bytes, threshold from surrounding factory closure" -->
```python
def policy(execution_time: float, size_bytes: int) -> bool:
    if execution_time < min_persist_compute_s:   # 0.1 s
        return False
    if size_bytes < small_result_bytes:          # 64 KiB
        return True
    if execution_time < threshold:               # smart_persistence_threshold, default 1.0 s
        return False
    disk_bandwidth = 100 * 1024 * 1024           # 100 MB/s
    io_time = (size_bytes / disk_bandwidth) * 2
    return execution_time > io_time
```

Three thresholds gate the promotion in order:

1. **Hard floor at 100 ms.** Anything that ran faster than `0.1 s` never reaches disk — the I/O alone would cost more than recomputing.
2. **Small-result fast path at 64 KiB.** Results under 64 KiB are cheap to write, so the moment compute clears the 100 ms floor they get promoted, regardless of compute cost above that floor.
3. **Bandwidth-vs-compute check.** Above 64 KiB and below `smart_persistence_threshold` (default 1.0 s), the call is skipped. Above the threshold, Cash compares `execution_time` against an estimated disk round-trip (`2 × size / 100 MB/s`) and promotes only if compute exceeds I/O.

The `100 ms` floor and `64 KiB` cutoff are hardcoded in `factory.py` — they are not config-driven. The only configurable knob on the decorator path is `smart_persistence_threshold`.

### Worked examples

Walking the policy with concrete values clarifies which gate fires for which workload:

| `execution_time` | `size_bytes` | Path | Outcome |
|---|---|---|---|
| 50 ms | 1 KiB | floor (`< 0.1 s`) | RAM only — cheap recompute beats any I/O |
| 50 ms | 100 MiB | floor (`< 0.1 s`) | RAM only — same gate, large size doesn't matter |
| 200 ms | 1 KiB | small-result fast path (`< 64 KiB`) | Promoted — tiny writes are cheap regardless of compute |
| 200 ms | 1 MiB | threshold (`< 1.0 s`, size ≥ 64 KiB) | RAM only — compute below the bandwidth-check threshold |
| 2 s | 50 MiB | bandwidth check: `2 × 50/100 = 1 s < 2 s` | Promoted — compute exceeds modelled I/O |
| 2 s | 500 MiB | bandwidth check: `2 × 500/100 = 10 s > 2 s` | RAM only — modelled I/O exceeds compute |
| 10 s | 500 MiB | bandwidth check: `10 s > 10 s` is false | RAM only — borderline; raise threshold or move to single-tier backend |

The bandwidth check uses `2 × size / 100 MB/s` rather than a single round-trip because the relevant comparison is *write-now-and-read-later* vs *recompute-now*. Both halves of the round-trip cost real wall-time eventually.

## Configuration

Two policy knobs are exposed via `CashConfig` (`src/cash/config.py:177-187`):

| Field | Default | Effect |
|---|---|---|
| `smart_persistence` | `True` | Master toggle. When `False`, the `TieredBackend` is built without a promotion policy — every set writes to every tier. |
| `smart_persistence_threshold` | `1.0` (seconds) | Compute time above which the bandwidth-vs-compute check is applied. Below this, sub-64-KiB results still get promoted; above it, large results need to beat the modelled disk round-trip. |

Set them via any layer (`pyproject.toml [tool.cash]`, `CASH_*` env vars, or kwargs):

```python
import cash

cash.configure(smart_persistence_threshold=0.5)   # promote anything >0.5 s
cash.configure(smart_persistence=False)           # force-disable; persist everything
```

`cash.configure(smart_persistence=...)` is **not** in the BACKEND_AFFECTING set (`src/cash/__init__.py:219-224`), so changing it at runtime updates the dataclass but does not rebuild the active backend's policy closure. To make a runtime change stick, restart the process or reconstruct the `Cash` instance.

## Inspecting where a value actually landed

The `TieredBackend.set` path records which tiers accepted the write in `metadata['storage']` (`src/cash/backends/tiered_backend.py:130-136`). This is a list of source labels — `"RAM"`, the file backend's `source_label`, etc. On a hit, `metadata['source']` records which tier served the read (`tiered_backend.py:72-73`).

For debugging, enable verbose logging:

```python
import cash
import logging
logging.basicConfig(level=logging.DEBUG)
cash.configure(debug=True)
```

The TieredBackend logs `[STORAGE] Stored in: RAM` for skipped-disk entries and `[STORAGE] Stored in: RAM, FileBackend` for promoted ones (`tiered_backend.py:139-140`). For per-call introspection use `f.explain(*args, **kwargs)` — it tells you whether the next call would hit and which tier the entry currently lives in (`src/cash/core.py:1470-1480`):

<!-- test:skip reason="time.sleep(2) makes this unsuitable for automated testing" -->
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

`explain` returns the `reason` string, the `cache_key`, and a `details` dict (`src/cash/core.py:65-93`). To see *which tier* a hit came from, read the metadata directly from the backend or watch the `[STORAGE]` log lines.

> **Note.** The decorator's `f.cache_info()` returns aggregate hit/miss/savings stats and a recent-warnings log — it does *not* expose persist/skip context per call. Inspecting promotion decisions is a debug-log job, not a `cache_info` field.

## The notebook path — a different cost model

The notebook integration (`%%cash` cells, `%cash_on` magic) uses its **own** filter on top of the tiered policy described above. It is implemented in `statement_processor.py` and uses a fitted linear-regression cost model that predicts serialize / deserialize wall-time per `(type_family, backend_kind, size_bytes)`. The fitted coefficients live in `src/cash/notebook/cost_model.py:33-124` and are re-fittable via:

1. `benchmarks/measure_ser_deser.py` — runs a measurement campaign across families and sizes, writing the matrix to `benchmarks/results/ser_deser_matrix.csv`.
2. `benchmarks/fit_cost_model.py` — fits per-(family, backend, op) `cost = a + b · size_bytes` lines and prints constants ready to paste into the module.

Three additional `CashConfig` fields control the notebook filter (`src/cash/config.py:189-202`):

| Field | Default | Effect |
|---|---|---|
| `min_execution_time_to_cache_seconds` | `0.01` | Statements faster than this are not cached at all — no metadata entry is written (`statement_processor.py:1253`). |
| `min_cache_savings_pct` | `0.20` | Skip caching when predicted savings drop below 20 % of compute (`statement_processor.py:1098`). |
| `min_cache_fixed_budget_seconds` | `0.05` | Flat floor on restore-time budget. Trivial cells get this much budget regardless of compute, so tiny results aren't refused over a few-ms ratio (`statement_processor.py:1099`). |

These three fields only apply to the notebook path. The plain `@cash.cache` decorator path uses the `TieredBackend` promotion policy described above and ignores them.

For a deep dive into the notebook filter and its skip-reason taxonomy, see [Cost Model](../../cost-model.md).

## Overriding the decision

Two override mechanisms exist, and they apply to different paths:

- **Notebook `# @cash:persist` annotation.** When a `%%cash` cell carries a `# @cash:persist` comment, the parser sets `force_persist=True` on the entry's metadata (`src/cash/notebook/annotations.py:33-60`, `src/cash/notebook/statement_processor.py:586-595`). The notebook filter then bypasses its skip checks (`statement_processor.py:1127`), and the `TieredBackend` also reads `metadata['force_persist']` and bypasses its promotion policy (`tiered_backend.py:105-109`). The annotation is the only way to force a single statement past both filters.
- **`smart_persistence=False`.** Disables the policy for every call. Useful for benchmarking, debugging, or workloads where you've measured that the heuristic is wrong on your data.

There is **no** `@cash.cache(persist=True)` decorator parameter. To force persistence of a specific function's results, the available options are: switch to a non-tiered backend (`Cash(backend=FileBackend(...))` writes everything), or lower `smart_persistence_threshold` until the call clears it. See [Controlling Cache Behavior](controlling-cache-behavior.md) for the full list of decorator knobs.

## Tuning for your workload

- **Many small fast computations.** Defaults are fine — the 100 ms floor already drops them from disk, and the 64 KiB fast path catches anything that crosses the floor.
- **Big slow rare computations.** Defaults are fine. The bandwidth-vs-compute check almost always promotes them.
- **Medium-cost, medium-size workloads that should land on disk.** The default `smart_persistence_threshold = 1.0 s` is conservative. If your typical "worth caching" call runs 200–500 ms, lower the threshold: `cash.configure(smart_persistence_threshold=0.2)`.
- **Notebook cells skipped by the fitted cost model.** That's filter 1, not the promotion policy. Tune `min_cache_savings_pct` (raise to skip more, lower to skip less) or use `# @cash:persist` on the cell. See [Cost Model](../../cost-model.md).
- **All results must land on disk regardless.** `cash.configure(smart_persistence=False)` at startup, or build the backend directly: `Cash(backend=FileBackend(...))`.

## TieredBackend interaction

Two decisions are made when a value is set on a `TieredBackend`:

1. **Tier 0 always writes.** Memory tier 0 takes every entry, no policy consulted (`tiered_backend.py:87-95`).
2. **Tiers 1..N consult the promotion policy.** If the policy returns `True` *and* the entry fits under each tier's `max_size_bytes` cap, the entry is set on that tier. A 20 MB entry might land in RAM + DISK but skip a Redis tier with a 10 MB cap (`tiered_backend.py:115-122`).

The promotion policy decides *whether to write past tier 0*. Each tier's `max_size_bytes` then decides *which tiers* among the eligible ones get a copy. The two checks are independent — a `True` from the policy is necessary, not sufficient.

See [Choosing a Backend](choosing-a-backend.md) for how to wire `TieredBackend` stacks and what each tier's cap means in practice.

## Built-in `_default_promotion_policy` fallback

If `_build_smart_persistence_policy` returns `None` (only happens when `smart_persistence=False`, in which case no policy is wired in), the `TieredBackend` falls back to its own bound method `_default_promotion_policy` (`src/cash/backends/tiered_backend.py:33-47`):

```python
def _default_promotion_policy(self, execution_time, size_bytes):
    if execution_time < 1.0:
        return False
    read_time = size_bytes / self._disk_bandwidth_est   # 100 MB/s
    return execution_time > read_time
```

This is a simpler version of the same logic. It's the path taken when a user passes their own `TieredBackend(..., promotion_policy=None)` directly without going through the factory. The factory path (`smart_persistence=True`, default) replaces it with the more nuanced threshold-aware policy.

## Caveats

- **Heuristic, not optimal.** The policy uses two hardcoded constants (`100 ms` floor, `64 KiB` small-result cutoff) and a fixed 100 MB/s disk bandwidth estimate. On NVMe (~3 GB/s) the bandwidth check is conservative; on a slow network share it's optimistic. Measure before tuning.
- **First call to a new function.** There's no history-tracking — every call's policy is decided from that call's own `execution_time` and `size_bytes`. Cold-start times that happen to be slow get promoted; cold-start times that happen to be fast (e.g. JIT not warmed up) skip disk and are recomputed on the next process.
- **`size_bytes` comes from the backend's serializer.** A pre-serialization size estimate isn't always accurate for objects that pickle to dramatically different sizes than their in-memory footprint (compressed numpy arrays, sparse matrices, dicts of small primitives).
- **No `@cash.cache(persist=True)` knob.** The only force-persist mechanism is the `# @cash:persist` notebook annotation. If you need to guarantee persistence for a decorator-wrapped function, either disable smart persistence globally or pick a single-tier backend.
- **Runtime `cash.configure(smart_persistence=...)` doesn't rebuild the backend.** It updates the dataclass field but the active `TieredBackend`'s `promotion_policy` closure is set at build time and not reread. Restart the process or reconstruct the `Cash` instance to apply.
- **Don't trust the heuristic blindly for production caches.** If a specific entry's freshness is critical (a tier 0 RAM-only entry disappears on restart), use a single-tier persistent backend or a `# @cash:persist` annotation.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `smart_persistence` | `CashConfig` field / `CASH_SMART_PERSISTENCE` | Master toggle. Default `True`. False disables policy entirely. |
| `smart_persistence_threshold` | `CashConfig` field / `CASH_SMART_PERSISTENCE_THRESHOLD` | Compute time (seconds) above which the bandwidth check applies. Default `1.0`. |
| `min_execution_time_to_cache_seconds` | `CashConfig` field | **Notebook path only.** Per-statement floor. Default `0.01 s`. |
| `min_cache_savings_pct` | `CashConfig` field | **Notebook path only.** Minimum predicted savings ratio. Default `0.20`. |
| `min_cache_fixed_budget_seconds` | `CashConfig` field | **Notebook path only.** Flat restore-time budget floor. Default `0.05 s`. |
| `_build_smart_persistence_policy` | Internal (`src/cash/backends/factory.py:109`) | Returns the closure that `TieredBackend` calls on each set. |
| `TieredBackend.promotion_policy` | `Callable[(float, int), bool]` | Per-`set` gate that decides whether to write past tier 0. Replaceable via constructor `promotion_policy=` kwarg. |
| `TieredBackend._default_promotion_policy` | Internal (`src/cash/backends/tiered_backend.py:33`) | Fallback policy when `promotion_policy=None` is passed to the constructor. |
| `metadata['force_persist']` | Backend metadata | Set by `# @cash:persist` notebook annotation. Bypasses the policy. |
| `metadata['storage']` | Backend metadata (list[str]) | Records which tiers accepted the write — `["RAM"]`, `["RAM", "FileBackend"]`, etc. |
| `cost_model.estimated_serialize_time` / `estimated_restore_time` | `src/cash/notebook/cost_model.py:142-169` | Fitted predictions used by the notebook filter, not by the decorator's tier policy. |

## Related

- [Choosing a Backend](choosing-a-backend.md) — what a `TieredBackend` is, what each tier's `max_size_bytes` means, and when to skip tiered entirely.
- [Cost Model](../../cost-model.md) — the notebook-only second filter, its fitted coefficients, and the skip-reason taxonomy you see in cell badges.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — every `@cash.cache` knob (TTL, `cache_if`, `file_depends_on`, etc.) and how they interact with the promotion policy.
- [Configuration](../../getting-started/configuration.md) — the full `CashConfig` field table and env-var bindings.
