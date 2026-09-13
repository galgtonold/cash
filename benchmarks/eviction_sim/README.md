# Eviction simulator

A trace-driven simulator for comparing cache eviction policies on
notebook-shaped workloads. The traces are synthetic, calibrated where data
exists. It answers "which ranking should the RAM and disk tiers use?" without
waiting for real users to fill real caches.

No recorded trace holds keys, sizes and compute times together, and evictions
are not logged, so the workload is generated. See *Limits* before trusting a
percentage.

| file | what it is |
|---|---|
| `workload.py` | The notebook-project generator (`Project`, `Params`, `session_ops`) and the `Engine` that turns user actions into cache requests the way cash does |
| `policies.py` | ~20 byte-capacity policies: models of cash's shipped RAM and disk eviction, the classic recency/frequency family, cost-aware policies and structure-aware policies |
| `run.py` | Sweeps policies × caps × seeds and prints the *lost savings* table |
| `calibrate.py` | Re-derives the size/compute distributions from an overhead sweep |
| `defect_disk_crumb_order.py` | Real `FileBackend`: hot small entries evicted ahead of stale large ones |
| `defect_ram_tier.py` | Real `InMemoryBackend`: an oversized write empties the tier; host-wide pressure empties the tier |

## Running

```bash
python benchmarks/eviction_sim/run.py --tier disk --seeds 5 -q
python benchmarks/eviction_sim/run.py --tier ram -q
python benchmarks/eviction_sim/run.py --policies cash-now,LRU,GDSF,Supersede+CostLRU --set p_undo=0.4
python benchmarks/eviction_sim/run.py --list
```

A single simulation takes about 1–3 s. The full disk sweep (all policies, 5 caps, 5 seeds) takes about 20 min on one core. Run archetypes as separate processes to use more cores. `--out benchmarks/results_eviction/<name>.json` keeps the per-seed numbers; `benchmarks/results_*` is gitignored.

## The model

**Workload.** Four notebooks share one cache directory. Each has 15–35 cells, and each statement reads a few earlier statements' outputs. A statement's key is `hash(slot, code version, input keys)`, which is cash's lineage rule. So an edit re-keys the edited statement and everything downstream of it.

Every simulated day starts with a restart and then either a run-all or a jump to the working cell. After that come 15–60 edits near the notebook's frontier. 15% of edits are undos, and some re-run every cell below. Notebooks grow, and the active notebook sometimes switches. Loop cells produce one entry per iteration.

**Engine.** It follows what `TieredBackend`, the statement processor and the upstream checker do:
- Running a cell looks up every statement, even when the kernel already holds its value.
- A miss after a restart fetches its inputs, cascading upstream.
- RAM is always written, subject to Gate 0 (10 ms) and Gate A.
- Disk only gets past the 0.1 s floor plus the 20%-savings cost-model gate.
- A disk hit is promoted into RAM without a size gate.

**Metric.** Lost savings of the tier under test is `(T_policy − T_unlimited) / (T_absent − T_unlimited)`. 0% is as good as an infinite tier. 100% is as bad as not having it.

Caps are multiples of the *live set*: the bytes a run-all of every notebook's current source would request. That is the only cap scale that transfers between users. The `disk` experiment holds RAM at cash's current policy and 0.25× its live set. The `ram` experiment makes disk unlimited.

**Calibration.** Compute times and sizes are fitted to an overhead sweep of the notebooks in `ref_notebooks.txt` (see the comment on `Params`).

## Findings (2026-09-13, v2 calibration)

Disk tier, typical workload, 5 seeds, lost savings at a cap of *n* × the live set. The other archetypes preserve the ordering: big data (10× sizes), ML-heavy (8× compute), 120 days × 8 notebooks, and 40% undo. The recommended policy loses 0.5–1.6% at 1× and 0.0–0.1% at 2× on all five workloads.

| policy | 0.5× | 1× | 2× | 4× |
|---|---|---|---|---|
| `LRU` | 37.2% | 12.9% | 7.5% | 3.3% |
| recency/frequency family (FIFO, LFU, SLRU, ARC, S3-FIFO, SIEVE, S/B/DRRIP) | 36–60% | 13–46% | 7–38% | 3–27% |
| `cash-now` (shipped disk policy) | 22.5% | 6.7% | **18.0%** | **15.4%** |
| `GDSF` | 16.6% | 3.9% | 1.7% | 0.5% |
| `Supersede+GDSF-noage` (recommended) | 10.7% | 1.6% | 0.1% | 0.0% |

- **Recency and frequency tricks do not help.** None of the family beats LRU meaningfully, and LFU, SLRU, SIEVE and BRRIP are much worse. Notebook traffic is not scan-shaped.
- **Value density (seconds saved per byte) is the first lever.** Among real disk-eligible results it spans ~7 orders of magnitude (p10 0.0004 to p90 2,600 s/MB) and is uncorrelated with size (log-corr 0.02). GDSF beats LRU at every cap and needs no tuning. `CostLRU` (recency shifted by log value density) is sensitive to τ: 5.9% at 1× with τ = 500, 16.4% with τ = 8000.
- **Generation is the second lever.** Take first the entries whose statement slot has since been written with a newer key, but keep the newest superseded generation (`k=1`) for undo. At 1×, `k=0` loses 9.3% and `k=2` loses 3.0%, against 1.6% for `k=1`.
- **Evicting a dead entry must not age the live ones.** With GDSF inside the dead-first rule, raising `L` on dead evictions costs 6.2% at 1× (`Supersede+GDSF`), against 1.6% without it (`-noage`).
- **GDSF has to rank exactly; sampling does not work.** Redis-style sampled eviction evicts the lowest of *k* random entries, which would let the disk tier read *k* headers per eviction instead of all of them. It gives back most of the gain: at 1× the live set, `GDSF-s32` loses 9.5% (typical) and 21.0% (ML-heavy), against 3.9% and 4.5% for exact GDSF. ML-heavy is worse than LRU's 19.6%. GDSF's clock only ages correctly when it evicts the true minimum. Under the generation rule, sampling hurts less (2.9% vs 1.6%), but it is still a regression.
- **Live-set GC** means removing what a run-all of the current sources would not request. It adds no hits over the generation signal: `Hybrid` scores exactly like `Supersede` in every run. Its value is disk hygiene, since an uncapped cache grows to 21–44× its live set in 40 days.
- **The shipped crumb split is non-monotonic in the cap.** It beats LRU at ≤ 1×, where it acts as crude size-awareness. At ≥ 2× it is much worse, because the crumb threshold grows with the cap and shreds the hot working set. Default caps (8–100 GiB) put real caches in that regime. `defect_disk_crumb_order.py` is the deterministic version.

The first published sweep used an uncoupled v1 size model, whose expensive results were mostly tiny. It showed the crumb split beating LRU at every cap. The recalibration reversed that, and nothing else. The v1 parameters are in the comment on `Params`.

## Limits

- **Edit behaviour is guessed, not measured:** the edit rate, undo rate and run-below probability. Treat the *ordering* of policies as the finding and the percentages as indicative. The first thing to change is recording real get/put/evict traces and replaying them through `Engine`.
- **Not modelled:** multi-process writers, file-dependency invalidation, TTL, metadata-only entries, partial loop re-execution, the RAM tier's psutil pressure path.
- **No offline optimum.** A miss changes which upstream entries are requested next, so Belady needs a fixed trace this model does not have.
- **A slightly negative RAM-tier loss is real in the model, not a bug.** Gate A's fixed budget (`max(0.05 s, 0.8 × compute)`) admits entries whose RAM copy costs more than recomputing them, e.g. 10 ms of compute and a 30 ms copy. The unlimited baseline keeps them. A value-aware policy scores them negative and evicts them first, so it can beat "unlimited" by a hair. Traced under the v1 calibration: −0.1%, 17 fewer RAM hits, +0.19 s compute, −0.51 s restore. The RAM copy cost here is `0.3 ms + size / 2 GB/s`, a guess, not cash's fitted `cost_model`.
- **`CashCurrent` and `CashRAM` are models written from reading the backends.** Re-check them against `FileBackend._check_and_evict` / `InMemoryBackend._evict_to_byte_cap` after changing either.
