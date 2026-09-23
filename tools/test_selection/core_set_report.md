# Core integration set

Baseline: commit `c211df3c01` on Linux-6.18.44-fc-v37-x86_64-with-glibc2.39, 4 CPUs, wall clock 35 min.

- Tests run: 4534 (passed 4507, skipped 27)
- Passed after a rerun (flaky here): 0
- Core set: **339 tests**, 12.2 min of test time vs 139.5 min for the full suite (9%)
- Passing tests with no coverage recorded (labelling failed?): 0

| Element | In suite | Covered by core set |
|---|---|---|
| function-body lines | 23833 | 23833 |
| features | 28 | 28 |
| feature pairs | 333 | 333 |
| step sequences | 159 | 159 |
| feature x disruptive step | 147 | 147 |

## Tests touching each feature

- randomness: 3070
- badges: 3055
- call_caching: 2574
- derivation_edges: 2360
- functions: 2224
- cacheability: 2079
- miss_guard: 2029
- file_deps: 1753
- mismatch: 1634
- reexecution: 1421
- decorator: 1093
- tiers: 852
- stateful_carriers: 671
- disk_tier: 660
- annotations: 524
- loops: 489
- restore: 397
- cost_model: 253
- modules: 233
- versions: 197
- try_blocks: 147
- branches: 126
- consumables: 119
- effects: 52
- decorator_purity: 52
- purity: 4
- cell_sources: 2
- remote_sources: 2
