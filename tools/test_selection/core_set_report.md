# Core integration set

Baseline: commit `489f9e8cbf` on Windows-10-10.0.26200-SP0, 32 CPUs, wall clock 8 min.

- Tests run: 3962 (passed 3936, skipped 26)
- Passed after a rerun (flaky here): 0
- Core set: **344 tests**, 15.5 min of test time vs 138.2 min for the full suite (11%)
- Passing tests with no coverage recorded (labelling failed?): 0

| Element | In suite | Covered by core set |
|---|---|---|
| function-body lines | 22654 | 22654 |
| features | 29 | 29 |
| feature pairs | 370 | 370 |
| step sequences | 161 | 161 |
| feature x disruptive step | 151 | 151 |

## Tests touching each feature

- randomness: 3121
- badges: 2891
- call_caching: 2468
- disk_tier: 2280
- tiers: 2150
- miss_guard: 2014
- versions: 1923
- functions: 1875
- cacheability: 1794
- file_deps: 1571
- mismatch: 1568
- reexecution: 1298
- derivation_edges: 460
- loops: 429
- restore: 395
- annotations: 378
- branches: 336
- stateful_carriers: 242
- effects: 187
- try_blocks: 129
- cost_model: 119
- consumables: 115
- modules: 112
- decorator: 80
- decorator_purity: 50
- remote_sources: 20
- cell_sources: 18
- value_policy: 12
- purity: 4
