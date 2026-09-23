# Core integration set

Baseline: commit `d791fa06df` on Windows-10-10.0.26200-SP0, 32 CPUs, wall clock 8 min.

- Tests run: 4544 (passed 4518, skipped 26)
- Passed after a rerun (flaky here): 0
- Core set: **347 tests**, 14.5 min of test time vs 144.1 min for the full suite (10%)
- Passing tests with no coverage recorded (labelling failed?): 0

| Element | In suite | Covered by core set |
|---|---|---|
| function-body lines | 22720 | 22720 |
| features | 28 | 28 |
| feature pairs | 347 | 347 |
| step sequences | 161 | 161 |
| feature x disruptive step | 147 | 147 |

## Tests touching each feature

- randomness: 3607
- badges: 3281
- call_caching: 2818
- functions: 2146
- cacheability: 2083
- miss_guard: 2021
- file_deps: 1725
- mismatch: 1701
- reexecution: 1417
- disk_tier: 906
- tiers: 776
- annotations: 525
- loops: 492
- restore: 396
- derivation_edges: 361
- branches: 350
- stateful_carriers: 241
- versions: 202
- effects: 186
- try_blocks: 146
- cost_model: 120
- consumables: 120
- modules: 112
- decorator: 78
- decorator_purity: 52
- remote_sources: 20
- cell_sources: 18
- purity: 4
