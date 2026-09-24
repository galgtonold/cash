# Core integration set

Baseline: commit `56f6516eca` on Windows-10-10.0.26200-SP0, 32 CPUs, wall clock 8 min.

- Tests run: 3971 (passed 3945, skipped 26)
- Passed after a rerun (flaky here): 1
- Core set: **342 tests**, 13.6 min of test time vs 128.8 min for the full suite (11%)
- Passing tests with no coverage recorded (labelling failed?): 0

| Element | In suite | Covered by core set |
|---|---|---|
| function-body lines | 22921 | 22921 |
| features | 27 | 27 |
| feature pairs | 329 | 329 |
| step sequences | 163 | 163 |
| feature x disruptive step | 147 | 147 |

## Tests touching each feature

- randomness: 3125
- badges: 2895
- call_caching: 2470
- miss_guard: 2019
- functions: 1881
- cacheability: 1794
- mismatch: 1568
- file_deps: 1322
- decorator: 989
- disk_tier: 866
- reexecution: 804
- tiers: 773
- effects: 682
- derivation_edges: 455
- annotations: 437
- loops: 425
- restore: 400
- branches: 336
- stateful_carriers: 236
- versions: 198
- try_blocks: 130
- cost_model: 118
- consumables: 114
- modules: 112
- decorator_purity: 51
- remote_sources: 20
- cell_sources: 18

## Needed a rerun

- `tests/test_notebook_integration/workflows/test_replay_acceptance.py::test_replay_matches_a_plain_run[sales:corrected_file+restart->c10->c9]`
