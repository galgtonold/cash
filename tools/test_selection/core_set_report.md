# Core integration set

Baseline: commit `99e873efe6` on Linux-6.18.44-fc-v37-x86_64-with-glibc2.39, 4 CPUs, wall clock 66 min.

- Tests run: 4534 (failed 34, passed 4474, skipped 26)
- Passed after a rerun (flaky here): 1
- Core set: **337 tests**, 15.2 min of test time vs 248.7 min for the full suite (6%)
- Passing tests with no coverage recorded (labelling failed?): 0

| Element | In suite | Covered by core set |
|---|---|---|
| function-body lines | 22755 | 22755 |
| features | 29 | 29 |
| feature pairs | 363 | 363 |
| step sequences | 158 | 158 |
| feature x disruptive step | 153 | 153 |

## Tests touching each feature

- badges: 3179
- randomness: 3045
- functions: 2776
- call_caching: 2543
- tiers: 2512
- derivation_edges: 2218
- cacheability: 2064
- miss_guard: 2012
- file_deps: 1807
- mismatch: 1612
- reexecution: 1400
- decorator: 1066
- purity: 927
- stateful_carriers: 664
- disk_tier: 652
- annotations: 531
- loops: 493
- restore: 386
- cost_model: 252
- modules: 228
- versions: 203
- try_blocks: 146
- branches: 125
- consumables: 119
- effects: 67
- decorator_purity: 51
- value_policy: 11
- cell_sources: 3
- remote_sources: 2

## Failed or errored in the baseline

- `tests/test_notebook_integration/test_a_file_read_by_its_size_is_a_dependency.py::test_a_size_shown_after_the_file_changed_is_the_new_one[os_path_getsize]`
- `tests/test_notebook_integration/test_a_file_read_by_its_size_is_a_dependency.py::test_a_size_shown_after_the_file_changed_is_the_new_one[path_stat]`
- `tests/test_notebook_integration/test_a_module_is_its_code_not_its_formatting.py::test_a_comment_added_to_a_module_does_not_re_run_its_callers`
- `tests/test_notebook_integration/test_a_module_is_its_code_not_its_formatting.py::test_a_docstring_reworded_in_a_module_does_not_re_run_its_callers`
- `tests/test_notebook_integration/test_a_module_is_its_code_not_its_formatting.py::test_the_same_through_an_alias`
- `tests/test_notebook_integration/test_a_statement_depends_on_the_symbols_it_reads.py::TestAnUnrelatedEditIsFree::test_and_nothing_built_on_it_re_runs_either`
- `tests/test_notebook_integration/test_a_statement_depends_on_the_symbols_it_reads.py::TestAnUnrelatedEditIsFree::test_editing_another_function_does_not_re_run_this_one[import symlib as sl-sl]`
- `tests/test_notebook_integration/test_a_statement_depends_on_the_symbols_it_reads.py::TestAnUnrelatedEditIsFree::test_editing_another_function_does_not_re_run_this_one[import symlib-symlib]`
- `tests/test_notebook_integration/test_global_savefig_guard.py::test_healthy_plt_savefig_writes_the_real_chart_and_does_not_refuse`
- `tests/test_notebook_integration/test_global_savefig_guard.py::test_orphaned_plt_savefig_is_refused_not_blanked`
- `tests/test_notebook_integration/test_interaction_async_edit.py::TestAsyncPatterns::test_async_function_edit`
- `tests/test_notebook_integration/test_interaction_async_edit.py::TestAsyncPatterns::test_async_gather_edit`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestAsyncBasics::test_async_function_change_propagates`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestAsyncBasics::test_async_function_definition_and_call`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestAsyncBasics::test_async_with_gather`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestAsyncBasics::test_multiple_async_functions`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestAsyncGenerators::test_async_context_manager`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestAsyncGenerators::test_async_generator_collected`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestCoroutinePatterns::test_async_exception_handling`
- `tests/test_notebook_integration/test_round3_async_patterns.py::TestCoroutinePatterns::test_coroutine_result_cached`
- `tests/test_notebook_integration/test_stress_batch5_devious.py::TestReexecutionPatterns::test_149_delete_variable_then_use_raises`
- `tests/test_notebook_integration/test_toplevel_await_caching.py::test_toplevel_await_result_is_cached_second_run`
- `tests/test_notebook_integration/test_toplevel_await_caching.py::test_toplevel_await_selfmod_stays_idempotent`
- `tests/test_notebook_integration/test_toplevel_await_lineage.py::test_async_def_edit_then_isolated_rerun_of_await_cell_picks_up_new_body`
- `tests/test_notebook_integration/test_toplevel_await_lineage.py::test_toplevel_await_selfmod_isolated_rerun_is_idempotent`
- `tests/test_notebook_integration/test_upstream_restore_is_cached.py::test_the_expensive_upstream_statement_restores_rather_than_re_running`
- `tests/test_notebook_integration/test_upstream_restore_is_cached.py::test_the_restored_upstream_row_reports_a_real_saving`
- `tests/test_notebook_integration/test_what_a_reload_computed_restores_after_a_restart.py::test_an_unrelated_edit_to_a_timed_helper_re_runs_nothing[aliased]`
- `tests/test_notebook_integration/test_zzprobe_async.py::test_async_def_edit_then_isolated_rerun_of_await_cell`
- `tests/test_notebook_integration/test_zzprobe_async.py::test_await_cell_edit_invalidates_downstream`
- `tests/test_notebook_integration/test_zzprobe_async.py::test_toplevel_await_cache_hit_second_run`
- `tests/test_notebook_integration/test_zzprobe_async.py::test_toplevel_await_selfmod_isolated_rerun_idempotent`
- `tests/test_notebook_integration/test_zzprobe_syntax.py::test_annassign_and_type_alias_edit_invalidation`
- `tests/test_notebook_integration/test_zzprobe_wave5.py::test_decorator_disk_hit_survives_restart`

## Needed a rerun

- `tests/test_notebook_integration/test_a_statement_depends_on_the_symbols_it_reads.py::TestAcrossARestart::test_an_unrelated_edit_across_a_restart_is_still_free`
