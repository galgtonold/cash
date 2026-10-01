from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus

"""
Tests for cache behavior when re-executing statements.

The ALREADY_EXECUTED skip optimization was removed because it was redundant
with cache lookups and caused correctness bugs (especially with self-assignments
like df['col'] = ...). Now all re-executed statements go through the normal
cache lookup path: COMPUTED on first run, RESTORED on subsequent identical runs.
"""


# Force caching regardless of the 10 ms min-execution-time floor.
_PERSIST = CacheAnnotation(persist=True)


class TestCacheRestoreBehavior:
    """Test that statements are properly cached and restored without ALREADY_EXECUTED optimization."""

    def test_first_run_computes(self, mock_shell, statement_processor):
        """First execution of a statement should COMPUTE."""

        metrics = statement_processor.process_statement("x = 42")
        assert metrics["status"] == CacheStatus.COMPUTED
        assert mock_shell.user_ns.get("x") == 42

    def test_second_run_restores_from_cache(self, mock_shell, statement_processor):
        """Second identical execution should RESTORE from cache.
        _PERSIST overrides the 10 ms min-execution-time floor."""

        metrics1 = statement_processor.process_statement("x = 42", annotation=_PERSIST)
        assert metrics1["status"] == CacheStatus.COMPUTED

        metrics2 = statement_processor.process_statement("x = 42", annotation=_PERSIST)
        assert metrics2["status"] == CacheStatus.RESTORED
        assert mock_shell.user_ns.get("x") == 42

    def test_multiple_reruns_always_restore(self, statement_processor):
        """Multiple re-runs of same statement should always RESTORE.
        _PERSIST overrides the 10 ms min-execution-time floor."""

        statement_processor.process_statement("x = 42", annotation=_PERSIST)
        for _ in range(3):
            metrics = statement_processor.process_statement("x = 42", annotation=_PERSIST)
            assert metrics["status"] == CacheStatus.RESTORED

    def test_cache_still_stores_results(self, mock_shell, statement_processor):
        """Results should be stored in cache and retrievable.
        _PERSIST overrides the 10 ms min-execution-time floor."""

        metrics = statement_processor.process_statement("z = 99", annotation=_PERSIST)
        assert metrics["status"] == CacheStatus.COMPUTED
        assert mock_shell.user_ns.get("z") == 99

        # Re-run should restore from cache
        metrics2 = statement_processor.process_statement("z = 99", annotation=_PERSIST)
        assert metrics2["status"] == CacheStatus.RESTORED

    def test_mutation_pattern_executes(self, mock_shell, statement_processor):
        """Mutation calls should be detected and handled properly."""

        statement_processor.process_statement("items = []")
        assert mock_shell.user_ns.get("items") == []

        # Mutation: append modifies items in-place
        statement_processor.process_statement("items.append(1)")
        assert mock_shell.user_ns.get("items") == [1]

        # Re-running the mutation after resetting should work
        mock_shell.user_ns["items"] = []
        statement_processor.process_statement("items.append(1)")
        assert mock_shell.user_ns.get("items") == [1]
