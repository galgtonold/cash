"""Unit tests for ForLoopHandler in isolation.

These tests construct ForLoopHandler directly with a mock shell,
mock statement processor, and mock dispatcher — exercising
per-iteration decomposition, iteration context, lineage
propagation across iterations, and the fast-loop heuristic
without going through ControlStructureProcessor.process() or
the full magics pipeline.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.control_structures.for_handler import ForLoopHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_shell():
    shell = MagicMock()
    shell.user_ns = {}
    return shell


@pytest.fixture
def mock_statement_processor():
    processor = MagicMock()
    processor.process_statement = MagicMock(return_value={
        'status': CacheStatus.COMPUTED,
        'execution_time': 0.01,
        'stdout': '', 'stderr': '', 'outputs': [],
    })
    processor.variable_lineage = {}
    processor.vars_with_mutation_lineage = set()
    processor.compute_hash = MagicMock(return_value='fakehash')
    return processor


@pytest.fixture
def mock_dispatcher():
    """Dispatcher mock — only used for nested control recursion / single-unit fallback."""
    return MagicMock()


@pytest.fixture
def handler(mock_shell, mock_statement_processor, mock_dispatcher):
    return ForLoopHandler(
        mock_shell, mock_statement_processor, debug=False, dispatcher=mock_dispatcher,
    )


def _parse_for(code: str) -> ast.For:
    return ast.parse(code).body[0]


# ---------------------------------------------------------------------------
# Per-iteration decomposition
# ---------------------------------------------------------------------------

def test_for_loop_one_call_per_iteration(handler, mock_statement_processor):
    """Each iteration → one call per body statement."""
    node = _parse_for("for i in range(4): x = i")
    result = handler.process(node, ttl=None, silent=True, parent_context=None)
    assert result.success
    assert result.total_iterations == 4
    assert mock_statement_processor.process_statement.call_count == 4


def test_for_loop_two_body_stmts_four_iterations(handler, mock_statement_processor):
    """2 body stmts × 3 iterations = 6 calls."""
    node = _parse_for("for i in range(3):\n    x = i\n    y = i * 2")
    handler.process(node, None, True, None)
    assert mock_statement_processor.process_statement.call_count == 6


# ---------------------------------------------------------------------------
# Iteration context
# ---------------------------------------------------------------------------

def test_iteration_context_marker_in_code(handler, mock_statement_processor):
    """Every body call must carry the __iteration_context__ comment."""
    node = _parse_for("for i in range(2): x = i")
    handler.process(node, None, True, None)
    for call in mock_statement_processor.process_statement.call_args_list:
        passed_code = call[0][0]
        assert passed_code.startswith("# __iteration_context__:")


def test_iteration_context_differs_per_iteration(handler, mock_statement_processor):
    """Different iteration values produce different context hashes."""
    node = _parse_for("for i in [10, 20, 30]: x = i")
    handler.process(node, None, True, None)
    ctx_hashes = {
        c[0][0].split('\n')[0].split(': ')[1]
        for c in mock_statement_processor.process_statement.call_args_list
    }
    assert len(ctx_hashes) == 3


def test_iteration_context_inherits_parent(handler, mock_statement_processor):
    """parent_context is folded into each iteration's context."""
    parent = {'outer_i': 7}
    node = _parse_for("for i in range(1): x = i")
    handler.process(node, None, True, parent_context=parent)
    # The hash should depend on parent contents — verify by running with a
    # different parent and confirming a different hash.
    h1 = mock_statement_processor.process_statement.call_args_list[0][0][0].split('\n')[0]
    mock_statement_processor.process_statement.reset_mock()
    handler.process(node, None, True, parent_context={'outer_i': 8})
    h2 = mock_statement_processor.process_statement.call_args_list[0][0][0].split('\n')[0]
    assert h1 != h2


# ---------------------------------------------------------------------------
# Lineage propagation
# ---------------------------------------------------------------------------

def test_loop_target_lineage_set_per_iteration(handler, mock_statement_processor, mock_shell):
    """Each iteration must set variable_lineage for the loop target."""
    node = _parse_for("for i in range(2): x = i")
    handler.process(node, None, True, None)
    # After the loop, 'i' must have a lineage entry (set on each iteration).
    assert 'i' in mock_statement_processor.variable_lineage


def test_loop_target_bound_in_user_ns(handler, mock_shell):
    """Last iteration's value persists in user_ns."""
    node = _parse_for("for i in range(3): x = i")
    handler.process(node, None, True, None)
    assert mock_shell.user_ns['i'] == 2  # last value of range(3)


# ---------------------------------------------------------------------------
# Mixed cache results
# ---------------------------------------------------------------------------

def test_for_loop_all_restored(handler, mock_statement_processor):
    mock_statement_processor.process_statement.return_value = {
        'status': CacheStatus.RESTORED,
        'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': [],
    }
    node = _parse_for("for i in range(3): x = i")
    result = handler.process(node, None, True, None)
    assert result.cached_iterations == 3
    assert result.computed_iterations == 0


def test_for_loop_mixed(handler, mock_statement_processor):
    mock_statement_processor.process_statement.side_effect = [
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.01, 'stdout': '', 'stderr': '', 'outputs': []},
        {'status': CacheStatus.RESTORED, 'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': []},
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.01, 'stdout': '', 'stderr': '', 'outputs': []},
    ]
    node = _parse_for("for i in range(3): x = i")
    result = handler.process(node, None, True, None)
    assert result.computed_iterations == 2
    assert result.cached_iterations == 1


# ---------------------------------------------------------------------------
# Fast-loop heuristic
# ---------------------------------------------------------------------------

def test_fast_loop_falls_back_to_single_unit(handler, mock_dispatcher):
    """Loops with many iterations × many stmts hit the fast-loop heuristic.

    1000 iterations × 200 body stmts × 8ms = 1600s estimated overhead,
    well over the 1s _MIN_OVERHEAD_SEC threshold.
    """
    body = "\n    ".join([f"x{i} = {i}" for i in range(200)])
    node = _parse_for(f"for i in range(1000):\n    {body}")
    handler.process(node, None, True, parent_context=None)
    # Should have called the dispatcher's single-unit fallback exactly once.
    assert mock_dispatcher._execute_as_single_unit.call_count == 1


def test_fast_loop_fires_even_when_nested(handler, mock_dispatcher, mock_statement_processor):
    """Nested huge loops also fall back to single-unit. (Earlier versions of
    the heuristic refused single-unit for any nested loop on the theory that
    convergence studies wanted granular cache invalidation. In practice the
    outer loop's per-iteration caching gives that benefit; the inner loop's
    7000+ trivial statements just cost time for no payoff.)"""
    body = "\n    ".join([f"x{i} = {i}" for i in range(200)])
    node = _parse_for(f"for i in range(1000):\n    {body}")
    handler.process(node, None, True, parent_context={'outer': 1})
    assert mock_dispatcher._execute_as_single_unit.call_count == 1
    # Per-iteration processing did NOT happen — single-unit short-circuited it.
    assert mock_statement_processor.process_statement.call_count == 0


def test_fast_loop_skipped_for_small_loop(handler, mock_dispatcher, mock_statement_processor):
    """Loops with few iterations always go per-iteration regardless of size."""
    # 10 iterations is well below _MIN_ITERATIONS_FOR_SINGLE_UNIT (50).
    node = _parse_for("for i in range(10): x = i")
    handler.process(node, None, True, None)
    assert mock_dispatcher._execute_as_single_unit.call_count == 0
    assert mock_statement_processor.process_statement.call_count == 10


def test_fast_loop_skipped_when_body_has_file_io(handler, mock_dispatcher, mock_statement_processor):
    """File-I/O calls in the body disable the fast-loop optimisation."""
    body = "\n    ".join([f"x{i} = {i}" for i in range(200)] + ["pd.read_csv('f.csv')"])
    node = _parse_for(f"for i in range(1000):\n    {body}")
    handler.process(node, None, True, None)
    assert mock_dispatcher._execute_as_single_unit.call_count == 0
