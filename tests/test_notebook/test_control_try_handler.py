"""Unit tests for TryHandler in isolation.

Construct TryHandler directly and exercise exception binding, handler
matching (typed and bare), else/finally execution, and the body_stmts
badge metadata for actually-executed branches.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.control_try_handler import TryHandler


@pytest.fixture
def mock_shell():
    shell = MagicMock()
    shell.user_ns = {
        'ValueError': ValueError,
        'KeyError': KeyError,
        'Exception': Exception,
        'RuntimeError': RuntimeError,
    }
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
    return MagicMock()


@pytest.fixture
def handler(mock_shell, mock_statement_processor, mock_dispatcher):
    return TryHandler(mock_shell, mock_statement_processor, debug=False, dispatcher=mock_dispatcher)


def _parse_try(code: str) -> ast.Try:
    return ast.parse(code).body[0]


# ---------------------------------------------------------------------------
# Try body success path
# ---------------------------------------------------------------------------

def test_try_success_runs_only_try_body(handler, mock_statement_processor):
    node = _parse_try("try:\n    x = 1\nexcept:\n    x = 0")
    handler.process(node, None, True)
    assert mock_statement_processor.process_statement.call_count == 1


def test_try_success_runs_else(handler, mock_statement_processor):
    """When try body succeeds, else body also runs."""
    node = _parse_try("try:\n    x = 1\nexcept:\n    x = 0\nelse:\n    y = 2")
    handler.process(node, None, True)
    # 1 try-body call + 1 else-body call = 2
    assert mock_statement_processor.process_statement.call_count == 2


def test_try_success_runs_finally(handler, mock_statement_processor):
    node = _parse_try("try:\n    x = 1\nfinally:\n    z = 3")
    handler.process(node, None, True)
    assert mock_statement_processor.process_statement.call_count == 2


# ---------------------------------------------------------------------------
# Exception → handler matching
# ---------------------------------------------------------------------------

def test_typed_handler_matches(handler, mock_statement_processor):
    """ValueError raised in try → ValueError handler runs, else suppressed."""
    mock_statement_processor.process_statement.side_effect = [
        ValueError("boom"),  # try body raises
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': []},
    ]
    node = _parse_try(
        "try:\n    x = int('abc')\nexcept ValueError:\n    x = -1\nelse:\n    y = 999"
    )
    result = handler.process(node, None, True)
    assert result.success is True
    # 1 try call (raises) + 1 handler call. else MUST NOT run.
    assert mock_statement_processor.process_statement.call_count == 2


def test_bare_except_catches_anything(handler, mock_statement_processor):
    mock_statement_processor.process_statement.side_effect = [
        RuntimeError("anything"),
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': []},
    ]
    node = _parse_try("try:\n    x = 1\nexcept:\n    x = -1")
    result = handler.process(node, None, True)
    assert result.success is True
    assert mock_statement_processor.process_statement.call_count == 2


def test_unmatched_handler_propagates(handler, mock_statement_processor):
    """ValueError → only KeyError handler defined → exception propagates and
    process() reports success=False."""
    mock_statement_processor.process_statement.side_effect = [
        ValueError("not a key error"),
    ]
    node = _parse_try("try:\n    x = 1\nexcept KeyError:\n    x = -1")
    result = handler.process(node, None, True)
    assert result.success is False
    assert isinstance(result.error, ValueError)


# ---------------------------------------------------------------------------
# Exception binding
# ---------------------------------------------------------------------------

def test_exception_bound_to_handler_var(handler, mock_shell, mock_statement_processor):
    """except X as e → e is bound in user_ns before handler body runs."""
    err = ValueError("captured")
    mock_statement_processor.process_statement.side_effect = [
        err,
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': []},
    ]
    node = _parse_try("try:\n    x = 1\nexcept ValueError as e:\n    y = 2")
    handler.process(node, None, True)
    assert mock_shell.user_ns.get('e') is err
    # Exception should have a lineage entry recorded through the LineageStore seam
    mock_statement_processor.lineage.record.assert_called_once()
    args, kwargs = mock_statement_processor.lineage.record.call_args
    assert args[0] == 'e'
    assert kwargs.get('value') is err


# ---------------------------------------------------------------------------
# Finally always runs
# ---------------------------------------------------------------------------

def test_finally_runs_after_caught_exception(handler, mock_statement_processor):
    mock_statement_processor.process_statement.side_effect = [
        ValueError("boom"),
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': []},  # handler
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': []},  # finally
    ]
    node = _parse_try(
        "try:\n    x = 1\nexcept ValueError:\n    x = -1\nfinally:\n    z = 3"
    )
    result = handler.process(node, None, True)
    assert result.success is True
    assert mock_statement_processor.process_statement.call_count == 3


# ---------------------------------------------------------------------------
# Badge metadata
# ---------------------------------------------------------------------------

def test_metrics_tagged_with_control_type(handler):
    node = _parse_try("try:\n    x = 1\nexcept:\n    x = 0")
    result = handler.process(node, None, True)
    for m in result.metrics:
        assert m.get('control_type') == 'try'


def test_body_statements_includes_matched_handler(handler, mock_statement_processor):
    """body_statements should include the executed try + matched handler branches."""
    mock_statement_processor.process_statement.side_effect = [
        ValueError("boom"),
        {'status': CacheStatus.COMPUTED, 'execution_time': 0.0, 'stdout': '', 'stderr': '', 'outputs': []},
    ]
    node = _parse_try("try:\n    x = 1\nexcept ValueError:\n    x = -1")
    result = handler.process(node, None, True)
    assert result.success
    stmts = result.metrics[0].get('body_statements', [])
    text = '\n'.join(stmts)
    assert 'try:' in text
    assert 'except ValueError' in text
