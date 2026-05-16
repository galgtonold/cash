"""Unit tests for IfHandler in isolation.

Construct IfHandler directly and exercise branch selection (true/false/elif),
per-statement decomposition, control_context discrimination, and condition
evaluation against the user namespace.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.control_if_handler import IfHandler


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
    return MagicMock()


@pytest.fixture
def handler(mock_shell, mock_statement_processor, mock_dispatcher):
    return IfHandler(mock_shell, mock_statement_processor, debug=False, dispatcher=mock_dispatcher)


def _parse_if(code: str) -> ast.If:
    return ast.parse(code).body[0]


# ---------------------------------------------------------------------------
# Branch selection
# ---------------------------------------------------------------------------

def test_if_true_branch_taken(handler, mock_shell, mock_statement_processor):
    mock_shell.user_ns['flag'] = True
    node = _parse_if("if flag:\n    x = 1\nelse:\n    x = 0")
    handler.process(node, None, True)
    # Only the 'if' branch should execute → 1 call with code containing "x = 1".
    assert mock_statement_processor.process_statement.call_count == 1
    code = mock_statement_processor.process_statement.call_args[0][0]
    assert 'x = 1' in code
    assert 'x = 0' not in code


def test_if_false_else_branch_taken(handler, mock_shell, mock_statement_processor):
    mock_shell.user_ns['flag'] = False
    node = _parse_if("if flag:\n    x = 1\nelse:\n    x = 0")
    handler.process(node, None, True)
    assert mock_statement_processor.process_statement.call_count == 1
    code = mock_statement_processor.process_statement.call_args[0][0]
    assert 'x = 0' in code
    assert 'x = 1' not in code


def test_elif_branch_taken(handler, mock_shell, mock_statement_processor):
    mock_shell.user_ns['a'] = 0
    mock_shell.user_ns['b'] = 1
    node = _parse_if("if a:\n    x = 1\nelif b:\n    x = 2\nelse:\n    x = 3")
    handler.process(node, None, True)
    code = mock_statement_processor.process_statement.call_args[0][0]
    assert 'x = 2' in code


def test_if_no_match_no_else_empty_branch(handler, mock_shell, mock_statement_processor):
    """If no condition matches and no else, no body statements execute."""
    mock_shell.user_ns['flag'] = False
    node = _parse_if("if flag:\n    x = 1")
    result = handler.process(node, None, True)
    assert result.success
    assert mock_statement_processor.process_statement.call_count == 0


# ---------------------------------------------------------------------------
# Per-statement decomposition
# ---------------------------------------------------------------------------

def test_if_each_body_stmt_one_call(handler, mock_shell, mock_statement_processor):
    mock_shell.user_ns['flag'] = True
    node = _parse_if("if flag:\n    x = 1\n    y = 2\n    z = 3")
    handler.process(node, None, True)
    assert mock_statement_processor.process_statement.call_count == 3


def test_if_control_context_marker_present(handler, mock_shell, mock_statement_processor):
    mock_shell.user_ns['flag'] = True
    node = _parse_if("if flag:\n    x = 1\n    y = 2")
    handler.process(node, None, True)
    for call in mock_statement_processor.process_statement.call_args_list:
        passed_code = call[0][0]
        assert passed_code.startswith("# control_context:")


def test_if_metrics_tagged_with_control_type(handler, mock_shell):
    mock_shell.user_ns['flag'] = True
    node = _parse_if("if flag:\n    x = 1")
    result = handler.process(node, None, True)
    for m in result.metrics:
        assert m.get('control_type') == 'if'


def test_if_metrics_tagged_with_branch_label(handler, mock_shell):
    mock_shell.user_ns['flag'] = True
    node = _parse_if("if flag:\n    x = 1")
    result = handler.process(node, None, True)
    for m in result.metrics:
        assert 'branch_label' in m
        assert m['branch_label'].startswith('if ')


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def test_condition_evaluation_failure_returns_failure(handler, mock_shell):
    """Undefined name in condition → process() reports success=False."""
    node = _parse_if("if undefined_name:\n    x = 1")
    result = handler.process(node, None, True)
    assert result.success is False
    assert result.error is not None
