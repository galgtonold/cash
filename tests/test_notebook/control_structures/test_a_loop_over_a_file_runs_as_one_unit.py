"""A long ``for line in open(path):`` loop runs as one unit, not one statement a line.

Per-iteration caching costs about 2 ms a body statement. Over an 87,000-line
log that was 9 min 20 s for a loop that prints each line in 14 s without
cash. Such a loop was never run as one unit: an open file is a one-shot
iterator with no ``len``, so it could neither be sized nor evaluated twice.
``open(path)`` in the header is a new handle at the start of the file on every
evaluation, so the second evaluation is safe, and the line count can be read
off the file.
"""

from __future__ import annotations

import ast

import pytest

from cash.notebook.control_structures import single_unit_policy as policy
from tests._cell_driver import run_cash_cell


def _safe(header: str, user_ns: dict) -> bool:
    node = ast.parse(f"for x in {header}:\n    pass").body[0]
    iterable = eval(header, dict(user_ns), dict(user_ns))
    try:
        return policy.header_safe_to_reevaluate(node.iter, iterable, user_ns)
    finally:
        close = getattr(iterable, "close", None)
        if close is not None:
            close()


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("".join(f"@User{i}: Action_{i % 7}\n" for i in range(3000)), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize(
    "header",
    [
        "open(path)",
        "open(path, 'r')",
        "open(path, 'rt')",
        "open(path, 'rb')",
        "open(path, mode='r', encoding='utf-8')",
        "open(path, encoding='utf-8')",
        "enumerate(open(path))",
        "zip(open(path), range(5))",
    ],
)
def test_a_file_opened_for_reading_in_the_header_may_be_evaluated_twice(header, path):
    assert _safe(header, {"path": path})


@pytest.mark.parametrize(
    "header",
    [
        "open(path, how)",
        "fh",
        "enumerate(fh)",
        "zip(open(path), fh)",
    ],
)
def test_a_mode_that_is_not_a_literal_and_a_stored_handle_may_not(header, path):
    ns = {"path": path, "how": "r", "fh": open(path, encoding="utf-8")}
    try:
        assert not _safe(header, ns)
    finally:
        ns["fh"].close()


@pytest.mark.parametrize("mode", ["w", "a", "r+", "x"])
def test_opening_for_writing_is_refused_without_opening_it(mode, path):
    node = ast.parse(f"for x in open(path, {mode!r}):\n    pass").body[0]
    assert not policy.header_safe_to_reevaluate(node.iter, iter(()), {"path": path})


def test_a_notebooks_own_open_gets_no_benefit_of_the_doubt(path):
    def open(*_a, **_k):
        return iter(())

    node = ast.parse("for x in open(path):\n    pass").body[0]
    assert not policy.header_safe_to_reevaluate(node.iter, iter(()), {"path": path, "open": open})


def test_the_line_count_of_a_file_is_estimated_from_its_size(tmp_path):
    uniform = tmp_path / "uniform.txt"
    uniform.write_text("abcdefghi\n" * 20_000, encoding="utf-8")
    with open(uniform, encoding="utf-8") as fh:
        estimate = policy.estimated_iterations(ast.parse("open(p)", mode="eval").body, fh, {})
    assert abs(estimate - 20_000) <= 200


def test_a_file_that_fits_the_sample_is_counted_exactly(tmp_path):
    for text, lines in (("", 0), ("a\n", 1), ("a\nb", 2), ("a\nb\n\n", 3), ("x" * 500, 1)):
        p = tmp_path / "small.txt"
        p.write_text(text, encoding="utf-8")
        with open(p, encoding="utf-8") as fh:
            assert policy.estimated_iterations(ast.parse("open(p)", mode="eval").body, fh, {}) == lines, text


def test_a_long_file_loop_is_run_as_one_unit_and_a_short_one_is_not(tmp_path, path):
    node = ast.parse("for line in open(path):\n    n += 1").body[0]
    with open(path, encoding="utf-8") as long_file:
        assert policy.should_run_as_single_unit(node, long_file, {"path": path})
    short = tmp_path / "short.txt"
    short.write_text("a\nb\nc\n", encoding="utf-8")
    with open(short, encoding="utf-8") as short_file:
        assert not policy.should_run_as_single_unit(node, short_file, {})


def test_a_loop_over_a_file_is_computed_right_and_follows_the_file(cash_magics, mock_shell, path):
    code = f"n = 0\nusers = set()\nfor line in open({path!r}):\n    n += 1\n    users.add(line.split(':')[0])"
    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["n"] == 3000 and len(mock_shell.user_ns["users"]) == 3000
    statements = cash_magics.cash_status("dict")["last_cell"].get("statements", [])
    assert len(statements) < 10, f"one statement per line: {len(statements)} statements"

    with open(path, "a", encoding="utf-8") as fh:
        fh.write("@User3000: Action_0\n")
    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["n"] == 3001, "the file changed, so the loop must run again"
    assert len(mock_shell.user_ns["users"]) == 3001

    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["n"] == 3001 and len(mock_shell.user_ns["users"]) == 3001


def test_a_loop_over_a_file_with_an_expensive_body_keeps_its_result_when_nothing_changed(cash_magics, mock_shell, path):
    code = f"total = 0\nfor line in open({path!r}):\n    total += len(line) + sum(range(20))"
    run_cash_cell(cash_magics, code)
    first = mock_shell.user_ns["total"]
    mock_shell.user_ns["total"] = -1
    run_cash_cell(cash_magics, code)
    assert mock_shell.user_ns["total"] == first


def test_the_open_ipython_puts_in_every_namespace_counts_as_the_builtin(path):
    """A real notebook namespace has its own `open` entry, IPython's wrapper."""
    from IPython.core.interactiveshell import InteractiveShell

    ipython_open = InteractiveShell.instance().user_ns["open"]
    assert _safe("open(path)", {"path": path, "open": ipython_open})
