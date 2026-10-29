"""What decides that a loop may leave an object cash did not store unfilled.

A hit of ``for f in files: ...; pbar.update()`` leaves ``pbar`` as it was.
That is safe only when the loop's use of it cannot reach anything else it
changes, and nothing after the loop reads it.
"""

from __future__ import annotations

import ast

from cash.notebook.statement.mutation_routing import _loaded_names, _only_called_on, _read_by_functions


def _tree(code: str) -> ast.Module:
    return ast.parse(code)


def test_a_dropped_method_call_with_literal_arguments_may_be_left():
    assert _only_called_on(_tree("for f in fs:\n    x = f\n    bar.update()\n    bar.set(1, k='a')"), "bar")
    assert _only_called_on(_tree("for f in fs:\n    bar.set_description(f'{f}')"), "bar")


def test_a_use_that_can_reach_other_values_may_not():
    assert not _only_called_on(_tree("for f in fs:\n    n = bar.update()"), "bar")
    assert not _only_called_on(_tree("for f in fs:\n    bar.update(rows)"), "bar")
    assert not _only_called_on(_tree("for f in fs:\n    rows.append(bar.n)"), "bar")
    assert not _only_called_on(_tree("for f in fs:\n    bar.update()\n    show(bar)"), "bar")


def test_reading_the_namespace_by_name_reads_everything():
    assert _loaded_names([ast.parse("x = globals()['bar']").body[0]]) is None
    assert _loaded_names([ast.parse("x = bar.n + y").body[0]]) == {"bar", "y"}


def test_a_notebook_function_reading_the_name_as_a_global_counts():
    ns: dict = {}
    exec(
        "def show():\n    return bar.n\n"
        "def other():\n    return 1\n"
        "class View:\n    @property\n    def seen(self):\n        return bar.n\n",
        {"__name__": "__main__"},
        ns,
    )
    assert _read_by_functions({"show": ns["show"]}, {"bar"}) == {"bar"}
    assert _read_by_functions({"other": ns["other"]}, {"bar"}) == set()
    assert _read_by_functions({"View": ns["View"]}, {"bar"}) == {"bar"}
