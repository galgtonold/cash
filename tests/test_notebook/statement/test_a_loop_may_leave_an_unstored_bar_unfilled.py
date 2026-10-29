"""What decides that a loop may leave an object cash did not store unfilled.

A hit of ``for f in files: ...; pbar.update()`` leaves ``pbar`` as it was.
That is safe only when the loop's use of it cannot reach anything else it
changes, and nothing after the loop reads it.
"""

from __future__ import annotations

import ast

from cash.notebook.statement.mutation_routing import _kept_to_itself, _loaded_names, _only_called_on, _read_by_functions


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


_BAR = (
    "import sys\n"
    "class Bar:\n"
    "    def __init__(self, held):\n        self.n = 0\n        self.held = held\n        self.me = self\n"
    "    def update(self):\n        self.n += 1\n"
    "class Logged(Bar):\n    def update(self):\n        log.append(1)\n"
)


def _notebook(code: str) -> dict:
    ns: dict = {"__name__": "__main__"}
    exec(_BAR + code, ns)
    return ns


def test_a_bar_only_its_name_holds_may_be_left():
    ns = _notebook("bar = Bar(sys.stdout)")
    assert _kept_to_itself(ns["bar"], "bar", ns, {"update"})


def test_a_bar_that_shares_a_value_or_is_held_elsewhere_may_not():
    ns = _notebook("sink = []\nbar = Bar(sink)")
    assert not _kept_to_itself(ns["bar"], "bar", ns, {"update"})
    ns = _notebook("bar = Bar(sys.stdout)\nbars = [bar]")
    assert not _kept_to_itself(ns["bar"], "bar", ns, {"update"})
    ns = _notebook("bar = Bar(sys.stdout)\ndef report(b=bar):\n    return b.n")
    assert not _kept_to_itself(ns["bar"], "bar", ns, {"update"})


def test_a_method_naming_a_notebook_value_may_not():
    ns = _notebook("log = []\nbar = Logged(sys.stdout)")
    assert not _kept_to_itself(ns["bar"], "bar", ns, {"update"})
