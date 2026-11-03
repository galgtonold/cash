"""The walk for the variables holding a class's objects skips what holds only values.

A class cell run again forgets every variable holding an object of the class
it replaced. When the climb up from those objects runs out of time, the walk
goes down from every variable instead, and it took a Python step for each
tuple, set and list of a helper's memo: in a TSP notebook, the half a million
index pairs of ``subsegment_indexes`` and the Held-Karp segments made each
cell defining a class take 4-7 s on every Run All. Nothing made only of
values (numbers, text, and tuples, sets, lists and dicts of them) can hold an
object of a class, so the walk does not go into it.
"""

from __future__ import annotations

import types

from cash.notebook.shared_objects import names_reaching
from tests._cell_driver import run_cash_cell


class Run:
    pass


def _values():
    """A memo's worth of plain values: index pairs, and segments keyed by
    ``(city, frozenset(cities), city)``."""
    return {
        "pairs": [(i, i + 2) for i in range(5_000)],
        "segments": {(complex(i, 1), frozenset({complex(i, 2), complex(i, 3)}), 1j): [1j, 2j] for i in range(5_000)},
        "lengths": {i: float(i) for i in range(5_000)},
    }


def test_variables_holding_only_values_are_not_walked():
    target = Run()
    ns = {**_values(), "results": {"run": [target]}, "target": target}
    assert names_reaching({id(target)}, ns, max_objects=100) == {"results", "target"}


def test_an_object_among_values_is_still_found():
    """Control: one record holding the object among thousands of plain ones,
    in a list and as a dict's key, is found."""
    target = Run()
    ns = {
        **_values(),
        "rows": [(i, i) for i in range(5_000)] + [(0, target)],
        "keyed": {(i, frozenset({i})): i for i in range(5_000)} | {(0, frozenset({1}), target): 1},
        "nested": [[(i, (i, i)) for i in range(100)], [(1, (2, target))]],
    }
    assert names_reaching({id(target)}, ns) == {"rows", "keyed", "nested"}


CLASS = "class Box:\n    def __init__(self, v):\n        self.v = v"


def _spy_walks(monkeypatch) -> list:
    from cash.notebook.statement import processor

    walk = processor.names_reaching
    walks: list = []

    def counted(*args, **kwargs):
        walks.append(1)
        return walk(*args, **kwargs)

    monkeypatch.setattr(processor, "names_reaching", counted)
    return walks


def _show(shell, n: int, value) -> None:
    """What IPython's display hook does with a cell's value: ``Out[n]`` and
    ``_n``, and its history manager holding ``Out``, as the shell does."""
    ns = shell.user_ns
    if "Out" not in ns:
        ns["Out"] = ns["_oh"] = {}
        shell.history_manager = types.SimpleNamespace(output_hist=ns["Out"])
    ns["Out"][n] = ns[f"_{n}"] = value


def test_an_object_only_displayed_ends_the_climb_at_the_output_history(cash_magics, monkeypatch):
    """``Box(1)`` shown by a cell is held by ``Out``, which the shell holds:
    climbing past it went through the shell and everything cash keeps on it
    until it ran out, and the class cell walked down every variable."""
    shell = cash_magics.shell
    cells = [CLASS, "kept = [Box(3)]", "Box(1)"]
    run_cash_cell(cash_magics, CLASS, cells=cells)
    run_cash_cell(cash_magics, "kept = [Box(3)]", cells=cells)
    _show(shell, 3, eval("Box(1)", shell.user_ns))
    walks = _spy_walks(monkeypatch)
    old_box = shell.user_ns["Box"]
    run_cash_cell(cash_magics, CLASS, cells=cells)
    assert shell.user_ns["Box"] is not old_box, "the class cell did not define the class anew"
    assert walks == []
    # Control: the variable holding an object of the old class is still
    # found, so the cell making it runs again and holds the class defined now.
    run_cash_cell(cash_magics, "kept = [Box(3)]", cells=cells)
    assert type(shell.user_ns["kept"][0]) is shell.user_ns["Box"]


def test_beside_a_large_memo_of_values_only_the_holders_are_forgotten(cash_magics, monkeypatch, recwarn):
    """The walk gives up past a million objects, and then every variable
    holding an object was forgotten, with a warning: E16's memos of index
    pairs and segments passed it, and every cell below ``class Runs`` re-ran
    the helpers' defs and every statement that had filled their memos."""
    import functools

    from cash.notebook.statement import processor

    # The climb ran out of a cheap class cell's budget, and the walk's
    # limit is lowered to what the test's memo passes.
    monkeypatch.setattr(processor, "referring_names", lambda *args, **kwargs: (set(), False))
    monkeypatch.setattr(processor, "names_reaching", functools.partial(processor.names_reaching, max_objects=1_000))
    shell = cash_magics.shell
    cells = [
        CLASS,
        "pairs = {n: [(i, i + 2) for i in range(n)] for n in (2_000, 3_000)}",
        "kept = [Box(3)]",
        "other = [1.0]",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    lineage = cash_magics.tracking_state.variable_lineage
    assert {"pairs", "kept", "other"} <= set(lineage)
    run_cash_cell(cash_magics, CLASS, cells=cells)
    assert "kept" not in lineage, "the variable holding an object of the old class was not forgotten"
    assert {"pairs", "other"} <= set(lineage)
    assert not [w for w in recwarn if "NOTEBOOK-SHARE-UNCHECKED" in str(w.message)]
