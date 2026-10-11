"""A loop that changes the items it walks changes the collection they are in.

``for r in records: r['tax'] = ...`` rebinds ``r`` each pass, but each object
``r`` is bound to is an item of ``records``. The loop's mutations count
``records`` (the names the loop's iterable is built from), so the lineage of
``records`` moves and a cell reading it is not served the old items.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.mutation_effects import control_structure_mutations


def _mutations(code: str) -> set[str]:
    return control_structure_mutations(ast.parse(code).body[0], lambda n: n in {"range", "zip"}, lambda n: n == "np")


@pytest.mark.parametrize(
    "loop, expected",
    [
        ("for r in records:\n    r['tax'] = 1", {"records"}),
        ("for r in rows:\n    r.append(5)", {"rows"}),
        ("for a in arrs:\n    a += 3", {"arrs"}),
        ("for ax in axes:\n    ax.plot([1, 2])", {"axes"}),
        ("for k, d in cfgs.items():\n    d['score'] = k", {"cfgs"}),
        ("for a, b in zip(xs, ys):\n    a.append(b)", {"xs", "ys"}),
    ],
    ids=["dict items", "list items", "arrays", "axes", "dict values", "zip"],
)
def test_changing_an_item_changes_the_iterable(loop, expected):
    assert _mutations(loop) == expected


@pytest.mark.parametrize(
    "loop",
    [
        "for line in lines:\n    line = line.strip()",
        "for x in xs:\n    total = x",
        "for i in range(3):\n    print(i)",
    ],
    ids=["rebinding the target", "reading the target", "a range"],
)
def test_rebinding_or_reading_an_item_changes_nothing(loop):
    assert _mutations(loop) == set()
