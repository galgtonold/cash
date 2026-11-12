"""Each output of a statement is linked to the variables it shares an object with.

The share check takes a statement's outputs together, a round at a time: a
round that finds ``train_dl`` holding ``training_data`` and ``test_dl``
holding ``test_data`` linked both loaders to both datasets. A loop handing
both loaders to its functions then moved ``test_data`` on through
``train_dl``, and the cell reading ``test_data`` below rebuilt it and all
that came after. Each output is linked to what it shares an object with;
a change through either loader still moves the dataset it holds.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

CELLS = [
    "class Loader:\n"
    "    def __init__(self, data):\n"
    "        self.dataset = data\n"
    "    def __iter__(self):\n"
    "        return iter(self.dataset)\n"
    "class Net:\n"
    "    def __init__(self):\n"
    "        self.w = 0.0\n"
    "def train(dl, m):\n"
    "    for x in dl:\n"
    "        m.w += x[0] * 1e-4\n"
    "def test(dl, m):\n"
    "    return sum(m.w * x[0] for x in dl)",
    "training_data = [[float(i)] for i in range(50)]\ntest_data = [[float(i)] for i in range(10)]",
    "train_dl = Loader(training_data)\ntest_dl = Loader(test_data)\nmodel = Net()",
    "for t in range(2):\n    train(train_dl, model)\n    acc = test(test_dl, model)",
]


def _partners(magics, name: str) -> set[str]:
    edges = magics.tracking_state.derivation_edges
    return set(edges.get(name, ())) | {src for src, targets in edges.items() if name in targets}


def _run(magics) -> None:
    magics.cash_on("")
    magics.badges.mode = "off"
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)


def test_a_loader_is_linked_to_its_own_dataset_only(cash_magics):
    _run(cash_magics)

    assert "training_data" in _partners(cash_magics, "train_dl"), "control: the loader holds its data"
    assert "test_data" not in _partners(cash_magics, "train_dl")
    assert "training_data" not in _partners(cash_magics, "test_dl")


@pytest.mark.parametrize(
    "change, moved, kept",
    [
        ("train_dl.dataset.append([1.0])", "training_data", "test_data"),
        ("test_dl.dataset[0][0] = 5.0", "test_data", "training_data"),
    ],
    ids=["through the train loader", "through the test loader"],
)
def test_a_change_through_a_loader_moves_its_dataset(cash_magics, change, moved, kept):
    _run(cash_magics)
    lineage = cash_magics.tracking_state.variable_lineage
    before = {name: lineage[name] for name in (moved, kept)}

    run_cash_cell(cash_magics, change, cells=[*CELLS, change])

    assert lineage[moved] != before[moved]
    assert lineage[kept] == before[kept]
