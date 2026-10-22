"""``voting.voting = "soft"`` moves on every name holding the voting
classifier's fitted forests, and reads each after the move: the forests'
padded node arrays were packed again through each list holding them. In
one such pass each array is packed once, to the digests a fresh read
gives; a change between two statements is read again."""

from __future__ import annotations

import contextlib

import pytest

from cash import kept_state, value_hash
from cash.notebook.statement import lineage
from tests._cell_driver import run_cash_cell

pytest.importorskip("numpy")

CELLS = [
    "import types\n"
    "import numpy as np\n"
    "NODE = np.dtype({'names': ['left', 'threshold', 'missing'], 'formats': ['i8', 'f8', 'u1'], 'aligned': True})\n"
    "def fitted(seed):\n"
    "    nodes = np.zeros(200, dtype=NODE)\n"
    "    nodes['left'] = np.arange(200) + seed\n"
    "    nodes['threshold'] = np.linspace(0, 1, 200) * seed\n"
    "    return types.SimpleNamespace(n=seed, nodes=nodes)",
    "a = fitted(1)\nb = fitted(2)",
    "models = [a, b]\nnamed = [('a', a), ('b', b)]",
    "holder = {'models': models, 'named': named}",
]
NAMES = ("a", "b", "models", "named", "holder")


@pytest.fixture
def session(cash_magics):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    return cash_magics


@pytest.fixture
def packs(monkeypatch):
    """The arrays packed in a pass, by id."""
    seen = []
    real = kept_state._pack

    def counting(array, packed):
        if kept_state._packing.memo is not None:
            seen.append(id(array))
        return real(array, packed)

    monkeypatch.setattr(kept_state, "_pack", counting)
    return seen


def _fresh(session, name):
    return value_hash.compute_hash(session.shell.user_ns[name])


def test_each_array_is_packed_once_and_digests_as_a_fresh_read(session, packs):
    cell = "holder['models'][0].n = 5"
    before = dict(session.tracking_state.current_session_hashes)
    run_cash_cell(session, cell, cells=[*CELLS, cell])
    hashes = session.tracking_state.current_session_hashes
    moved = [name for name in NAMES if hashes.get(name) != before.get(name)]
    assert "a" in moved and "models" in moved, moved
    assert packs, "nothing was packed in a pass"
    assert len(packs) == len(set(packs)), "an array was packed twice in one pass"
    packs.clear()
    for name in moved:
        assert hashes[name] == _fresh(session, name), name


def test_the_digests_are_those_without_the_reuse(session, monkeypatch):
    """The same names read in one pass with the reuse give, digest for
    digest, what the move gave them without it."""

    monkeypatch.setattr(lineage, "packing_once", contextlib.nullcontext)
    cell = "holder['models'][0].n = 5"
    run_cash_cell(session, cell, cells=[*CELLS, cell])
    without = {name: session.tracking_state.current_session_hashes.get(name) for name in NAMES}
    monkeypatch.undo()
    with kept_state.packing_once():
        with_reuse = {name: _fresh(session, name) for name in NAMES if without[name] is not None}
    assert with_reuse == {name: digest for name, digest in without.items() if digest is not None}


def test_an_array_changed_by_the_next_statement_is_read_again(session):
    cell = "holder['models'][0].n = 5\na.nodes['left'][0] = 999"
    before = dict(session.tracking_state.current_session_hashes)
    run_cash_cell(session, cell, cells=[*CELLS, cell])
    hashes = session.tracking_state.current_session_hashes
    for name in ("a", "models", "named"):
        assert hashes[name] != before[name], name
        assert hashes[name] == _fresh(session, name), name


def test_nothing_is_kept_past_the_pass(session):
    cell = "holder['models'][0].n = 5"
    run_cash_cell(session, cell, cells=[*CELLS, cell])
    assert kept_state._packing.memo is None
