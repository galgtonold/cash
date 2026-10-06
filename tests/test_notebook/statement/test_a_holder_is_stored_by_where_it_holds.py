"""A variable stored with a statement's outputs because it holds one of their
objects is stored by where it holds it, not whole.

``train = data['train']``, then ``train['f1'] = slow(...)``: ``data`` holds the
frame the statement changes, so a hit must leave ``data['train']`` the restored
frame. Stored whole, ``data`` brought every other frame it holds (``raw``)
into the entry of each feature statement: three statements on an 8 MB frame
wrote the 80 MB one next to it three times. The entry now records the place
(``data['train']``), and a hit puts the restored frame there, in the live
``data``.
"""

from __future__ import annotations

from cash.notebook.holder_patches import HolderPatch, apply_patch, holder_patches
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import time\ndef slow(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x"
CELLS = [
    SETUP,
    "data = {'train': {'a': 1}, 'raw': list(range(100_000)), 'meta': [{'n': 1}]}",
    "train = data['train']",
    "train['f1'] = slow(train['a'] * 2)",
    "train['f2'] = slow(train['a'] * 3)",
]


def _payload(cash_magics, statement_processor, code):
    metrics = statement_processor.process_statement(code)
    _metadata, payload = cash_magics._cash_instance.backend.get(metrics["cache_key"])
    return payload


def test_the_entry_holds_the_place_not_the_container(cash_magics, statement_processor):
    for cell in CELLS[:3]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    payload = _payload(cash_magics, statement_processor, CELLS[3])
    held = payload["variables"]["data"]
    assert isinstance(held, HolderPatch), type(held)
    assert "raw" not in repr(held)


def test_a_second_run_all_keeps_the_container_holding_the_frame(cash_magics):
    ns = cash_magics.shell.user_ns
    for run in ("first", "second"):
        for cell in CELLS:
            run_cash_cell(cash_magics, cell, cells=CELLS)
        assert ns["data"]["train"] is ns["train"], f"{run} Run All"
        assert ns["train"] == {"a": 1, "f1": 2, "f2": 3}, f"{run} Run All"
        assert len(ns["data"]["raw"]) == 100_000


def test_a_holder_whose_place_cannot_be_set_is_stored_whole():
    """A tuple's items cannot be set, nor a set's: such a holder is stored
    whole, as before."""
    d = {"a": 1}
    assert holder_patches({"pair": (d, 0)}, {"d": d}) is None
    assert holder_patches({"rows": [d], "pair": (d, 0)}, {"d": d}) is None
    nested = {"x": [d]}
    patches = holder_patches({"nested": nested}, {"d": d})
    assert patches is not None
    restored = {"a": 1, "f": 2}
    assert apply_patch(nested, patches["nested"], {"d": restored}) is nested
    assert nested["x"][0] is restored
