"""A value a statement binds and stores is pickled once.

After a statement ran, its digest of an output (the session hash, the
value cutoff) pickled the value, and its store pickled it again, with
nothing of the user's run in between: two pickles of a graph of 350,000
edges, 0.4-0.7 s each. A value pickled whole is now digested over the very
bytes the store keeps (`stored_pickle.object_digest`), and the store takes
them.

What must hold: an edit made before the store, by a later statement, is
what that statement stores; a value digests alike before and after a
statement that did not change it, and in a new process; a value whose
store bytes leave something out (an attribute its class's own state skips)
still keys on it; two outputs sharing an object come back sharing it.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell
from tests._scripts import run_python

PICKLED: list[int] = []


class Fit:
    """A results object holding a 2 MiB array, counted each time it is pickled."""

    def __init__(self, rows: int = 1 << 18) -> None:
        self.coef = np.arange(rows, dtype=float)
        self.names = ["a", "b"]

    def __reduce_ex__(self, protocol):
        PICKLED.append(1)
        return object.__reduce_ex__(self, protocol)


class Thresholded:
    """A model whose own ``__getstate__`` leaves a setting out of its state."""

    def __init__(self) -> None:
        self.coef = np.arange(1 << 17, dtype=float)
        self.threshold = 0.5

    def __getstate__(self):
        return {"coef": self.coef}


# Slow enough to be stored, and no call the call cache takes apart.
BUILD = "fit = Fit() if time.sleep(0.3) is None else None"


@pytest.fixture
def tiers(cash_magics, cash_instance, mock_shell, tmp_path):
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "nbcache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    run_cash_cell(cash_magics, f"import time\nfrom {__name__} import Fit")
    PICKLED.clear()
    yield cash_magics, ram, disk
    disk.shutdown()


def _entry(ram, name):
    """The variables of the RAM tier's entry holding *name*, as a hit reads
    them, and whether the tier keeps that entry as pickle bytes."""
    for key, (_meta, stored) in list(ram._store.items()):
        variables = ram.get(key)[1].get("variables", {})
        if name in variables:
            return variables, type(stored) is memory_backend._Pickled
    return None, False


def test_a_stored_output_is_pickled_once(tiers, mock_shell):
    magics, ram, disk = tiers
    run_cash_cell(magics, BUILD)
    disk._writes.wait_all()
    assert PICKLED == [1], "digested and stored by two pickles"
    variables, as_bytes = _entry(ram, "fit")
    assert as_bytes, "not kept as pickle bytes: the test proves nothing"
    served = variables["fit"]
    assert np.array_equal(served.coef, mock_shell.user_ns["fit"].coef) and served.coef.flags.writeable
    assert served is not mock_shell.user_ns["fit"]


def test_an_edit_after_the_statement_is_not_what_a_rerun_restores(tiers, mock_shell):
    from cash import stored_pickle

    magics, _ram, _disk = tiers
    run_cash_cell(magics, BUILD)
    assert stored_pickle.kept(mock_shell.user_ns["fit"]) is None, "kept past its statement"
    run_cash_cell(magics, "fit.coef[0] = -1.0")
    PICKLED.clear()
    run_cash_cell(magics, BUILD)
    assert mock_shell.user_ns["fit"].coef[0] == 0.0


def test_two_outputs_sharing_an_object_come_back_sharing_it(tiers, mock_shell):
    magics, ram, _disk = tiers
    run_cash_cell(magics, "fit, names = [(f, f.names) for f in [Fit()] if time.sleep(0.3) is None][0]")
    variables, _as_bytes = _entry(ram, "fit")
    assert variables is not None, "not stored: the test proves nothing"
    assert variables["fit"].names is variables["names"]


def test_an_unchanged_value_digests_alike_before_and_after_a_statement(tiers, mock_shell):
    from cash import stored_pickle

    magics, _ram, _disk = tiers
    run_cash_cell(magics, BUILD)
    before = compute_hash(mock_shell.user_ns["fit"])
    run_cash_cell(magics, "print(len(fit.names))")
    assert compute_hash(mock_shell.user_ns["fit"]) == before
    with stored_pickle.keeping(lambda value: True):
        assert compute_hash(mock_shell.user_ns["fit"]) == before, "a kept digest is another digest"


def test_a_digest_is_the_same_in_a_new_process(tmp_path):
    script = """
import numpy as np
from cash.value_hash import compute_hash

class Box:
    def __init__(self):
        self.data = np.arange(1 << 17, dtype=float)
        self.label = "x"

print(compute_hash(Box()))
"""
    first = run_python("-c", script, cwd=tmp_path).stdout.strip()
    assert first and first == run_python("-c", script, cwd=tmp_path).stdout.strip()


def test_a_setting_the_stored_bytes_leave_out_still_moves_the_digest():
    from cash import stored_pickle

    model = Thresholded()
    before = compute_hash(model)
    with stored_pickle.keeping(lambda value: True):
        assert compute_hash(model) == before
        assert stored_pickle.kept(model) is None, "store bytes leaving a setting out kept as a digest's"
    model.threshold = 0.9
    assert compute_hash(model) != before


def test_a_record_array_with_padding_keys_without_it():
    dtype = np.dtype({"names": ["a", "b"], "formats": ["f8", "u1"], "offsets": [0, 16], "itemsize": 24})
    left, right = (
        types.SimpleNamespace(rows=np.zeros(4, dtype=dtype)),
        types.SimpleNamespace(rows=np.zeros(4, dtype=dtype)),
    )
    right.rows.view(np.uint8).reshape(4, 24)[:, 20] = 7  # padding only
    assert compute_hash(left) == compute_hash(right)
    right.rows["b"][0] = 1
    assert compute_hash(left) != compute_hash(right)


def test_a_digest_taken_outside_a_store_is_never_kept():
    from cash import stored_pickle

    fit = Fit()
    compute_hash(fit)
    with stored_pickle.keeping(lambda value: True):
        assert stored_pickle.kept(fit) is None
        fit.coef[0] = 5.0
        payload = {"variables": {"fit": fit}}
        stored_pickle.nest(payload, fit)
        assert memory_backend._snapshot(payload).load()["variables"]["fit"].coef[0] == 5.0
