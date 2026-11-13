"""A long list of numpy scalars is digested over its values gathered into
one array (`value_hash._scalar_run_digest`), not pickled scalar by scalar.

A list of two million ``np.float64`` deltas, appended in a loop and holding
no lineage, was pickled whole before and after every statement that read
it: ``len(deltas)`` cost about 17 s. What must hold: such a read costs well
under a second, an append is still seen, and the digest keeps apart every
pair of lists the pickle kept apart (dtype, ``-0.0``, a NaN's payload, a
unit, one changed item).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cash import kept_state, value_hash
from tests._cell_driver import run_cash_cell

N = value_hash.SCALAR_RUN_MIN * 2

BUILD = "import numpy as np\ndeltas = []\nfor i in range(1_000_000):\n    deltas.append(np.float64(i) * 0.5)"


def test_a_read_of_a_list_of_a_million_scalars_is_quick(cash_magics, monkeypatch):
    # Built before cash is on: the list has no lineage, so a statement
    # reading it is keyed on its digest.
    exec(BUILD, cash_magics.shell.user_ns)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    pickled = []
    dumps = kept_state.dumps

    def counted(value, *args, **kwargs):
        if value is cash_magics.shell.user_ns["deltas"]:
            pickled.append(1)
        return dumps(value, *args, **kwargs)

    monkeypatch.setattr(kept_state, "dumps", counted)
    start = time.perf_counter()
    run_cash_cell(cash_magics, "n = len(deltas)")
    assert time.perf_counter() - start < 1.0
    assert not pickled
    assert cash_magics.shell.user_ns["n"] == 1_000_000

    run_cash_cell(cash_magics, "deltas.append(np.float64(-0.0))")
    run_cash_cell(cash_magics, "n = len(deltas)")
    run_cash_cell(cash_magics, "last = deltas[-1]")

    plain: dict = {}
    exec(BUILD + "\ndeltas.append(np.float64(-0.0))\nn = len(deltas)\nlast = deltas[-1]", plain)
    ns = cash_magics.shell.user_ns
    assert ns["n"] == plain["n"] == 1_000_001
    assert type(ns["last"]) is type(plain["last"]) and np.signbit(ns["last"]) == np.signbit(plain["last"])


def _digest(values):
    return value_hash.compute_hash(values)


def _floats(dtype, last=1.0):
    return [dtype(i) for i in range(N - 1)] + [dtype(last)]


def test_float64_and_float32_lists_digest_apart():
    assert _digest(_floats(np.float64)) != _digest(_floats(np.float32))
    assert _digest(_floats(np.float64)) == _digest(_floats(np.float64))


def test_negative_zero_and_zero_digest_apart():
    assert _digest(_floats(np.float64, -0.0)) != _digest(_floats(np.float64, 0.0))


def test_lists_differing_in_one_item_digest_apart():
    one = _floats(np.float64)
    other = list(one)
    other[N // 2] = np.float64(other[N // 2] + 1e-12)
    assert _digest(one) != _digest(other)


@pytest.mark.parametrize(
    "dtype, bits", [(np.float64, (0x7FF8000000000001, 0x7FF8000000000002)), (np.float32, (0x7FC00001, 0x7FC00002))]
)
def test_nans_of_other_payloads_digest_apart(dtype, bits):
    size = np.dtype(dtype).itemsize

    def nan(b):
        return np.frombuffer(b.to_bytes(size, "little"), dtype=dtype)[0]

    head = _floats(dtype)[:-1]
    assert _digest([*head, nan(bits[0])]) != _digest([*head, nan(bits[1])])


def test_timedeltas_of_other_units_digest_apart():
    seconds = [np.timedelta64(i, "s") for i in range(N)]
    millis = [np.timedelta64(i, "ms") for i in range(N)]
    assert _digest(seconds) != _digest(millis)
    # Units mixed: pickled, as before, and apart from each.
    mixed = [*seconds[:-1], millis[-1]]
    assert value_hash._scalar_run_digest(mixed) is None
    assert len({_digest(seconds), _digest(millis), _digest(mixed)}) == 3


def test_a_list_and_a_tuple_and_python_floats_digest_apart():
    floats = _floats(np.float64)
    plain = [float(x) for x in floats]
    assert len({_digest(floats), _digest(tuple(floats)), _digest(plain)}) == 3


def test_a_run_is_not_said_to_digest_over_its_pickle():
    floats = _floats(np.float64)
    assert not value_hash.digest_via_pickle(floats)
    assert value_hash.digest_via_pickle(floats[:10])
    assert value_hash.unshared_hash(floats) == _digest(floats)
