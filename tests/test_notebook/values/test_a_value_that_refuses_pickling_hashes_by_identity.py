"""A value whose pickling raises is hashed by identity, never fails the cell.

``sr = random.SystemRandom()``: the new variable is hashed after the cell
runs, by pickling it, and ``SystemRandom.__reduce__`` asks for a state the
system entropy source does not have (``NotImplementedError``). That error
reached the user and failed a cell plain Python runs. A user's own
``__reduce__`` or ``__getstate__`` can raise anything; the value then has no
content hash and gets the identity tier, which callers recognise as
content-blind. An interrupt still stops the hash.
"""

from __future__ import annotations

import random

import pytest

from cash.mutation_fingerprint import mutation_fingerprint
from cash.value_hash import compute_hash, is_identity_fallback_hash
from tests._cell_driver import run_cash_cell


class _RefusesPickling:
    def __reduce__(self):
        raise RuntimeError("no state")


class _Interrupts:
    def __reduce__(self):
        raise KeyboardInterrupt


@pytest.mark.parametrize("value", [random.SystemRandom(), _RefusesPickling()], ids=["SystemRandom", "own reduce"])
def test_compute_hash_falls_back_to_identity(value):
    assert is_identity_fallback_hash(value, compute_hash(value))


def test_mutation_fingerprint_reports_it_unobservable():
    assert mutation_fingerprint(random.SystemRandom()) is None


def test_an_interrupt_still_stops_the_hash():
    with pytest.raises(KeyboardInterrupt):
        compute_hash(_Interrupts())


def test_a_cell_binding_system_random_runs(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "import random\nsr = random.SystemRandom()\ndraw = sr.randrange(10)")
    assert isinstance(mock_shell.user_ns["sr"], random.SystemRandom)
    assert 0 <= mock_shell.user_ns["draw"] < 10
