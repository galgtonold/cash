"""An object no RAM hit could serve faster than its compute is not copied.

The RAM tier judged an object of a class of its own by what unpickling
objects of its class took in this process. Before the first of a class was
stored there was no reading, so it was copied whatever it was -- a graph
built in a millisecond and as big as a cached model, pickled only to be
worth nothing. With no reading the fitted cost model's RAM price of reading
an object of its size back stands in: a value that would be no faster to
serve than to compute is refused before anything is pickled, and one that
is clearly worth it is kept. An entry the caller decided to keep is kept
whatever the price.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash import cost_model
from cash.backends import InMemoryBackend, memory_backend

PICKLED: list[int] = []


class Graph:
    """A graph-like object, 16 MiB of node data."""

    def __init__(self) -> None:
        self.nodes = np.ones(1 << 21)

    def __getstate__(self) -> dict:
        PICKLED.append(1)
        return self.__dict__


#: What the cost model says reading one back into RAM takes.
READ_BACK_S = cost_model.estimated_restore_time("Graph", Graph().nodes.nbytes, "ram")


@pytest.fixture(autouse=True)
def no_readings():
    rates = memory_backend._ROUND_TRIP_RATES
    saved = dict(rates)
    rates.clear()
    PICKLED.clear()
    yield
    rates.clear()
    rates.update(saved)


def test_one_cheaper_to_build_than_to_read_back_is_not_copied():
    ram = InMemoryBackend()
    assert ram.set("g", Graph(), {"execution_time": READ_BACK_S / 10}) is False
    assert not PICKLED, "it was pickled before it was refused"
    assert ram.get("g") == (None, None)


def test_one_clearly_worth_keeping_is_kept():
    ram = InMemoryBackend()
    ram.set("g", Graph(), {"execution_time": READ_BACK_S * 10})
    assert isinstance(ram.get("g")[1], Graph)


@pytest.mark.parametrize("kept_by", ["force_persist", "decorator_entry", "copy_required"])
def test_one_the_caller_keeps_is_kept_whatever_it_costs(kept_by):
    ram = InMemoryBackend()
    ram.set("g", Graph(), {"execution_time": READ_BACK_S / 10, kept_by: True})
    assert isinstance(ram.get("g")[1], Graph)
