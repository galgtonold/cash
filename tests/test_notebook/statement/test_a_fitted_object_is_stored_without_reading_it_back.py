"""A fitted object a statement leaves is stored without being read back.

``fe_dummy = smf.ols(...).fit()`` left an object of a class of its own,
and the RAM tier stored it by a pickle round trip: the unpickling rebuilt
its design matrices, 9.5 s on top of a 10.6 s fit, for a value the cold
run never read again. The tier now keeps the pickle bytes, and a hit
unpickles them; the first hit, from memory or from disk, teaches its class's
rate, which judges the next one.

What a statement hands out stays the value it computed: a cell that edits
the object afterwards, in Python or in an array's memory, does not change
what a re-run restores. ``# @cash:persist`` writes it to disk whatever the
cost model says.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend
from tests._cell_driver import run_cash_cell

COMPUTE_S = 0.3
LOADS: list[int] = []


class Fit:
    """A results object holding a 4 MiB design matrix."""

    def __init__(self, rows: int) -> None:
        self.model = {"exog": np.ones((rows, 4)), "terms": ["x"]}
        self.params = np.arange(4.0)

    def __setstate__(self, state: dict) -> None:
        LOADS.append(1)
        self.__dict__.update(state)


SETUP = f"""
import time
import numpy as np

def fit_model(rows=1 << 17):
    time.sleep({COMPUTE_S})
    return Fit(rows)
"""


@pytest.fixture
def tiers(cash_magics, cash_instance, mock_shell, tmp_path):
    rates = memory_backend._ROUND_TRIP_RATES
    saved = dict(rates)
    rates.clear()
    LOADS.clear()
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "nbcache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    mock_shell.user_ns["Fit"] = Fit
    run_cash_cell(cash_magics, SETUP)
    yield cash_magics, ram, disk
    disk.shutdown()
    rates.clear()
    rates.update(saved)


def _on_disk(disk) -> int:
    disk._writes.wait_all()
    return sum(int(meta.get("size") or 0) for meta in disk.list_entries() or ())


def test_an_edit_after_the_statement_is_not_what_a_rerun_restores(tiers, mock_shell):
    magics, _ram, _disk = tiers
    run_cash_cell(magics, "fit = fit_model()")
    run_cash_cell(magics, "fit.params[0] = 99.0\nnp.add(fit.model['exog'], 1.0, out=fit.model['exog'])")
    run_cash_cell(magics, "fit.model['terms'].append('edited')")
    loads = len(LOADS)
    started = time.perf_counter()
    run_cash_cell(magics, "fit = fit_model()")
    assert time.perf_counter() - started < COMPUTE_S, "the re-run computed: the test checks no restore"
    assert len(LOADS) > loads
    fit = mock_shell.user_ns["fit"]
    assert fit.params[0] == 0.0
    assert fit.model["exog"].sum() == fit.model["exog"].size
    assert fit.model["terms"] == ["x"]


def test_a_stored_object_never_served_is_never_read_back(tiers):
    """Not on the store, and not before its disk write: the first of its
    class is judged by the cost model, and its first restore is the reading."""
    magics, _ram, disk = tiers
    run_cash_cell(magics, "first = fit_model()")
    run_cash_cell(magics, "second = fit_model(1 << 16)")
    assert not LOADS, "a stored object was read back although nothing served it"
    assert _on_disk(disk) >= Fit(1 << 17).model["exog"].nbytes


def test_persist_writes_it_whatever_the_cost_model_says(tiers):
    magics, _ram, disk = tiers
    run_cash_cell(magics, "def quick():\n    return Fit(1 << 17)")
    run_cash_cell(magics, "# @cash:persist\nquick_fit = quick()")
    assert _on_disk(disk) >= Fit(1 << 17).model["exog"].nbytes
