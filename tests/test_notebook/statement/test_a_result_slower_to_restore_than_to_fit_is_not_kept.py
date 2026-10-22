"""A result object that restores slower than it computes does not stay on disk.

A statsmodels results object holds its design matrix and rebuilds it with
patsy when it is unpickled: ``pooled`` (145 MiB, a 0.8 s fit) took 0.6 s to
load, ``fe_dummy`` (873 MiB) 4.6 s. cash sized such an object at its own 48
bytes, so the cost model priced its restore at nothing: it pickled 145 MiB
only to have the bytes ceiling refuse it, and wrote values whose hit was no
faster than their fit.

Now the object is sized by the frames and arrays it holds, and the time
its restore takes -- ``__setstate__`` included -- is what decides. The RAM
tier keeps it as its pickle bytes without reading them back, so nothing of
its class was read when the first one is stored: it is written to disk
judged by the cost model. Its first restore is the reading. A restore no
faster than the fit takes it off disk, and the class remembers the rate,
so the next statement making one is refused before anything is pickled.
Counted by ``__setstate__`` calls, with sleeps far
apart.

One slow reading is not enough: an unpickling that something else held up
(a full garbage collection, the writer thread holding the interpreter lock)
read 3.3 s in p20 for a graph that unpickles in 0.25 s, and its 1.4 s build
was kept nowhere, so after a restart it and everything keyed on it ran
again. A reading that would drop a value is taken twice and the faster one
kept.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend
from cash.sizing import pickled_size_estimate
from tests._cell_driver import run_cash_cell

#: A fit's compute, and what unpickling one takes when it is slow to load.
COMPUTE_S = 0.3
SLOW_LOAD_S = 0.9

#: What the tests import while a restore runs.
_IMPORTED = "_cash_test_imported_during_a_restore_"

#: ``__setstate__`` calls, by the test's own clock.
LOADS: list[float] = []
LOAD_S = [0.0]
#: One-off stalls, each taken by the next ``__setstate__`` in place of ``LOAD_S``.
STALLS: list[float] = []
#: Imports, each done by the next ``__setstate__``: ``(name, on another thread)``.
IMPORTS: list[tuple[str, bool]] = []


class Fit:
    """A results object: a few attributes, one of them a 4 MiB design matrix,
    and an unpickling that rebuilds something."""

    def __init__(self, rows: int) -> None:
        self.model = {"exog": np.ones((rows, 4)), "formula": "y ~ x"}
        self.params = np.arange(4.0)

    def __setstate__(self, state: dict) -> None:
        seconds = STALLS.pop(0) if STALLS else LOAD_S[0]
        if IMPORTS:
            _import(*IMPORTS.pop(0))
        time.sleep(seconds)
        LOADS.append(seconds)
        self.__dict__.update(state)


def _import(name: str, elsewhere: bool) -> None:
    """A module imported while a restore runs: by the restore itself, or by
    another thread, done before the restore ends."""

    def load() -> None:
        sys.modules[name] = types.ModuleType(name)

    if not elsewhere:
        load()
        return
    other = threading.Thread(target=load)
    other.start()
    other.join()


SETUP = f"""
import time

def fit_model(rows=1 << 17):
    time.sleep({COMPUTE_S})
    return Fit(rows)
"""


@pytest.fixture
def tiers(cash_magics, cash_instance, mock_shell, tmp_path):
    """``cash_magics`` over RAM and disk, with nothing learned about copies
    before the test and nothing left after it."""
    rates = getattr(memory_backend, "_ROUND_TRIP_RATES", {})
    saved = dict(rates)
    rates.clear()
    LOADS.clear()
    STALLS.clear()
    IMPORTS.clear()
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "nbcache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    mock_shell.user_ns["Fit"] = Fit
    run_cash_cell(cash_magics, SETUP)
    yield cash_magics, ram, disk
    disk.shutdown()
    getattr(cash_instance.backend, "shutdown", lambda: None)()
    rates.clear()
    rates.update(saved)
    LOAD_S[0] = 0.0
    STALLS.clear()
    IMPORTS.clear()
    for name in [name for name in sys.modules if name.startswith(_IMPORTED)]:
        del sys.modules[name]


def _disk_bytes(disk) -> int:
    """What the disk tier holds, once its writes are done."""
    disk._writes.wait_all()
    return sum(int(meta.get("size") or 0) for meta in disk.list_entries() or ())


class TestSizing:
    def test_an_object_is_sized_by_the_arrays_it_holds(self):
        fit = Fit(1 << 17)
        assert pickled_size_estimate(fit) >= fit.model["exog"].nbytes

    def test_an_object_holding_itself_is_sized_once(self):
        fit = Fit(1 << 10)
        fit.model["owner"] = fit
        assert fit.model["exog"].nbytes <= pickled_size_estimate(fit) < 2 * fit.model["exog"].nbytes


def _restart(cash_instance, disk, tmp_path):
    """A new process over the same cache folder: an empty RAM tier, and
    nothing learned about copies."""
    disk.shutdown()
    memory_backend._ROUND_TRIP_RATES.clear()
    fresh = FileBackend(str(tmp_path / "nbcache"), flush_interval=0)
    cash_instance.backend = TieredBackend([InMemoryBackend(), fresh])
    return fresh


def _rerun_restores(magics) -> bool:
    """Re-run the fit; True when it was restored (an object of it was
    unpickled: storing one never unpickles it), False when it computed."""
    loads = len(LOADS)
    run_cash_cell(magics, "fit = fit_model()")
    return len(LOADS) > loads


def test_a_result_that_loads_slower_than_its_fit_leaves_disk_after_its_first_restore(tiers, cash_instance, tmp_path):
    """Nothing of its class was read back when it was stored, so it is
    written, judged by the cost model. Its first restore is the reading:
    slower than the fit (twice, so not a stall), it is dropped from disk,
    and the next run computes it."""
    magics, _ram, disk = tiers
    LOAD_S[0] = SLOW_LOAD_S
    run_cash_cell(magics, "fit = fit_model()")
    assert not LOADS, "it was read back before it was written"
    assert _disk_bytes(disk) >= Fit(1 << 17).model["exog"].nbytes
    disk = _restart(cash_instance, disk, tmp_path)
    assert _rerun_restores(magics), "the first restore did not come from disk: the test checks nothing"
    assert len(LOADS) == 2, "a slow restore was not read again before it was dropped"
    assert _disk_bytes(disk) < 1 << 20, "a result restoring slower than its fit stayed on disk"
    _restart(cash_instance, disk, tmp_path)
    assert not _rerun_restores(magics), "the dropped result was restored again"


def test_the_next_one_is_refused_before_it_is_copied(tiers, cash_instance, tmp_path, mock_shell, monkeypatch):
    """The slow restore taught the class: the next result of it is refused
    before anything of it is kept."""
    magics, _ram, disk = tiers
    LOAD_S[0] = SLOW_LOAD_S
    run_cash_cell(magics, "fit = fit_model()")
    _restart(cash_instance, disk, tmp_path)
    run_cash_cell(magics, "fit = fit_model()")
    kept: list[str] = []
    snapshot = memory_backend._snapshot
    monkeypatch.setattr(memory_backend, "_snapshot", lambda value: kept.append(1) or snapshot(value))
    run_cash_cell(magics, "other = fit_model()")
    assert isinstance(mock_shell.user_ns["other"], Fit)
    assert not kept, "the second result was pickled to keep although its class restores slower than it fits"


def test_a_result_that_loads_fast_is_written(tiers):
    """The control: the same object, unpickled at once, is worth its disk."""
    magics, _ram, disk = tiers
    run_cash_cell(magics, "fit = fit_model()")
    assert _disk_bytes(disk) >= Fit(1 << 17).model["exog"].nbytes


def test_one_stalled_load_does_not_take_a_result_off_disk(tiers, cash_instance, tmp_path):
    """The first restore stalls, the second reading does not: the result
    stays on disk, as when nothing stalled."""
    magics, _ram, disk = tiers
    run_cash_cell(magics, "fit = fit_model()")
    disk = _restart(cash_instance, disk, tmp_path)
    STALLS.append(SLOW_LOAD_S)
    assert _rerun_restores(magics)
    assert SLOW_LOAD_S in LOADS, "the stall was not taken: the test measures nothing"
    assert _disk_bytes(disk) >= Fit(1 << 17).model["exog"].nbytes, "one stalled restore took the result off disk"


@pytest.mark.parametrize("elsewhere", [False, True], ids=["by the restore", "on another thread"])
def test_a_slow_restore_that_imported_is_judged_on_its_second_read(tiers, cash_instance, tmp_path, elsewhere):
    """A module imported during the first restore (the first statsmodels
    result after a restart imports statsmodels; any thread's import shows
    in sys.modules) leaves that reading unjudged. Read again, the second
    reading imports nothing and is the one judged: slower than the fit,
    the result leaves disk. Before, it was never judged and stayed."""
    magics, _ram, disk = tiers
    LOAD_S[0] = SLOW_LOAD_S
    run_cash_cell(magics, "fit = fit_model()")
    disk = _restart(cash_instance, disk, tmp_path)
    IMPORTS.append((_IMPORTED + "first", elsewhere))
    assert _rerun_restores(magics), "the first restore did not come from disk: the test checks nothing"
    assert len(LOADS) == 2, "a slow restore that imported was not read again"
    assert _disk_bytes(disk) < 1 << 20, "a result restoring slower than its fit stayed on disk"


def test_a_restore_importing_on_both_reads_is_not_judged(tiers, cash_instance, tmp_path):
    """Imports during both readings leave neither judged: the result
    stays on disk, as a restore that imported always did."""
    magics, _ram, disk = tiers
    LOAD_S[0] = SLOW_LOAD_S
    run_cash_cell(magics, "fit = fit_model()")
    disk = _restart(cash_instance, disk, tmp_path)
    IMPORTS.extend([(_IMPORTED + "first", True), (_IMPORTED + "second", True)])
    assert _rerun_restores(magics)
    assert len(LOADS) == 2
    assert _disk_bytes(disk) >= Fit(1 << 17).model["exog"].nbytes, "a reading that imported was judged"


@pytest.mark.parametrize("how", ["raises", "vanished"])
def test_a_second_read_that_fails_after_an_import_leaves_the_restore_unjudged(
    tiers, cash_instance, tmp_path, monkeypatch, how
):
    """The first reading imported, and the second failed (a MemoryError
    holding two copies, an unpickling error) or found the entry gone: its
    time is no reading. It was taken as an infinitely slow one, the entry
    left disk and the class was refused from then on."""
    magics, _ram, disk = tiers
    run_cash_cell(magics, "fit = fit_model()")
    disk = _restart(cash_instance, disk, tmp_path)
    STALLS.append(SLOW_LOAD_S)
    IMPORTS.append((_IMPORTED + "failed", True))
    fired, real_get = [], disk.get

    def get(key):
        if LOADS and not fired:  # the second reading
            fired.append(1)
            if how == "raises":
                raise OSError("a read error on the second reading")
            disk.delete(key)
        return real_get(key)

    monkeypatch.setattr(disk, "get", get)
    assert _rerun_restores(magics)
    assert fired, "the restore was not read again: the test checks nothing"
    rates = memory_backend._ROUND_TRIP_RATES
    assert not any(rate == float("inf") for kept in rates.values() for rate in kept), dict(rates)
    if how == "raises":
        assert _disk_bytes(disk) >= Fit(1 << 17).model["exog"].nbytes, "a failed reading took the result off disk"
