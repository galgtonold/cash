"""A statement's entry holds an output's kept pickle beside its other parts.

A statement's store takes the pickle the digest of an output made a moment
before (`stored_pickle.kept`), nested in the entry's own, when nothing else
in the entry is part of that output: two pickles restore an object both
hold as two objects. That was proved only for immutable data, so an entry
holding an echoed value or a second output beside a 900 MB model copied it
the slow way (8.5 s for one store). The digest now proves it while its
pickle is still being made: no object but immutable data, numpy dtypes and
objects pickled by name is in both pickles (`stored_pickle.apart`).

And a change check that ran out of its budget while pickling such an output
(`value_hash.budgeted_hash`) gave up that pickle, and the store copied the
value the slow way: 9.8 s for a statsmodels result whose pickle needed 0.7
s. The pickle is now left to the store (`keeping`'s *past_budget*), which
makes it when it keeps the value as bytes, and nests it as a kept one. It
is not made on in the digest: a store that refuses the value after it was
digested (an openpyxl workbook its own sheets hold) threw 8 s of pickling
away.

What must hold: an object the echo or another output shares with the
output comes back shared, from the RAM tier and from disk after a restart;
what is stored is the value as the statement left it; each value is pickled
once whole.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cash import stored_pickle, value_hash
from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend
from cash.notebook.statement.run import ECHO_FIELD
from cash.value_hash import budgeted_hash, compute_hash
from tests._cell_driver import run_cash_cell

PICKLED: list[int] = []


class Fit:
    """A results object holding a 2 MiB array, counted each time it is pickled."""

    def __init__(self, rows: int = 1 << 18) -> None:
        self.coef = np.arange(rows, dtype=float)
        self.names = ["a", "b"]

    def __reduce_ex__(self, protocol):
        PICKLED.append(1)
        return object.__reduce_ex__(self, protocol)


class SlowFit(Fit):
    """A `Fit` whose pickling takes a while, past a short budget."""

    def __reduce_ex__(self, protocol):
        time.sleep(0.05)
        return super().__reduce_ex__(protocol)


#: The clock `stalled_clock` gives the digest and the store: still but for
#: a `TickingFit`'s pickling.
CLOCK = [0.0]


class TickingFit(Fit):
    """A `Fit` whose pickling takes a second of `stalled_clock`'s time."""

    def __reduce_ex__(self, protocol):
        CLOCK[0] += 1.0
        return super().__reduce_ex__(protocol)


@pytest.fixture
def stalled_clock(monkeypatch):
    """A clock only a `TickingFit`'s pickling moves. With the real one, a
    busy machine spent a 0.01 s budget before the pickle began, and the
    digest gave up there rather than while pickling."""
    CLOCK[0] = 0.0
    monkeypatch.setattr(value_hash, "perf_counter", lambda: CLOCK[0])
    monkeypatch.setattr(stored_pickle, "perf_counter", lambda: CLOCK[0])


class SlowFits:
    """Four `SlowFit`s."""

    def __init__(self) -> None:
        self.fits = [SlowFit(1 << 12) for _ in range(4)]


# Slow enough to be stored; one statement binding ``fit`` and echoing a value.
ECHO_APART = "(fit := Fit() if time.sleep(0.3) is None else None).names[:]"
ECHO_SHARED = "(fit := Fit() if time.sleep(0.3) is None else None).names"


@pytest.fixture
def tiers(cash_magics, cash_instance, mock_shell, tmp_path):
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "nbcache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    run_cash_cell(cash_magics, f"import time\nfrom {__name__} import Fit, SlowFit, SlowFits")
    PICKLED.clear()
    yield cash_magics, ram, disk, tmp_path / "nbcache"
    disk.shutdown()


def _entry(ram, name):
    """``(key, payload as a hit reads it, kept as pickle bytes?)`` of the
    RAM tier's entry holding *name*."""
    for key, (_meta, stored) in list(ram._store.items()):
        payload = ram.get(key)[1]
        if name in payload.get("variables", {}):
            return key, payload, type(stored) is memory_backend._Pickled
    return None, None, False


def _from_disk(cache_dir, key):
    """The entry under *key* as a new process reads it from disk."""
    fresh = FileBackend(str(cache_dir), flush_interval=0)
    try:
        return fresh.get(key)[1]
    finally:
        fresh.shutdown()


def test_an_echo_apart_from_the_output_takes_the_digests_pickle(tiers, mock_shell):
    magics, ram, disk, _cache_dir = tiers
    run_cash_cell(magics, ECHO_APART)
    disk._writes.wait_all()
    _key, payload, as_bytes = _entry(ram, "fit")
    assert payload is not None and payload.get(ECHO_FIELD) == ["a", "b"], "no echo in the entry: proves nothing"
    assert as_bytes, "not kept as pickle bytes"
    assert PICKLED == [1], "the output was pickled again for its store"
    assert np.array_equal(payload["variables"]["fit"].coef, mock_shell.user_ns["fit"].coef)


@pytest.mark.parametrize("source", ["ram", "disk"])
def test_an_echo_sharing_an_object_with_the_output_comes_back_sharing_it(tiers, source):
    magics, ram, disk, cache_dir = tiers
    run_cash_cell(magics, ECHO_SHARED)
    disk._writes.wait_all()
    key, payload, _as_bytes = _entry(ram, "fit")
    assert payload is not None and ECHO_FIELD in payload, "no echo in the entry: proves nothing"
    if source == "disk":
        payload = _from_disk(cache_dir, key)
    assert payload[ECHO_FIELD] is payload["variables"]["fit"].names


def test_a_second_output_apart_from_the_first_takes_the_digests_pickle(tiers, mock_shell):
    magics, ram, disk, _cache_dir = tiers
    run_cash_cell(magics, "fit, rows = (Fit(), {'r': [1, 2]}) if time.sleep(0.3) is None else (None, None)")
    disk._writes.wait_all()
    _key, payload, as_bytes = _entry(ram, "fit")
    assert payload is not None and payload["variables"]["rows"] == {"r": [1, 2]}
    assert as_bytes and PICKLED == [1]


@pytest.mark.parametrize("source", ["ram", "disk"])
def test_a_second_output_sharing_an_object_comes_back_sharing_it(tiers, source):
    magics, ram, disk, cache_dir = tiers
    run_cash_cell(magics, "fit, box = [(f, {'names': f.names}) for f in [Fit()] if time.sleep(0.3) is None][0]")
    disk._writes.wait_all()
    key, payload, _as_bytes = _entry(ram, "fit")
    assert payload is not None and "box" in payload["variables"], "not stored together: proves nothing"
    if source == "disk":
        payload = _from_disk(cache_dir, key)
    assert payload["variables"]["box"]["names"] is payload["variables"]["fit"].names


def test_a_change_after_the_store_is_not_in_the_entry(tiers, mock_shell):
    magics, ram, disk, _cache_dir = tiers
    run_cash_cell(magics, ECHO_APART)
    disk._writes.wait_all()
    fit = mock_shell.user_ns["fit"]
    fit.coef[0] = -1.0
    fit.names.append("z")
    _key, payload, _as_bytes = _entry(ram, "fit")
    assert payload["variables"]["fit"].coef[0] == 0.0
    assert payload["variables"]["fit"].names == ["a", "b"] and payload[ECHO_FIELD] == ["a", "b"]


def test_a_change_check_past_its_budget_leaves_the_pickle_to_the_store(tiers, mock_shell, monkeypatch):
    from cash.notebook.statement import processor

    magics, ram, disk, _cache_dir = tiers
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 0.02)
    # All the time it needs to make the pickle: by default half of the 0.3 s
    # statement, and a busy runner's 0.05 s sleep in `SlowFit` overran it,
    # so the store copied the value the other way. Running out of that time
    # is `test_a_pickle_left_to_the_store_is_made_within_its_time`.
    monkeypatch.setattr(processor, "PAST_BUDGET_SHARE", 100.0)
    run_cash_cell(magics, "fit = SlowFit() if time.sleep(0.3) is None else None")
    disk._writes.wait_all()
    assert value_hash.is_costly(mock_shell.user_ns["fit"]), "the check did not run out of its budget: proves nothing"
    _key, payload, as_bytes = _entry(ram, "fit")
    # Once begun by the check, which gave up; once whole, by the store.
    assert as_bytes and PICKLED == [1, 1], "the store copied the output another way"
    assert np.array_equal(payload["variables"]["fit"].coef, mock_shell.user_ns["fit"].coef)


def test_a_change_check_past_its_budget_makes_no_pickle_a_store_refuses(tiers, mock_shell, monkeypatch):
    from cash.notebook.statement import processor

    magics, ram, _disk, _cache_dir = tiers
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 0.02)
    # Refused after its change check, as an openpyxl workbook its own sheets
    # hold is: cash cannot tell another holder from them.
    refuse = processor.StatementProcessor._refuse_unrestorable_outputs

    def refusing(self, run, *args, **kwargs):
        found = refuse(self, run, *args, **kwargs)
        if "fit" in run.outputs:
            run.skip_cache = True
        return found

    monkeypatch.setattr(processor.StatementProcessor, "_refuse_unrestorable_outputs", refusing)
    begun, made = [], []
    real = stored_pickle.pickle_value

    def counted(value, **kwargs):
        if isinstance(value, SlowFits):
            begun.append(kwargs)
        part = real(value, **kwargs)  # raises past its budget: not counted
        if isinstance(value, SlowFits):
            made.append(part)
        return part

    monkeypatch.setattr(stored_pickle, "pickle_value", counted)
    run_cash_cell(magics, "fit = SlowFits() if time.sleep(0.3) is None else None")
    assert value_hash.is_costly(mock_shell.user_ns["fit"]), "the check did not run out of its budget: proves nothing"
    assert begun, "no store pickle begun by its check: proves nothing"
    assert _entry(ram, "fit")[0] is None, "stored: proves nothing"
    assert made == [], "pickled on for a store that did not take it"


def test_a_pickle_left_to_the_store_is_made_within_its_time(tiers, cash_instance, mock_shell, monkeypatch):
    from cash.notebook.statement import processor

    magics, _ram, _disk, _cache_dir = tiers
    ram = cash_instance.backend = InMemoryBackend()  # no disk tier to pickle it its own way
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 0.02)
    # 0.3 s of statement: at most 0.003 s to pickle, and pickling takes 0.05 s.
    monkeypatch.setattr(processor, "PAST_BUDGET_SHARE", 0.01)
    run_cash_cell(magics, "fit = SlowFit() if time.sleep(0.3) is None else None")
    assert value_hash.is_costly(mock_shell.user_ns["fit"]), "the check did not run out of its budget: proves nothing"
    assert _entry(ram, "fit")[0] is None, "kept in RAM past its time"
    assert PICKLED == [1, 1], "copied another way after its pickle ran out of time"


# -- the proof itself ---------------------------------------------------------


def _apart(value, others, past_budget=None):
    """Digest *value* as a store's keeping does, and say which of *others*
    it found apart."""
    with stored_pickle.keeping(lambda v: v is value, lambda v: others, past_budget):
        compute_hash(value)
        assert stored_pickle.kept(value) is not None
        return [stored_pickle.apart(value, other) for other in others]


def test_only_objects_that_restore_alike_apart_may_be_shared():
    fit = Fit()
    names = list(fit.names)  # the same strings: alike apart
    assert _apart(fit, [fit.names, names, {"coef": fit.coef}, [Fit, np.dtype(float)], [fit]]) == [
        False,  # the list itself
        True,
        False,  # the array
        True,  # a class and a dtype: by name, and identityless
        False,  # the value itself, inside another
    ]


def test_an_other_too_big_to_check_is_not_apart(monkeypatch):
    monkeypatch.setattr(stored_pickle, "_APART_MAX_BYTES", 1000)
    fit = Fit()
    assert _apart(fit, [[str(i) for i in range(1000)]]) == [False]


def test_a_budgeted_digest_leaves_its_pickle_to_the_store_within_its_bytes(stalled_clock):
    for past, made in ((None, None), (1000, False), (1 << 30, True)):
        fit = TickingFit()
        with stored_pickle.keeping(lambda v, fit=fit: v is fit, None, past):
            assert budgeted_hash(fit, seconds=0.01) is None, "within its budget: proves nothing"
            assert stored_pickle.kept(fit) is None, "made on in the digest"
            assert stored_pickle.pending(fit) is (past is not None)
            payload = {"variables": {"fit": fit}}
            stored_pickle.nest(payload, fit)
            assert stored_pickle.nesting(payload) is (made is not None)
            if made is False:
                with pytest.raises(stored_pickle.HashBudgetExceeded):
                    memory_backend._snapshot(payload)
            elif made:
                back = memory_backend._snapshot(payload).load()["variables"]["fit"]
                assert np.array_equal(back.coef, fit.coef) and back.names == fit.names
                assert stored_pickle.kept(fit).digest is None


def test_a_pickle_left_to_the_store_is_nested_only_apart_from_the_rest(stalled_clock):
    fit = TickingFit()
    for others, nests in (([["x"]], True), ([fit.names], False)):
        with stored_pickle.keeping(lambda v, fit=fit: v is fit, None, 1 << 30):
            assert budgeted_hash(fit, seconds=0.01) is None, "within its budget: proves nothing"
            payload = {"variables": {"fit": fit, "other": others[0]}}
            stored_pickle.nest(payload, fit, others)
            assert (stored_pickle.nested(payload) is not None) is nests
