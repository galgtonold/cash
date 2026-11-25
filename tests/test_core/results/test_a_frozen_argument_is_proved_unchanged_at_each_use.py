"""A ``frozen=True`` result passed on as an argument is proved unchanged at
each use where cash can, instead of audited now and then.

The audit compared a full hash at the object's 8th use and every 64th after
that, and the shape (lengths) at every use: a value overwritten in place at
the same length was keyed by its producer for up to 63 uses, each a stale
key. A frozen result is now proved unchanged on every use the way a frozen
hit's kept object is (`backends.frozen_handback`): a copy-on-write pandas
frame by its blocks, a dict, list or plain instance of 64 KiB or more by the
C extension's watchers. A smaller one is hashed at every use. Only what
nothing can prove (no C extension, a type the watchers cannot follow) keeps
the audit's schedule, which ``key.frozen_audit`` times.
"""

from __future__ import annotations

import types
import warnings
import weakref

import numpy as np
import pandas as pd
import pytest

from cash import Cash, phase_times, value_watch
from cash.backends import InMemoryBackend
from cash.decorator import frozen
from cash.decorator.frozen import FROZEN_AUDIT_EVERY, FROZEN_AUDIT_FIRST

cow = pytest.mark.skipif(
    int(pd.__version__.split(".", 1)[0]) < 3 and pd.options.mode.copy_on_write is not True,
    reason="a pandas table is proved unchanged only under copy-on-write",
)

watched = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="a dict, list or instance is proved unchanged by the C extension's watchers",
)

N = 20_000  # floats: well over the watchers' 64 KiB floor


@pytest.fixture
def cash(monkeypatch):
    monkeypatch.setattr(phase_times, "ON", True)
    return Cash(backend=InMemoryBackend(), register_magic=False)


def _audits(func) -> int:
    times = func.cache_info()["phase_times"]
    return sum(v["count"] for k, v in times.items() if k.endswith(".key.frozen_audit"))


def _use(func, *args):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = func(*args)
    return value, [str(w.message) for w in seen if "KEY-FROZEN-MUTATED" in str(w.message)]


class Model:
    def __init__(self, n):
        self.weights = [float(i) for i in range(n)]


@watched
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d["k3"].__setitem__(0, 1e6), id="a-value-inside"),
        pytest.param(lambda d: d.__setitem__("k3", [1e6, 4.0]), id="a-value-rebound"),
    ],
)
def test_a_dict_overwritten_in_place_is_caught_at_its_next_use(cash, mutate):
    @cash.cache(frozen=True)
    def load(n):
        return {f"k{i}": [float(i), float(i + 1)] for i in range(n)}

    @cash.cache
    def total(table, bias):
        return sum(v[0] for v in table.values()) + bias

    table = load(N // 2)
    assert _use(total, table, 0) == (total.__wrapped__(table, 0), [])
    assert _use(total, table, 0)[1] == []
    mutate(table)
    after, said = _use(total, table, 0)
    assert after == sum(v[0] for v in table.values()), "the consumer was served the pre-change answer"
    assert said, "the change was not reported"
    assert _use(total, table, 0) == (after, [])


@watched
def test_a_list_overwritten_in_place_is_caught_at_its_next_use(cash, monkeypatch):
    class Shell:
        user_ns: dict = {}

    shell = Shell()
    shell.user_ns = {}
    # A list has no weak reference: the watchers follow one a namespace holds.
    monkeypatch.setattr(value_watch, "_SHELLS", [weakref.ref(shell)])

    @cash.cache(frozen=True)
    def load(n):
        return [float(i) for i in range(n)]

    @cash.cache
    def total(rows, bias):
        return sum(rows) + bias

    rows = shell.user_ns["rows"] = load(N)
    first = sum(rows)
    assert _use(total, rows, 0) == (first, [])
    assert _use(total, rows, 0) == (first, [])
    rows[3] = 1e6
    after, said = _use(total, rows, 0)
    assert after == sum(rows) != first
    assert said


@watched
def test_an_instance_overwritten_in_place_is_caught_at_its_next_use(cash):
    @cash.cache(frozen=True)
    def train(n):
        return Model(n)

    @cash.cache
    def score(model, bias):
        return sum(model.weights) + bias

    model = train(N)
    first = sum(model.weights)
    assert _use(score, model, 0) == (first, [])
    assert _use(score, model, 0) == (first, [])
    model.weights[5] = 1e6
    after, said = _use(score, model, 0)
    assert after == sum(model.weights) != first
    assert said


@cow
def test_a_frame_cell_overwritten_in_place_is_caught_at_its_next_use(cash):
    @cash.cache(frozen=True)
    def load(n):
        return pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.ones(n)})

    @cash.cache
    def total(frame, bias):
        return float(frame["a"].sum()) + bias

    frame = load(1000)
    first = float(frame["a"].sum())
    assert _use(total, frame, 0) == (first, [])
    assert _use(total, frame, 0) == (first, [])
    frame.iloc[3, 0] = 1e6
    after, said = _use(total, frame, 0)
    assert after == float(frame["a"].sum()) != first
    assert said


@watched
def test_a_proved_result_is_never_hashed_again(cash, monkeypatch):
    # Every full hash the audit takes, timed or not.
    pickled = []
    real = frozen.kept_state.dumps
    monkeypatch.setattr(frozen, "kept_state", types.SimpleNamespace(dumps=lambda obj: pickled.append(obj) or real(obj)))

    @cash.cache(frozen=True)
    def load(n):
        return {f"k{i}": [float(i), float(i + 1)] for i in range(n)}

    @cash.cache(frozen=True)
    def train(n):
        return Model(n)

    @cash.cache
    def total(table, model, bias):
        return len(table) + len(model.weights) + bias

    table, model = load(N // 2), train(N)
    uses = FROZEN_AUDIT_EVERY * 2
    assert [total(table, model, i % 2) for i in range(uses)] == [N // 2 + N + i % 2 for i in range(uses)]
    assert total.cache_info()["misses"] == 2
    assert _audits(total) == 0
    assert not any(obj is table or obj is model for obj in pickled)


@watched
def test_a_result_under_the_watchers_floor_is_hashed_at_every_use(cash):
    @cash.cache(frozen=True)
    def load():
        return {"a": 1, "b": 2}

    @cash.cache
    def total(table):
        return sum(table.values())

    table = load()
    assert [total(table) for _ in range(3)] == [3, 3, 3]
    assert _audits(total) == 3
    table["a"] = 100
    after, said = _use(total, table)
    assert after == 102
    assert said


def test_a_numpy_result_is_still_keyed_by_its_hash_and_read_only(cash):
    calls = []

    @cash.cache(frozen=True)
    def load(n):
        return np.arange(n, dtype=float)

    @cash.cache
    def total(arr, bias):
        calls.append(1)
        return float(arr.sum()) + bias

    arr = load(N)
    assert [total(arr, 0) for _ in range(FROZEN_AUDIT_FIRST * 2)] == [float(arr.sum())] * (FROZEN_AUDIT_FIRST * 2)
    assert len(calls) == 1
    assert _audits(total) == 0
    with pytest.raises(ValueError):
        arr[0] = 1.0


class Opaque:
    """A class with its own pickled state: the watchers cannot follow it."""

    def __init__(self, n):
        self.values = [float(i) for i in range(n)]

    def __getstate__(self):
        return {"values": self.values}

    def __setstate__(self, state):
        self.values = state["values"]


def test_a_result_nothing_proves_is_still_audited_on_schedule(cash):
    @cash.cache(frozen=True)
    def load(n):
        return Opaque(n)

    @cash.cache
    def total(obj, bias):
        return sum(obj.values) + bias

    obj = load(100)
    first = sum(obj.values)
    for _ in range(FROZEN_AUDIT_FIRST):
        assert _use(total, obj, 0) == (first, [])
    assert _audits(total) == 1  # the baseline, at the 8th use
    obj.values[3] = 1e6
    said = []
    for _ in range(FROZEN_AUDIT_EVERY - FROZEN_AUDIT_FIRST):
        _value, said = _use(total, obj, 0)
    assert _audits(total) == 2
    assert said, "the audit at the 64th use did not catch the change"
    assert _use(total, obj, 0) == (sum(obj.values), [])
