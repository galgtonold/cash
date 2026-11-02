"""The identity check reads each object of a column of sessions, lists of
``(action, time)`` records, once for its type: the leaves' types were read
again for the numbers they spell, and each level's lists counted in a pass
of their own, three passes over two million records more than needed after
each statement that rebinds the frame (student_8)."""

from __future__ import annotations

import builtins
import datetime

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import forget_flat_reads, identity_held_reason

pd = pytest.importorskip("pandas")

SESSIONS = 400
ACTIONS = 5


def _sessions() -> list:
    t0 = datetime.datetime(2020, 1, 1)
    return [
        (f"u{s % 7}", [(f"a{a}", t0 + datetime.timedelta(seconds=s + a)) for a in range(ACTIONS)])
        for s in range(SESSIONS)
    ]


def _count_type_reads(monkeypatch) -> list[int]:
    read = [0]

    def counting_map(func, *iterables):
        if func is not type:
            return builtins.map(func, *iterables)

        def types():
            for item in iterables[0]:
                read[0] += 1
                yield type(item)

        return types()

    monkeypatch.setattr(identity_holders, "map", counting_map, raising=False)
    return read


def test_each_record_and_leaf_of_the_column_is_read_once(monkeypatch):
    logs = _sessions()
    df = pd.DataFrame({"actions": [actions for _user, actions in logs]})
    read = _count_type_reads(monkeypatch)
    forget_flat_reads()
    assert identity_held_reason("df", df, {"df": df, "logs": logs}) is None
    records = SESSIONS * ACTIONS
    # The column's label too where it is a Python string (pandas 2 keeps
    # the labels in an object array; pandas 3 in Arrow text).
    labels = len(df.columns) if df.columns.dtype == object else 0
    # The lists twice (to tell them from flat records first), the records
    # once, and their two leaves each twice: for the level's types, and for
    # the texts among them.
    assert read[0] <= 2 * SESSIONS + records + 2 * 2 * records + labels


def test_an_id_spelled_in_a_record_is_still_found():
    target = object()
    logs = _sessions()
    logs[123][1][2] = ("a2", str(id(target)))
    df = pd.DataFrame({"actions": [actions for _user, actions in logs]})
    forget_flat_reads()
    assert identity_held_reason("df", df, {"df": df, "target": target}) is not None


def test_an_id_kept_as_a_number_in_a_record_beside_lists_is_still_found():
    target = object()
    logs = _sessions()
    logs[7][1].append([("nested", id(target))])
    df = pd.DataFrame({"actions": [actions for _user, actions in logs]})
    forget_flat_reads()
    assert identity_held_reason("df", df, {"df": df, "target": target}) is not None
