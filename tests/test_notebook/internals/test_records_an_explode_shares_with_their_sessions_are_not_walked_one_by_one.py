"""A frame whose object column holds ``(action, time)`` records, or lists
of them, that a list of sessions holds too is checked at C speed, not a
record at a time.

``df = pd.DataFrame(logs).explode('actions')`` leaves each record held by
its session's list and by the column. The share walk, the view search and
the alias check each took the column's records one by one: 10 to 30 s a
statement over two million of them. A tuple of values cannot change and
holds nothing that can, so no holder can rely on its identity, and a list
of such tuples holds no array. What can be shared still is found: a list
of the column the sessions hold, an array view in a cell.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from cash.notebook import shared_objects as so

T0 = datetime.datetime(2019, 10, 1)


def _sessions(n: int = 2000) -> list:
    return [
        (f"@User{i % 50}", [(f"Action_{j % 11}", T0 + datetime.timedelta(minutes=j)) for j in range(20)])
        for i in range(n)
    ]


def _exploded(logs: list) -> pd.DataFrame:
    return pd.DataFrame(logs, columns=["user", "actions"]).explode("actions")


def test_records_shared_with_their_sessions_leave_nothing_to_look_for():
    logs = _sessions()
    df = _exploded(logs)
    assert df["actions"].iloc[0] is logs[0][1][0]
    assert not so.children_of(df)
    ns = {"logs": logs, "df": df}
    found, shared = so.share_group(["df"], ns, ns)
    assert (set(found), shared) == (set(), set())


def test_a_list_of_records_the_sessions_hold_is_still_found():
    logs = _sessions()
    df = pd.DataFrame(logs, columns=["user", "actions"])
    assert df["actions"].iloc[0] is logs[0][1]
    ns = {"logs": logs, "df": df}
    found, _shared = so.share_group(["df"], ns, ns)
    assert "logs" in found


def test_the_view_search_reads_lists_of_records_at_once(monkeypatch):
    logs = _sessions()
    df = pd.DataFrame(logs, columns=["user", "actions"])
    ns = {"logs": logs, "df": df}
    asked = [0]
    children_of = so.children_of

    def counted(value):
        asked[0] += 1
        return children_of(value)

    monkeypatch.setattr(so, "children_of", counted)
    assert so.view_base_names([df], ns) == set()
    assert asked[0] < 50, asked[0]


def test_a_view_in_a_cell_beside_lists_of_records_is_still_found():
    base = np.arange(10.0)
    logs = _sessions(200)
    cells = [session[1] for session in logs] + [[("x", 1), base[2:5]]]
    df = pd.DataFrame({"actions": pd.Series(cells, dtype=object)})
    ns = {"logs": logs, "df": df, "base": base}
    assert so.view_base_names([df], ns) == {"base"}
