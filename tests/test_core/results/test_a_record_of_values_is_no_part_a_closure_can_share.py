"""A tuple of values is no part of a result a closure can share with it.

The closure check (STORE-PARTS-SHARED) made a place of every object a
result holds, one at a time: a frame's two million ``(action, time)``
records cost 2 s a statement. Such a tuple cannot change and holds
nothing that can, so a function holding the very tuple and a restored copy
of it read the same; the records are passed over at C speed. A function
holding a list of the result is still found.
"""

from __future__ import annotations

import datetime

from cash.closure_sharing import closures_sharing

T0 = datetime.datetime(2019, 10, 1)


def _sessions(n: int = 100) -> list:
    return [
        (f"@User{i}", [(f"Action_{j % 11}", T0 + datetime.timedelta(minutes=j)) for j in range(20)]) for i in range(n)
    ]


def test_a_closure_holding_a_record_of_values_shares_nothing_that_can_change():
    logs = _sessions()
    record = logs[3][1][4]
    assert closures_sharing({"out": (logs, lambda: record)}, 10.0).names == []


def test_a_closure_holding_a_list_of_the_result_is_still_found():
    logs = _sessions()
    rows = logs[5][1]
    assert closures_sharing({"out": (logs, lambda: rows)}, 10.0).names == ["out"]
