"""The identity check reads a list of sessions, each a ``(user, [(action,
time), ...])`` record, a level at a time.

Each session's list was read on its own: 87,000 reads of 20 records, 3 s
after every statement that bound or changed the list. Read a level at a
time, the same numbers are found: an ``id()`` of a notebook's object kept
in a record, as a number or spelled in hex, still sends the statement to
run again.
"""

from __future__ import annotations

import datetime

import pytest

from cash.notebook.statement import identity_holders as ih

T0 = datetime.datetime(2019, 10, 1)


class Thing:
    pass


def _sessions(n: int = 3000) -> list:
    return [
        (f"@User{i % 50}", [(f"Action_{j % 11}", T0 + datetime.timedelta(minutes=j)) for j in range(20)])
        for i in range(n)
    ]


def test_sessions_are_read_in_a_few_reads(monkeypatch):
    logs = _sessions()
    reads = [0]
    read_flat = ih._Walk._read_flat

    def counted(self, items):
        reads[0] += 1
        return read_flat(self, items)

    monkeypatch.setattr(ih._Walk, "_read_flat", counted)
    assert ih.identity_held_reason("logs", logs, {"logs": logs}) is None
    assert reads[0] < 10, reads[0]


@pytest.mark.parametrize("spell", [lambda n: n, hex], ids=["a number", "hex text"])
def test_an_id_kept_in_a_session_is_still_found(spell):
    thing = Thing()
    logs = _sessions(500)
    logs[7][1][3] = ("Action_1", spell(id(thing)))
    found = ih.identity_held_reason("logs", logs, {"logs": logs, "thing": thing})
    assert found is not None and found.kind == "an id()"
