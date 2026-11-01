"""A frame whose object column holds lists of ``(action, time)`` records,
shared with the sessions it was built from, has those records read once
between the checks after its statement.

The share walk, the holder search, the view search and the alias check
each read every record of the column again to tell that the lists hold
nothing but values: for 87,000 sessions of 20 records, a second or more
apiece after each statement that rebinds the frame (student_8). The
answer is kept for the look those checks share, for the very lists it was
found for.
"""

from __future__ import annotations

import operator

import pytest

from cash import _plain_data
from cash.notebook import shared_objects
from cash.notebook.shared_objects import referring_names, value_record_lists, view_base_names
from cash.notebook.statement.derivation_edges import is_uncacheable_alias
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

SESSIONS = 300
ACTIONS = 4

BUILD = (
    "import datetime\n"
    "t0 = datetime.datetime(2020, 1, 1)\n"
    f"logs = [(f'u{{s % 7}}', [(f'a{{a}}', t0 + datetime.timedelta(seconds=s + a)) for a in range({ACTIONS})])"
    f" for s in range({SESSIONS})]\n"
)
FRAME = "import pandas as pd\ndf = pd.DataFrame(logs, columns=['user', 'actions'])\n"


def _count_record_reads(monkeypatch) -> list[int]:
    reads: list[int] = []
    real = shared_objects._values_in_tuples

    def counted(items, exact):
        if (
            type(items) is list
            and len(items) == SESSIONS * ACTIONS
            and all(map(operator.is_, map(type, items), [tuple] * len(items)))
        ):
            reads.append(1)
        return real(items, exact)

    monkeypatch.setattr(shared_objects, "_values_in_tuples", counted)
    return reads


def test_the_records_of_the_column_are_read_once_after_the_statement(cash_magics, mock_shell, monkeypatch):
    run_cash_cell(cash_magics, BUILD)
    reads = _count_record_reads(monkeypatch)
    run_cash_cell(cash_magics, FRAME)
    assert mock_shell.user_ns["df"]["actions"].iloc[0] is mock_shell.user_ns["logs"][0][1]
    assert len(reads) == 1


def _sessions() -> list:
    return [(f"u{s % 7}", [(f"a{a}", s + a) for a in range(ACTIONS)]) for s in range(SESSIONS)]


def test_each_check_in_a_look_reads_the_records_once(monkeypatch):
    logs = _sessions()
    df = pd.DataFrame(logs, columns=["user", "actions"])
    ns = {"logs": logs, "df": df}
    reads = _count_record_reads(monkeypatch)
    with _plain_data.one_look({"df": df}):
        assert view_base_names([df], ns) == set()
        assert referring_names([df], ns, 30.0, skip_name=lambda name: name == "df")[0] == {"logs"}
        assert not is_uncacheable_alias(df, ns)
    assert len(reads) == 1


def test_a_view_in_one_session_is_still_found_in_the_look():
    base = np.arange(10)
    logs = _sessions()
    logs[150][1].append(base[2:5])
    df = pd.DataFrame(logs, columns=["user", "actions"])
    ns = {"logs": logs, "df": df, "base": base}
    with _plain_data.one_look({"df": df}):
        assert not value_record_lists(list(df["actions"]))
        assert view_base_names([df], ns) >= {"base"}
        assert is_uncacheable_alias(df, ns)


def test_other_lists_with_the_same_ends_are_read_for_themselves():
    """The answer kept for a column is not given for other lists that only
    begin and end with the same two: the middle one here holds a view."""
    base = np.arange(10)
    logs = _sessions()
    df = pd.DataFrame(logs, columns=["user", "actions"])
    other = list(df["actions"])
    other[150] = [("a", 1), base[2:5]]
    with _plain_data.one_look({"df": df}):
        assert value_record_lists(list(df["actions"]))
        view_base_names([df], {"df": df})
        assert value_record_lists(list(df["actions"]))
        assert not value_record_lists(other)
