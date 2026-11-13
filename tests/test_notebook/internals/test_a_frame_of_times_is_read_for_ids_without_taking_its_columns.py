"""The identity check takes out of a frame only the columns whose values
could keep an ``id()``: a datetime or float column was taken as a frame of
its own, then passed over, which for a list of thousands of small frames
took most of the check's time."""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import forget_flat_reads, identity_held_reason

pd = pytest.importorskip("pandas")


def test_a_time_column_is_not_taken_out(monkeypatch):
    frames = [
        pd.DataFrame({"at": pd.to_datetime(["2020-01-01", "2020-01-02"]), "x": [1.5, 2.5], "n": [1, 2]})
        for _ in range(200)
    ]
    taken: list = []
    read = identity_holders._pandas_column

    def counted(column, dtype):
        taken.append(str(dtype))
        return read(column, dtype)

    monkeypatch.setattr(identity_holders, "_pandas_column", counted)
    forget_flat_reads()
    assert identity_held_reason("frames", frames, {"frames": frames}) is None
    # Of the columns, the 8-byte integers alone, and those straight from
    # the frame's own arrays: none is made a column of its own.
    assert "int64" not in taken
    assert not {"datetime64[ns]", "float64"} & set(taken)


def test_an_id_in_an_integer_column_beside_times_is_still_found():
    target = object()
    frames = [pd.DataFrame({"at": pd.to_datetime(["2020-01-01"]), "n": [7]}) for _ in range(20)]
    frames[9]["n"] = [id(target)]
    ns = {"frames": frames, "target": target}
    forget_flat_reads()
    assert identity_held_reason("frames", frames, ns) is not None
