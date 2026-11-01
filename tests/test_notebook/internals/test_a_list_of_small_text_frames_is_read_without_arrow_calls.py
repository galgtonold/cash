"""The identity check reads a list of small Arrow text frames without an
Arrow call or a column made per frame: each cost a fixed tens of
microseconds, several per column, for thousands of frames."""

from __future__ import annotations

import pytest

from cash.notebook.statement.identity_holders import forget_flat_reads, identity_held_reason

pd = pytest.importorskip("pandas")
pc = pytest.importorskip("pyarrow.compute")


def test_small_text_frames_make_no_arrow_call_or_column(monkeypatch):
    frames = [pd.DataFrame({"user": "u1", "at": ["(1/2/2020)", "(3/4/2020)"], "what": ["a", "b"]}) for _ in range(500)]
    if not hasattr(frames[0]["at"].array, "_pa_array"):
        pytest.skip("this pandas keeps text in Python objects")
    calls = {"arrow": 0, "iloc": 0, "dtypes": 0}
    length = pc.binary_length

    def counted_length(*args, **kwargs):
        calls["arrow"] += 1
        return length(*args, **kwargs)

    monkeypatch.setattr(pc, "binary_length", counted_length)
    for name in ("iloc", "dtypes"):
        prop = getattr(pd.DataFrame, name)

        def counted(self, _prop=prop, _name=name):
            calls[_name] += 1
            return _prop.fget(self)

        monkeypatch.setattr(pd.DataFrame, name, property(counted))
    forget_flat_reads()
    assert identity_held_reason("df_array", frames, {"df_array": frames}) is None
    assert calls == {"arrow": 0, "iloc": 0, "dtypes": 0}

    # A long column still goes to Arrow.
    forget_flat_reads()
    big = pd.DataFrame({"at": ["(1/2/2020)"] * 1000})
    assert identity_held_reason("big", big, {"big": big}) is None
    assert calls["arrow"] > 0
