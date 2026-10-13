"""A polars call result is digested from its Arrow buffers, not polars' pickle.

A statement refers to a call's entry when the result it holds is still what
the call returned, which a digest of the result's pickle proves. Polars' own
pickle first copies the whole frame into one IPC blob: 2.2-4 s for a 4.2M-row
log frame, against 0.3 s for its buffers, which is how the store writes it.
"""

import pytest

from cash.notebook.call_refs import digest_and_size
from cash.sizing import pickled_size_estimate

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")


def _frame(last: float = 9.0):
    return pl.DataFrame({"x": np.arange(1000, dtype=float), "s": ["a", "bb"] * 500}).with_columns(
        pl.when(pl.int_range(pl.len()) == 999).then(last).otherwise(pl.col("x")).alias("x")
    )


def test_polars_own_pickle_is_not_used(monkeypatch):
    def refuse(self):
        raise AssertionError("polars' own pickle copies the whole frame")

    monkeypatch.setattr(pl.DataFrame, "__getstate__", refuse)
    assert digest_and_size(_frame()) is not None


def test_the_digest_follows_the_content():
    assert digest_and_size(_frame())[0] == digest_and_size(_frame())[0]
    assert digest_and_size(_frame())[0] != digest_and_size(_frame(last=8.0))[0]


def test_a_polars_frame_has_a_size_estimate():
    frame = _frame()
    assert pickled_size_estimate(frame) == frame.estimated_size() > 0
    assert pickled_size_estimate(frame["x"]) == frame["x"].estimated_size()
