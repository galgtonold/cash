"""A text column of thousands of small Arrow chunks is hashed in a few
calls, not several NumPy calls per chunk: ``pd.concat`` of 20,000 small
frames keeps a chunk per frame."""

from __future__ import annotations

import hashlib

import pytest

from cash import content_hashers

pa = pytest.importorskip("pyarrow")


def test_small_text_chunks_are_read_in_runs(monkeypatch):
    import numpy as np

    calls = [0]
    real = np.diff

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "diff", counted)
    column = pa.chunked_array([pa.array(["ab", "c", None])] * 5_000)
    content_hashers.fold_arrow_column(hashlib.sha256(), column)
    assert 0 < calls[0] <= 5
