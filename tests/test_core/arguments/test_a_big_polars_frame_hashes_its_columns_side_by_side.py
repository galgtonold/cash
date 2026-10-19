"""A big polars frame's columns are hashed side by side, to the same digest.

Each column folds a short header and its parts' digests into the frame's
hash, so a frame of big chunks has each column read on its own thread and
the records handed over in column order. p08's clean() result hashed in
0.25 s instead of 0.48 s; the digest, and so every key, is the one the
columns gave one after another.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from cash import content_hashers

WORDS = ["", "a", "é", "x" * 12, "y" * 13, "z" * 100, None, "ünïcödé" * 5]


def _frame(n):
    rng = np.random.default_rng(0)
    words = [WORDS[i] for i in rng.integers(0, len(WORDS), n)]
    return pl.DataFrame(
        {
            "s": words,
            "i": rng.integers(0, 9, n),
            "f": rng.random(n),
            "b": rng.random(n) > 0.5,
            "bin": [w.encode() if w is not None else None for w in words],
            "l": [[w] for w in words],
            "c": pl.Series(words).cast(pl.Categorical),
        }
    )


def _one_after_another(frame, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(content_hashers, "PARALLEL_COLUMNS_MIN_ROWS", 10**12)
        return content_hashers.hash_polars(frame)


@pytest.mark.parametrize("n", [5_000, 60_000])
def test_side_by_side_gives_the_digest_of_one_after_another(n, monkeypatch):
    frame = _frame(n)
    expected = _one_after_another(frame, monkeypatch)
    monkeypatch.setattr(content_hashers, "PARALLEL_COLUMNS_MIN_ROWS", 1_000)
    monkeypatch.setattr(content_hashers, "hashing_threads", lambda: 4)
    threads = []
    real = content_hashers._fold_polars_columns

    def counting(h, columns, rows_per_chunk):
        threads.append(rows_per_chunk >= content_hashers.PARALLEL_COLUMNS_MIN_ROWS)
        return real(h, columns, rows_per_chunk)

    monkeypatch.setattr(content_hashers, "_fold_polars_columns", counting)
    assert content_hashers.hash_polars(frame) == expected
    assert content_hashers.hash_polars(frame.slice(7, n - 20)) == _one_after_another(
        frame.slice(7, n - 20), monkeypatch
    )
    assert threads[0] is True


def test_a_frame_of_small_chunks_is_hashed_on_one_thread(monkeypatch):
    frame = pl.concat([_frame(3_000)] * 4, rechunk=False)
    monkeypatch.setattr(content_hashers, "PARALLEL_COLUMNS_MIN_ROWS", 5_000)
    assert content_hashers._rows_per_chunk(frame) == 3_000
    started = []
    monkeypatch.setattr(content_hashers.threading, "Thread", lambda *a, **k: started.append(1))
    assert content_hashers.hash_polars(frame) == _one_after_another(frame, monkeypatch)
    assert not started


def test_a_column_that_fails_raises_on_the_caller(monkeypatch):
    frame = _frame(5_000)
    monkeypatch.setattr(content_hashers, "PARALLEL_COLUMNS_MIN_ROWS", 1_000)
    monkeypatch.setattr(content_hashers, "hashing_threads", lambda: 4)

    def broken(self, h):
        raise RuntimeError("boom")

    monkeypatch.setattr(content_hashers._ArrowColumn, "fold", broken)
    with pytest.raises(RuntimeError, match="boom"):
        content_hashers._fold_polars_columns(
            content_hashers.hashlib.sha256(), content_hashers._polars_arrow_columns(frame, pl), 5_000
        )
