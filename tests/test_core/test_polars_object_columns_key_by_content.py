"""A polars Object column keys on its values' content, and a panic never escapes.

Polars hashes an Object value with Python's ``hash()``. On an unhashable one
(an ndarray, a dict) that panics in Rust, and the panic, a ``BaseException``,
passed every handler and ended the user's call. On a hashable one it is no
content key: ``hash(-1) == hash(-2)``, and ``1``, ``1.0`` and ``True`` hash
alike, so those shared an entry.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cash import Cash, FileBackend
from cash.exceptions import CashCacheIneffectiveWarning
from cash.object_hashing import compute_hash, compute_hash_full, hash_polars

pl = pytest.importorskip("polars")


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def _embeddings(*rows):
    return pl.DataFrame({"emb": pl.Series([np.array(r) for r in rows], dtype=pl.Object), "x": list(range(len(rows)))})


def test_a_frame_of_unhashable_objects_is_cached(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def n_rows(df):
        return df.height

    assert n_rows(_embeddings([1.0, 2.0])) == 1
    assert n_rows(_embeddings([1.0, 2.0])) == 1
    assert n_rows.cache_info()["hits"] == 1


def test_unhashable_objects_key_on_their_content():
    assert hash_polars(_embeddings([1.0, 2.0])) == hash_polars(_embeddings([1.0, 2.0]))
    assert hash_polars(_embeddings([1.0, 2.0])) != hash_polars(_embeddings([1.0, 3.0]))


@pytest.mark.parametrize("left, right", [(-1, -2), (1, 1.0), (1, True)])
def test_objects_python_hashes_alike_key_apart(tmp_path, left, right):
    c = _cash(tmp_path)

    @c.cache
    def first(s):
        return s.to_list()[0]

    one, two = first(pl.Series("o", [left], dtype=pl.Object)), first(pl.Series("o", [right], dtype=pl.Object))
    assert (one, type(one)) == (left, type(left))
    assert (two, type(two)) == (right, type(right))


def _panic(*_args, **_kwargs):
    raise pl.exceptions.PanicException("should be hashable")


def test_a_panic_while_hashing_runs_the_call_uncached(tmp_path, monkeypatch):
    """Whatever polars panics on, the call returns its result and warns."""
    monkeypatch.setattr(pl.DataFrame, "hash_rows", _panic)
    monkeypatch.setattr(pl.DataFrame, "__getstate__", _panic)
    c = _cash(tmp_path)

    @c.cache
    def width(df):
        return df.width

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert width(pl.DataFrame({"a": [1]})) == 1
    codes = [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert "KEY-UNHASHABLE-ARG" in codes


def test_a_panic_never_escapes_the_notebook_hashes(monkeypatch):
    monkeypatch.setattr(pl.DataFrame, "hash_rows", _panic)
    monkeypatch.setattr(pl.DataFrame, "__getstate__", _panic)
    frame = pl.DataFrame({"a": [1]})
    assert hash_polars(frame) is None
    assert isinstance(compute_hash_full(frame), str)
    assert isinstance(compute_hash(frame), str)
