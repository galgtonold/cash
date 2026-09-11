"""A file read in a thread pool the cached function starts is one of its inputs.

Round 19 (r19s2): ``ThreadPoolExecutor(4).map(np.load, shards)`` inside a
cached function -- editing a shard HIT the pre-edit value (3 of 3), while the
serial loop beside it invalidated. The tracker is found through a ContextVar,
and a pool's worker threads start with an empty context.
"""
from __future__ import annotations

import contextvars
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return int(fh.read())


def _write(path, value):
    # A different size, so the edit is visible without waiting out mtime ticks.
    path.write_text(str(value), encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.fixture
def shards(tmp_path):
    paths = [tmp_path / f"shard{i}.txt" for i in range(4)]
    for i, p in enumerate(paths):
        p.write_text(str(i + 1), encoding="utf-8")
    return paths


@pytest.mark.parametrize("how", ["map", "submit"])
def test_an_edit_to_a_file_read_in_a_thread_pool_invalidates(tmp_path, shards, how):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    calls = []

    @c.cache
    def total(paths):
        calls.append(1)
        with ThreadPoolExecutor(4) as ex:
            if how == "map":
                values = list(ex.map(_read, paths))
            else:
                values = [f.result() for f in [ex.submit(_read, p) for p in paths]]
        return sum(values)

    names = [str(p) for p in shards]
    assert total(names) == 10
    assert total(names) == 10
    assert len(calls) == 1, "an unedited second call did not hit"
    _write(shards[2], 30)
    assert total(names) == 37, "the pre-edit total was served"
    assert len(calls) == 2


def test_a_pool_used_outside_a_cached_call_keeps_its_own_context():
    """Control: with no tracker active, submit is the original -- a value set
    in the submitting thread does not leak into the worker."""
    var = contextvars.ContextVar("probe", default="empty")
    var.set("submitter")
    with ThreadPoolExecutor(1) as ex:
        assert ex.submit(var.get).result() == "empty"
