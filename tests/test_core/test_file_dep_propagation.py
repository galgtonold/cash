"""Regression for finding #11: file-dependency invalidation must propagate
through a ``depends_on`` chain.

A cached function that reads a file only via a nested cached call used to keep
returning a stale result after the file changed - the nested read was attributed
only to the innermost function. Now reads propagate to the enclosing cached
function (on a cold miss via the tracker parent-chain, and on a hit by replaying
the dependency's recorded deps), so the outer entry invalidates too - regardless
of the order the functions were first computed in.
"""
from __future__ import annotations

import time

import pytest

from cash import Cash, FileBackend


def _build(c: Cash, data_path):
    @c.cache
    def load():
        with open(data_path) as f:
            return f.read().strip()

    @c.cache(depends_on=[load])
    def upper():
        return load().upper()

    return load, upper


@pytest.mark.parametrize("compute_inner_first", [False, True])
def test_file_change_propagates_through_depends_on(tmp_path, compute_inner_first):
    data = tmp_path / "data.txt"
    data.write_text("hello")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))
    load, upper = _build(c, str(data))

    if compute_inner_first:
        load()              # inner cached first -> inner is a HIT inside upper
    assert upper() == "HELLO"

    time.sleep(0.02)
    data.write_text("world-changed")

    # The outer function must notice the change, not return the stale value.
    assert upper.explain().reason != "hit"
    assert upper() == "WORLD-CHANGED"


def test_outer_records_nested_file_dep(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))
    load, upper = _build(c, str(data))

    upper()   # cold: nested load() reads the file
    raw_meta, _ = c.backend.get(upper.explain().cache_key)
    recorded = (raw_meta or {}).get("auto_file_deps") or {}
    assert any(p.endswith("data.txt") for p in recorded), recorded
