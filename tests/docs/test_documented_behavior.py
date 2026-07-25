"""Behavioral grounding for the caching guarantees the docs *promise* (P2).

The fence harness runs examples and the claim-checker asserts opt-in hit/miss
counts, but neither exercises the load-bearing *behaviors* the prose sells —
and those are exactly what drifted (the data_sources ``has_changed`` bool bug,
the ``file_depends_on`` mtime-vs-content mix-up). Each test here encodes one
documented promise against a real ``Cash`` instance, so if the behavior changes
the doc claim it grounds is caught.

Call counting uses ``assume_safe=True`` + a dict counter: on a cache *hit* the
body doesn't run, so the counter is the number of real computes. Tests about the
warning contract use pure functions instead, so the only warning that can fire
is the one under test.
"""
from __future__ import annotations

import os
import threading
import time
import warnings

import pytest

from cash import Cash
from cash.data_source import DataSource
from cash.exceptions import CashCacheIneffectiveWarning

# Module globals used by tests where a nested function must read/call a real
# module attribute via ``__globals__`` (a name defined *inside* a test would be
# a closure freevar instead, which the analyzer treats differently).
_TAX_RATE = 0.2

# NOTE: transitive helper-edit invalidation (cash folding a called helper's
# source into the key — the headline difference from joblib in
# docs/migration_guide.md) is intentionally NOT grounded here. It models an
# edit-and-rerun of a helper's *source*, which is a file/notebook operation;
# reproducing it in one in-process test is unreliable (the decorator's
# re-resolution interacts with how the test function nests its defs). That
# invariant is exercised by the core decorator suite under tests/test_core.


def _cash(tmp_path) -> Cash:
    return Cash(cache_dir=str(tmp_path), register_magic=False)


# --------------------------------------------------------------------------- #
# DataSource token contract  (docs/api/data_sources.md,                       #
#                             docs/tutorials/feature-guides/dynamic-dependencies.md)
# --------------------------------------------------------------------------- #

def test_datasource_token_invalidates_when_it_changes(tmp_path):
    """A ``depends_on`` DataSource whose ``has_changed()`` returns a state
    *token* invalidates the cache exactly when that token changes."""
    c = _cash(tmp_path)
    state = {"v": 1}

    class TokenSource(DataSource):
        def get_id(self):
            return "tok"

        def has_changed(self):
            return state["v"]      # a token (int), not a bool

        def update_state(self):
            pass

    n = {"c": 0}

    @c.cache(depends_on=[TokenSource()], assume_safe=True)
    def f():
        n["c"] += 1
        return n["c"]

    f()
    f()
    assert n["c"] == 1, "token unchanged -> cached"
    state["v"] = 2
    f()
    assert n["c"] == 2, "token changed -> recomputed"


def test_datasource_nonbool_token_does_not_warn(tmp_path):
    """A non-bool ``has_changed()`` token is valid and fires no warning."""
    c = _cash(tmp_path)

    class IntSource(DataSource):
        def get_id(self):
            return "int"

        def has_changed(self):
            return 7

        def update_state(self):
            pass

    @c.cache(depends_on=[IntSource()])
    def pure():
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", CashCacheIneffectiveWarning)
        pure()  # must not raise


def test_datasource_bool_has_changed_warns(tmp_path):
    """A ``bool`` ``has_changed()`` can't track changes, so cash warns — this
    is the exact misconfiguration behind the old broken DBTableSource example."""
    c = _cash(tmp_path)

    class BoolSource(DataSource):
        def get_id(self):
            return "bool"

        def has_changed(self):
            return True

        def update_state(self):
            pass

    @c.cache(depends_on=[BoolSource()])
    def pure():
        return 1

    with pytest.warns(CashCacheIneffectiveWarning):
        pure()


# --------------------------------------------------------------------------- #
# File tracking: file_depends_on = mtime, auto = content  (docs/decorator.md)  #
# --------------------------------------------------------------------------- #

def test_file_depends_on_tracks_mtime_not_content(tmp_path):
    """``file_depends_on=`` keys on the file **mtime**, not its content: a
    content edit that leaves the mtime unchanged stays cached; bumping the
    mtime recomputes. (The function must NOT read the file, or auto content
    tracking would mask the mtime-only behavior.)"""
    c = _cash(tmp_path)
    p = tmp_path / "cfg.bin"
    p.write_text("aaaa")
    st = p.stat()
    n = {"c": 0}

    @c.cache(file_depends_on=str(p), assume_safe=True)
    def g():
        n["c"] += 1
        return n["c"]

    g()
    g()
    assert n["c"] == 1

    p.write_text("bbbb")                                   # same size, new content
    os.utime(p, (st.st_atime, st.st_mtime))                # reset mtime to original
    g()
    assert n["c"] == 1, "content changed but mtime reset -> still cached (mtime-only)"

    time.sleep(0.02)
    os.utime(p, None)                                      # bump mtime only
    g()
    assert n["c"] == 2, "mtime changed -> recomputed"


def test_auto_file_tracking_is_content_hash(tmp_path):
    """Automatic file-read tracking fingerprints file **content**: a content
    edit recomputes even if the mtime is reset to its original value."""
    c = _cash(tmp_path)
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n")
    st = p.stat()
    n = {"c": 0}

    @c.cache(assume_safe=True)
    def load():
        n["c"] += 1
        return p.read_text()

    load()
    load()
    assert n["c"] == 1

    p.write_text("a,b\n9,9\n")                             # same size, new content
    os.utime(p, (st.st_atime, st.st_mtime))                # reset mtime
    load()
    assert n["c"] == 2, "content changed (mtime reset) -> recomputed (content-hash)"


# --------------------------------------------------------------------------- #
# Content hashing, module globals, cache_if, helpers, unhashable args         #
# --------------------------------------------------------------------------- #

def test_content_equal_args_share_one_entry(tmp_path):
    """Two content-equal but non-identical args (here dicts differing only in
    insertion order) hit the same cache entry — cash hashes by content, unlike
    ``functools.lru_cache``. The same holds for DataFrames/arrays via hashers."""
    c = _cash(tmp_path)
    n = {"c": 0}

    @c.cache(assume_safe=True)
    def summ(d):
        n["c"] += 1
        return sum(d.values())

    summ({"a": 1, "b": 2})
    summ({"b": 2, "a": 1})
    assert n["c"] == 1


def test_read_module_global_invalidates_on_change(tmp_path):
    """A cached function that *reads* a module global recomputes when that
    global changes (docs/decorator.md 'Module globals a function reads')."""
    global _TAX_RATE
    _TAX_RATE = 0.2
    c = _cash(tmp_path)
    n = {"c": 0}

    @c.cache(assume_safe=True)
    def net(x):
        n["c"] += 1
        return x * (1 - _TAX_RATE)   # reads the module global

    assert net(100) == 80.0
    _TAX_RATE = 0.5
    assert net(100) == 50.0
    assert n["c"] == 2


def test_cache_if_falsy_result_is_not_cached(tmp_path):
    """``cache_if=`` skips caching on a falsy predicate but still returns the
    value (docs/decorator.md 'cache_if — skip caching by result')."""
    c = _cash(tmp_path)
    n = {"c": 0}

    @c.cache(cache_if=lambda r: r is not None, assume_safe=True)
    def maybe(x):
        n["c"] += 1
        return None if x < 0 else x

    assert maybe(-1) is None
    assert maybe(-1) is None
    assert n["c"] == 2, "predicate rejected -> not cached, recomputed"

    n["c"] = 0
    assert maybe(5) == 5
    assert maybe(5) == 5
    assert n["c"] == 1, "predicate accepted -> cached"


def test_unhashable_arg_warns_but_returns_correct_value(tmp_path):
    """An unpicklable argument can't be keyed, so cash warns and skips caching
    — but the call still returns the correct value (docs/decorator.md
    'Unhashable arguments')."""
    c = _cash(tmp_path)

    @c.cache
    def f(lock):
        return "computed"

    with pytest.warns(CashCacheIneffectiveWarning):
        result = f(threading.Lock())
    assert result == "computed"
