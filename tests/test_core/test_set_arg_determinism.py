"""Set/frozenset arguments must hash deterministically.

A set pickles in iteration order, which for str/bytes elements depends on
PYTHONHASHSEED - so the same set argument produced a different cache key in
every process, silently breaking cross-process restore (and in-process matching
of equal sets built in different orders). Cash canonicalises set/dict ordering
before hashing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from cash import Cash, FileBackend


def _key_in_subprocess(hashseed: str) -> str:
    code = (
        "import tempfile\n"
        "from cash import Cash, FileBackend\n"
        "c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))\n"
        "@c.cache\n"
        "def f(tags):\n    return 1\n"
        "s = {'alpha','beta','gamma','delta','epsilon','zeta','eta','theta'}\n"
        "print(f.explain(s).cache_key)\n"
    )
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    return out.stdout.strip().splitlines()[-1]


def test_set_arg_key_is_stable_across_hash_seeds():
    # Different PYTHONHASHSEED == different process-local str hashing. The key
    # for the same set argument must be identical regardless.
    k0 = _key_in_subprocess("0")
    k1 = _key_in_subprocess("1")
    k2 = _key_in_subprocess("12345")
    assert k0 == k1 == k2, (k0, k1, k2)


def test_equal_sets_built_in_different_orders_hit():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))
    calls = {"n": 0}

    @c.cache
    def f(tags):
        calls["n"] += 1
        return sorted(tags)

    f({"x", "y", "z"})
    f({"z", "y", "x"})              # equal set, different construction order
    assert calls["n"] == 1


def test_different_sets_do_not_collide():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    @c.cache
    def f(tags):
        return 1

    assert f.explain({1, 2, 3}).cache_key != f.explain({1, 2, 4}).cache_key


def test_nested_set_in_container_is_canonical():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))
    calls = {"n": 0}

    @c.cache
    def g(data):
        calls["n"] += 1
        return len(data)

    g([1, {"a", "b"}, 2])
    g([1, {"b", "a"}, 2])          # nested set, different order
    g({"tags": {"x", "y"}})
    g({"tags": {"y", "x"}})        # set as dict value, different order
    assert calls["n"] == 2          # two distinct shapes, each computed once


def test_set_inside_object_is_canonical_across_hash_seeds():
    """A set nested inside a dataclass/object (not a plain container) must also
    hash deterministically across processes."""
    code = (
        "import tempfile\n"
        "from dataclasses import dataclass\n"
        "from cash import Cash, FileBackend\n"
        "@dataclass(frozen=True)\n"
        "class Cfg:\n"
        "    name: str\n"
        "    tags: frozenset\n"
        "c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))\n"
        "@c.cache\n"
        "def f(cfg):\n    return 1\n"
        "cfg = Cfg('a', frozenset({'x','y','z','p','q','r','s','t'}))\n"
        "print(f.explain(cfg).cache_key)\n"
    )

    def run(seed):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env)
        return out.stdout.strip().splitlines()[-1]

    assert run("0") == run("1") == run("314")


def test_set_in_object_order_independent_in_process():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    class Holder:
        def __init__(self, tags):
            self.tags = tags

    @c.cache
    def f(h):
        return 1

    k1 = f.explain(Holder({"a", "b", "c"})).cache_key
    k2 = f.explain(Holder({"c", "b", "a"})).cache_key
    k3 = f.explain(Holder({"a", "b", "d"})).cache_key
    assert k1 == k2          # equal set, different order -> same key
    assert k1 != k3          # different set -> different key


def test_frozenset_arg_stable_and_distinct_from_set():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    @c.cache
    def f(x):
        return 1

    # frozenset is order-independent...
    assert f.explain(frozenset({1, 2, 3})).cache_key == \
           f.explain(frozenset({3, 2, 1})).cache_key
    # ...and not confused with the equivalent set (different type identity).
    assert f.explain(frozenset({1, 2, 3})).cache_key != \
           f.explain({1, 2, 3}).cache_key
