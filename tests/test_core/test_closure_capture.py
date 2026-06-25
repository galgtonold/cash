"""Closures that differ only in captured free variables must not collide.

Two closures from the same factory share source AND qualname
(``factory.<locals>.f``) but capture different values. The cache key now folds
in IMMUTABLE, read-only captures so they stay distinct - while NOT folding in
mutable captures or reassigned (`nonlocal`) counters, which drift between calls.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from cash import Cash, FileBackend, InMemoryBackend


def _make(c, factor):
    @c.cache
    def f(x):
        return x * factor          # captures immutable `factor`, read-only
    return f


def test_closures_with_different_immutable_captures_do_not_collide():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))
    f2 = _make(c, 2)
    f5 = _make(c, 5)
    assert f2(10) == 20
    assert f5(10) == 50            # must NOT return f2's cached 20


def test_str_capture_distinguishes_closures():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    def tagger(tag):
        @c.cache
        def g(x):
            return f"{tag}:{x}"
        return g

    assert tagger("a")(1) == "a:1"
    assert tagger("b")(1) == "b:1"


def test_nonlocal_counter_still_hits():
    """A reassigned (`nonlocal`) counter must be excluded from the key so the
    function still caches across calls."""
    c = Cash(backend=InMemoryBackend())

    def build():
        n = 0

        @c.cache
        def f(a, b):
            nonlocal n
            n += 1                 # STORE_DEREF -> excluded from key
            return a + b

        f(1, 2)
        f(1, 2)
        return n

    assert build() == 1


def test_mutable_dict_capture_still_hits():
    c = Cash(backend=InMemoryBackend())
    calls = {"n": 0}

    @c.cache
    def f(x):
        calls["n"] += 1            # mutates a captured dict -> excluded from key
        return x * 2

    f(3)
    f(3)
    assert calls["n"] == 1


def test_closure_key_stable_across_processes():
    code = (
        "import tempfile\n"
        "from cash import Cash, FileBackend\n"
        "c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))\n"
        "def make(factor):\n"
        "    @c.cache\n"
        "    def f(x):\n        return x * factor\n"
        "    return f\n"
        "print(make(7).explain(10).cache_key)\n"
    )

    def run(seed):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env)
        return out.stdout.strip().splitlines()[-1]

    assert run("0") == run("1") == run("99")
