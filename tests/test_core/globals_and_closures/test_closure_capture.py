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
import types

from cash import Cash, FileBackend, InMemoryBackend


def _make(c, factor):
    @c.cache
    def f(x):
        return x * factor  # captures immutable `factor`, read-only

    return f


def test_closures_with_different_immutable_captures_do_not_collide():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))
    f2 = _make(c, 2)
    f5 = _make(c, 5)
    assert f2(10) == 20
    assert f5(10) == 50  # must NOT return f2's cached 20


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
            n += 1  # STORE_DEREF -> excluded from key
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
        calls["n"] += 1  # mutates a captured dict -> excluded from key
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
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        return out.stdout.strip().splitlines()[-1]

    assert run("0") == run("1") == run("99")


def _module(name, step):
    """A module whose ``helper`` adds *step*, built without a file."""
    mod = types.ModuleType(name)
    exec(f"def helper(x):\n    return x + {step}\n", mod.__dict__)
    return mod


def _make_reader(cap):
    def total(x):
        return cap.helper(x)

    return total


def test_closures_capturing_different_modules_do_not_collide():
    """One factory, two captured modules: same text, same qualname, and
    two different helpers behind ``cap.helper``."""
    c = Cash(backend=InMemoryBackend())
    total_a = c.cache(_make_reader(_module("_capture_mod_a", 1)))
    total_b = c.cache(_make_reader(_module("_capture_mod_b", 2)))
    assert total_a.explain(1).cache_key != total_b.explain(1).cache_key
    assert total_a(1) == 2
    assert total_b(1) == 3


def test_closure_key_follows_the_code_it_reads_from_a_captured_module():
    c = Cash(backend=InMemoryBackend())
    mod = _module("_capture_mod_edit", 1)
    total = c.cache(_make_reader(mod))
    before = total.explain(1).cache_key

    mod.unrelated = lambda x: x  # not read through the capture
    assert total.explain(1).cache_key == before

    exec("def helper(x):\n    return x + 10\n", mod.__dict__)
    assert total.explain(1).cache_key != before
    assert total(1) == 11


def test_module_capture_key_stable_across_processes(tmp_path):
    """A captured module is keyed by what it is, not where it lives in memory."""
    (tmp_path / "_capture_mod_disk_a.py").write_text("def helper(x):\n    return x + 1\n", encoding="utf-8")
    (tmp_path / "_capture_mod_disk_b.py").write_text("def helper(x):\n    return x + 2\n", encoding="utf-8")
    code = (
        "import tempfile\n"
        "import _capture_mod_disk_a, _capture_mod_disk_b\n"
        "from cash import Cash, FileBackend\n"
        "c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))\n"
        "def make(cap):\n"
        "    def total(x):\n        return cap.helper(x)\n"
        "    return total\n"
        "print(c.cache(make(_capture_mod_disk_a)).explain(1).cache_key)\n"
        "print(c.cache(make(_capture_mod_disk_b)).explain(1).cache_key)\n"
    )

    def run(seed):
        path = os.pathsep.join([str(tmp_path), *sys.path])
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=path, PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path)
        return tuple(out.stdout.strip().splitlines()[-2:])

    first = run("0")
    assert len(set(first)) == 2  # the two modules key apart
    assert run("1") == run("99") == first
