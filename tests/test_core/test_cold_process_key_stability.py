"""Regressions for the cold-process cache-key bug (#7) and the decorator-line
purity false-positive (#9).

#7: the cache key folds in transitive purity-report helper hashes, which used to
be populated lazily on each dependency's first call. So the key deepened after
the chain warmed and a fresh process missed the first call to every cached
function. The fix eagerly analyzes the whole dependency closure before the first
key is computed, making the key identical on the lookup and store paths and
across processes.

#9: ``inspect.getsource`` includes the ``@c.cache(...)`` decorator line; the
analyzer used to treat that call as a body statement and walk into cash's own
decorator machinery, flagging its internal mutations as the user's.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from cash import Cash, CashImpurityWarning, FileBackend


def _build_chain(c: Cash):
    """A 3-level depends_on chain bound to *c*. Returns the top function."""

    def base_helper(x):
        return x + 1

    @c.cache
    def load(x):
        return base_helper(x) * 2

    @c.cache(depends_on=[load])
    def mid(x):
        return load(x) + 10

    @c.cache(depends_on=[mid])
    def top(x):
        return mid(x) + 100

    return top


# --- #7 -------------------------------------------------------------------

def test_lookup_key_equals_store_key_first_call(tmp_path):
    """explain() (pre-execution) and the actual stored key must match on the
    very first call - no key drift after the chain warms."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "c")))
    top = _build_chain(c)
    pre = top.explain(5).cache_key
    top(5)
    post = top.explain(5).cache_key
    assert pre == post
    assert top.explain(5).reason == "hit"


def test_fresh_instance_hits_on_first_call(tmp_path):
    """A second Cash instance on the same persistent backend restores on the
    FIRST call to each function - zero warm-up recomputes."""
    cache_dir = str(tmp_path / "c")

    c1 = Cash(backend=FileBackend(cache_dir=cache_dir))
    top1 = _build_chain(c1)
    for s in (1, 2, 3):
        top1(s)

    c2 = Cash(backend=FileBackend(cache_dir=cache_dir))
    top2 = _build_chain(c2)
    # Every first call in the fresh instance hits.
    for s in (1, 2, 3):
        assert top2.explain(s).reason == "hit", s
    for s in (1, 2, 3):
        top2(s)
    info = top2.cache_info()
    assert info["misses"] == 0, info
    assert info["hits"] == 3, info


def test_cross_process_consumer_zero_warmup_misses(tmp_path):
    """End-to-end: a fresh *process* that only reads results another process
    computed gets 100% hits (no per-function warm-up recompute)."""
    mod = tmp_path / "qf_chain.py"
    mod.write_text(textwrap.dedent(f"""
        import cash
        from cash import FileBackend
        c = cash.Cash(backend=FileBackend(cache_dir={str(tmp_path / 'c')!r}))

        def base_helper(x): return x + 1

        @c.cache
        def load(x): return base_helper(x) * 2

        @c.cache(depends_on=[load])
        def mid(x): return load(x) + 10

        @c.cache(depends_on=[mid])
        def top(x): return mid(x) + 100
    """))
    driver = textwrap.dedent("""
        import sys, json, qf_chain
        phase = sys.argv[1]
        syms = [1, 2, 3]
        for s in syms:
            qf_chain.top(s)
        if phase == "read":
            info = qf_chain.top.cache_info()
            print("RESULT " + json.dumps({"hits": info["hits"], "misses": info["misses"]}))
    """)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))

    def run(phase):
        p = subprocess.run([sys.executable, "-c", driver, phase],
                           capture_output=True, text=True, env=env)
        return p

    run("write")
    out = run("read")
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, f"no result:\n{out.stdout}\n{out.stderr}"
    res = json.loads(line[0][len("RESULT "):])
    assert res == {"hits": 3, "misses": 0}, res


# --- #9 -------------------------------------------------------------------

def test_decorator_line_does_not_leak_cash_internals(tmp_path):
    """A cached function whose decorator is ``@c.cache(depends_on=[...])`` must
    not draw impurity warnings about cash's own machinery."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "c")))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        top = _build_chain(c)
        top(5)
        impurity = [x for x in w if issubclass(x.category, CashImpurityWarning)]
    assert not impurity, [str(x.message) for x in impurity]


def test_calling_factory_in_decorator_is_not_analyzed(tmp_path):
    """`@get_c().cache` - a call in the decorator expression - must not be
    walked into (the quantforge get_cash() case)."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "c")))

    def get_c():
        # An impure-looking factory: mutates a module global, discards a call.
        globals().setdefault("_HITS", []).append(1)
        return c

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @get_c().cache
        def f(x):
            return x * 2

        f(3)
        impurity = [x for x in w if issubclass(x.category, CashImpurityWarning)]
    assert not impurity, [str(x.message) for x in impurity]
