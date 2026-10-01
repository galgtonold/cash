"""What a callable instance captured by a closure or a decorator holds reaches the key.

``make(Scorer(10))`` returns a closure over the instance; a decorator can
build one too (``c = Scale(10)`` in its body). The helper walk treated the
captured instance as a function -- code only -- so ``Scorer(11)`` kept the key
of ``Scorer(10)`` and served its result. A plain (non-callable) instance
captured the same way was keyed all along, as was a bare global callable
instance. A torch module or sklearn model built in a factory closure has
this shape.

Fresh process per run.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

LIB = textwrap.dedent("""
    import functools

    class CC:
        def __init__(self, k):
            self.k = k

        def __call__(self, x):
            return x * self.k

    def make(cfg):
        def score(x):
            return cfg(x)
        return score

    SCORE = make(CC({k}))

    def base(x):
        return x

    def deco(fn):
        c = CC({k})
        @functools.wraps(fn)
        def w(x):
            return fn(x) * c(1)
        return w

    DECO = deco(base)
""")

MAIN = textwrap.dedent("""
    import json, sys, time, warnings
    warnings.simplefilter("ignore")
    import cash, lib

    @cash.cache
    def closure(x):
        print("RAN", file=sys.stderr)  # @cash:assume-safe
        return lib.SCORE(x)

    @cash.cache
    def decorated(x):
        print("RAN", file=sys.stderr)  # @cash:assume-safe
        return lib.DECO(x)

    print(json.dumps([closure(2), decorated(2)]))
""")


def _run(tmp_path, k):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "lib.py").write_text(LIB.format(k=k), encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    out = run_python("main.py", cwd=proj, cache_dir=tmp_path / "cache")
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr.count("RAN")


def test_a_new_constructor_argument_recomputes(tmp_path):
    """THE BUG: CC(10) -> CC(11) served 20 where an uncached run gives 22."""
    assert _run(tmp_path, 10) == ([20, 20], 2)
    assert _run(tmp_path, 11) == ([22, 22], 2)


def test_the_same_instance_still_hits(tmp_path):
    """The control: keying the captured instance costs no hit."""
    _run(tmp_path, 10)
    assert _run(tmp_path, 10) == ([20, 20], 0)
