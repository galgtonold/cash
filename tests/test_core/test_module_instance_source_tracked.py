"""Editing a method on the CLASS behind a pre-built module-level object invalidates.

A cached function often consumes a pre-built module-level object as data: a
transformer constructed once at import and dropped into a pipeline
(``pre = MyTransformer(); ... steps = [("pre", pre)]; run(steps)``). That object
is value-hashed from its ``__dict__``, which carries no method source, so editing
``MyTransformer.transform`` used to leave the key unchanged and serve a stale
result -- a wrong-value bug replaying a real repo's git history surfaced (a
house-prices pipeline whose median-imputation transformer's method was edited;
the method is called deep inside sklearn, never syntactically in the cached
function, so the only signal is the object's class source).

cash now folds the source of such an object's class -- and, bounded, of the
user-class instances it holds. The class lives in a written module so its source
genuinely changes between runs; the cache instance is shared across re-imports
(via ``builtins``) so a stale hit is observable.

Boundary (deliberate, matches the accumulator-drift protection): this fires only
for objects used as pure DATA. An object you METHOD-CALL directly
(``obj.method()``) or pass as a BARE argument (``fn(obj)``) is excluded from
value-folding -- but a directly-called method's own source edit is already
caught by the purity/helper channel; only a method reached solely through such
an excluded object is missed.

Without the fix these tests HIT and return the OLD value -- run them through
``scripts/fails_first.py`` to confirm they can fail.
"""
from __future__ import annotations

import builtins
import importlib.util
import sys

import pytest

from cash import Cash
from cash.backends import FileBackend

# OBJ (a user-class instance) is an element of a list literal assigned to a
# local, then handed to a helper that calls its method -- pure-data use, so the
# object flows through the global-data fold where the class-source channel runs.
SINGLE = '''\
import builtins
_inst = builtins._CASH_TEST_INST

class Thing:
    def val(self):
        return {body}

def run(steps):
    return sum(o.val() for _, o in steps)

OBJ = Thing()

@_inst.cache
def f(x):
    builtins._CASH_TEST_CALLS.append(1)
    steps = [("a", OBJ)]
    return x + run(steps)
'''

# The edited method is on an object the read global merely HOLDS (Container.inner
# is an Inner) -- two hops from the read global, mirroring the real pipeline.
NESTED = '''\
import builtins
_inst = builtins._CASH_TEST_INST

class Inner:
    def val(self):
        return {body}

class Container:
    def __init__(self):
        self.inner = Inner()

def run(steps):
    return sum(o.inner.val() for _, o in steps)

OBJ = Container()

@_inst.cache
def f(x):
    builtins._CASH_TEST_CALLS.append(1)
    steps = [("a", OBJ)]
    return x + run(steps)
'''


_FILE_N = [0]


def _reimport(tmp_path, src):
    # Constant MODULE NAME so the cached function's identity (its
    # module-qualified name, part of the key) is stable across versions -- the
    # canary relies on identical source keying the same. But a UNIQUE FILE PATH
    # per call, so ``inspect.getsource`` (which reads through linecache, keyed by
    # filename) always hits a fresh path and re-reads the rewritten class
    # source -- no dependence on filesystem mtime resolution, and no collision
    # with another test file's reused module/path under xdist worksteal.
    _FILE_N[0] += 1
    path = tmp_path / f"cmod_{_FILE_N[0]}.py"
    path.write_text(src, encoding="utf-8")
    sys.modules.pop("cash_modinst_mod", None)
    spec = importlib.util.spec_from_file_location("cash_modinst_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cash_modinst_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path):
    inst = Cash(backend=FileBackend(str(tmp_path / "cache")), register_magic=False)
    builtins._CASH_TEST_INST = inst
    builtins._CASH_TEST_CALLS = []
    sys.path.insert(0, str(tmp_path))
    try:
        yield tmp_path
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("cash_modinst_mod", None)
        del builtins._CASH_TEST_INST
        del builtins._CASH_TEST_CALLS


def test_data_passed_object_class_method_edit_invalidates(env):
    m1 = _reimport(env, SINGLE.format(body="1"))
    assert m1.f(10) == 11
    # Edit the method body of the class behind the data-passed instance OBJ.
    m2 = _reimport(env, SINGLE.format(body="5"))
    assert m2.f(10) == 15  # recomputed -- must NOT serve the stale 11


def test_nested_held_instance_method_edit_invalidates(env):
    m1 = _reimport(env, NESTED.format(body="1"))
    assert m1.f(10) == 11
    m2 = _reimport(env, NESTED.format(body="100"))
    assert m2.f(10) == 110  # recomputed through the held Inner -- not stale 11


def test_identical_source_still_hits(env):
    """Canary: folding class source must not over-invalidate. Reloading identical
    source (a fresh class object, same text) still serves the cached value."""
    m1 = _reimport(env, SINGLE.format(body="1"))
    assert m1.f(10) == 11
    n = len(builtins._CASH_TEST_CALLS)
    m2 = _reimport(env, SINGLE.format(body="1"))
    assert m2.f(10) == 11
    assert len(builtins._CASH_TEST_CALLS) == n  # no recompute: HIT
