"""Globals a HELPER reads must invalidate the function that calls it.

The cached function's own read-globals were folded; a helper's were not.
So this returned a wrong answer, silently::

    THRESHOLD = 5
    def helper(): return THRESHOLD

    @cash.cache
    def via_helper(n): return helper()   # THRESHOLD=6 -> HIT -> 5

The counterweight is the drift guard. A global that changes *because* the
call ran (an accumulator) must NOT be folded: keying an entry on the
function's own output means it can never hit again. That trap is what
`test_accumulator_does_not_miss_forever` pins, and it is the reason this
cannot simply reuse the existing fold as-is -- drift is learned against the
CACHED function's code object, while the fold reads it under the helper's.
"""

import importlib
import importlib.util
import linecache
import shutil
import sys

import pytest

from cash import Cash


@pytest.fixture()
def env(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.syspath_prepend(str(work))
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    yield cash, work
    sys.modules.pop("usermod", None)


def load(work, cash, source):
    (work / "usermod.py").write_text(source)
    shutil.rmtree(work / "__pycache__", ignore_errors=True)
    sys.modules.pop("usermod", None)
    importlib.invalidate_caches()
    linecache.clearcache()
    spec = importlib.util.spec_from_file_location("usermod", str(work / "usermod.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["usermod"] = mod
    mod.__dict__["cash_instance"] = cash
    spec.loader.exec_module(mod)
    return mod


def call(work, cash, source, fn_name="via_helper"):
    """Return (was_hit, value)."""
    mod = load(work, cash, source)
    fn = getattr(mod, fn_name)
    before = fn.cache_info()["hits"]
    value = fn(1)
    return fn.cache_info()["hits"] > before, value


CONST = """
THRESHOLD = {thr}

def helper():
    return THRESHOLD

@cash_instance.cache
def via_helper(n):
    return helper()

@cash_instance.cache
def direct(n):
    return THRESHOLD
"""


def test_unchanged_still_hits(env):
    """Null control. Without it, every miss below could be a miss-always bug."""
    cash, work = env
    hit, _ = call(work, cash, CONST.format(thr=5))
    assert hit is False
    hit, value = call(work, cash, CONST.format(thr=5))
    assert hit is True and value == 5


def test_global_read_by_helper_invalidates(env):
    """The reported hole."""
    cash, work = env
    call(work, cash, CONST.format(thr=5))
    hit, value = call(work, cash, CONST.format(thr=6))
    assert value == 6, f"served {value}, the module now says 6"
    assert hit is False


def test_direct_read_still_invalidates(env):
    """Regression guard: the existing one-level fold must keep working."""
    cash, work = env
    call(work, cash, CONST.format(thr=5), fn_name="direct")
    hit, value = call(work, cash, CONST.format(thr=6), fn_name="direct")
    assert value == 6 and hit is False


TRANSITIVE = """
CONFIG = {cfg}

def inner():
    return CONFIG

def outer():
    return inner()

@cash_instance.cache
def via_helper(n):
    return outer()
"""


def test_global_read_by_a_helpers_helper_invalidates(env):
    """The walk is transitive, so the fold must be too."""
    cash, work = env
    call(work, cash, TRANSITIVE.format(cfg=1))
    hit, value = call(work, cash, TRANSITIVE.format(cfg=2))
    assert value == 2 and hit is False


ACCUMULATOR = """
COUNTER = 0

def bump():
    global COUNTER
    COUNTER += 1
    return COUNTER

@cash_instance.cache
def via_helper(n):
    bump()
    return n * 10
"""


def test_accumulator_does_not_miss_forever(env):
    """The drift guard: a global the helper WRITES must not enter the key.

    Folding it would key the entry on the call's own side effect, so every
    later call would compute a fresh key and never hit -- a silent
    performance cliff, and the exact trap that makes this change risky.
    """
    cash, work = env
    hit, _ = call(work, cash, ACCUMULATOR)
    assert hit is False, "first call computes"
    hit, _ = call(work, cash, ACCUMULATOR)
    assert hit is True, "a helper's own accumulator must not defeat the cache"
    hit, _ = call(work, cash, ACCUMULATOR)
    assert hit is True, "and must keep hitting, not just on the second call"


PREBUILT = """
from dataclasses import dataclass, field

class B:
    def __init__(self, value):
        self.value = value {b_expr}
    def __repr__(self):
        return "B(value=%r)" % (self.value,)

@dataclass
class A:
    value: B = field(default_factory=lambda: B(0))

x = A()

def sub():
    return x

@cash_instance.cache
def via_helper(n):
    return sub()
"""


def test_prebuilt_instance_behind_a_helper_invalidates(env):
    """The reported shape: `x = A()` at module level, returned by a helper.

    Asserts against the uncached truth rather than a literal, so the test
    states the property -- the cache must not disagree with a fresh call.
    """
    cash, work = env
    call(work, cash, PREBUILT.format(b_expr=""))
    mod = load(work, cash, PREBUILT.format(b_expr="+ 1000"))
    truth = repr(mod.sub())
    before = mod.via_helper.cache_info()["hits"]
    got = repr(mod.via_helper(1))
    hit = mod.via_helper.cache_info()["hits"] > before
    assert got == truth, f"cache served {got}, a fresh call gives {truth}"
    assert hit is False
