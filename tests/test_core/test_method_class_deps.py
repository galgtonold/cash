"""A cached method must see the class-level code it reaches.

`@cash.cache` on a method keyed on the method's own source + self's instance
state, but nothing it reached through ``self``/``cls``/``super()`` -- helper
methods, class constants, base-class bodies -- so editing those silently served
a stale result (CAS-237). At decoration time the class does not exist yet; the
fix resolves these against the real class at CALL time, where ``self`` is known.

Cross-process, because the stale serve only appears when a second process
rebuilds the key and matches the persisted entry. ``time.sleep(0.3)`` clears the
persistence floor, or nothing persists and the test is vacuous.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow


def _run(tmp_path, script="main.py"):
    cp = subprocess.run(
        [sys.executable, script], cwd=str(tmp_path),
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, f"{script} failed:\n{cp.stdout}\n{cp.stderr}"
    return cp.stdout.strip()


def _result(out):
    return out.splitlines()[-1].split("R ", 1)[1].strip()


HELPER_METHOD = '''\
import warnings; warnings.simplefilter("ignore")
import time, cash
class Model:
    def helper(self, x):
        return x * {k}
    @cash.cache
    def compute(self, x):
        time.sleep(0.3)
        return self.helper(x) + 1
print("R", Model().compute(21))
'''

CLASS_ATTR = '''\
import warnings; warnings.simplefilter("ignore")
import time, cash
class M:
    RATE = {k}
    @cash.cache
    def price(self, b):
        time.sleep(0.3)
        return b * self.RATE
print("R", M().price(100))
'''

SUPER_CALL = '''\
import warnings; warnings.simplefilter("ignore")
import time, cash
class Base:
    def base_calc(self, x):
        return x + {k}
class Child(Base):
    @cash.cache
    def compute(self, x):
        time.sleep(0.3)
        return super().base_calc(x) * 2
print("R", Child().compute(10))
'''

CLASSMETHOD = '''\
import warnings; warnings.simplefilter("ignore")
import time, cash
class M:
    FACTOR = {k}
    @classmethod
    @cash.cache
    def scaled(cls, x):
        time.sleep(0.3)
        return x * cls.FACTOR
print("R", M.scaled(10))
'''


@pytest.mark.parametrize("template,k1,r1,k2,r2", [
    (HELPER_METHOD, "2", "43", "10", "211"),
    (CLASS_ATTR, "0.10", "10.0", "0.50", "50.0"),
    (SUPER_CALL, "5", "30", "100", "220"),
    (CLASSMETHOD, "3", "30", "7", "70"),
])
def test_class_level_edit_invalidates(tmp_path, template, k1, r1, k2, r2):
    main = tmp_path / "main.py"
    main.write_text(template.format(k=k1), encoding="utf-8")
    assert _result(_run(tmp_path)) == r1

    main.write_text(template.format(k=k2), encoding="utf-8")
    # Cache is NOT cleared: a stale serve would return r1.
    assert _result(_run(tmp_path)) == r2, (
        "class-level change did not invalidate the cached method"
    )


# --- Transitive (multi-hop) reachability: a helper's helper, a constant read
# INSIDE a called helper, and a property getter must all invalidate too. ---

CONST_BEHIND_HELPER = """import warnings; warnings.simplefilter("ignore")
import time, cash
class Model:
    RATE = {k}
    def helper(self, x):
        return x + self.RATE          # helper reads RATE; compute does not
    @cash.cache
    def compute(self, x):
        time.sleep(0.3)
        return self.helper(x) + 4
print("R", Model().compute(10))
"""

DEEP_CHAIN = """import warnings; warnings.simplefilter("ignore")
import time, cash
class Model:
    def c(self, x):
        return x + {k}                # 3 hops from compute
    def b(self, x):
        return self.c(x) * 2
    def a(self, x):
        return self.b(x)
    @cash.cache
    def compute(self, x):
        time.sleep(0.3)
        return self.a(x)
print("R", Model().compute(10))
"""

PROPERTY_GETTER = """import warnings; warnings.simplefilter("ignore")
import time, cash
class Model:
    def __init__(self):
        self.x = 9
    @property
    def computed(self):
        return self.x + {k}
    @cash.cache
    def total(self, y):
        time.sleep(0.3)
        return self.computed + y
print("R", Model().total(0))
"""


@pytest.mark.parametrize("template,k1,r1,k2,r2", [
    (CONST_BEHIND_HELPER, "10", "24", "100", "114"),
    (DEEP_CHAIN, "1", "22", "99", "218"),
    (PROPERTY_GETTER, "1", "10", "100", "109"),
])
def test_transitive_class_edit_invalidates(tmp_path, template, k1, r1, k2, r2):
    main = tmp_path / "main.py"
    main.write_text(template.format(k=k1), encoding="utf-8")
    assert _result(_run(tmp_path)) == r1

    main.write_text(template.format(k=k2), encoding="utf-8")
    # Cache NOT cleared: a stale serve (one-hop-only tracking) returns r1.
    assert _result(_run(tmp_path)) == r2, (
        "a transitively-reached class dependency did not invalidate"
    )


import cash


class _HitC:
    """Module-level so instances pickle (a nested class's self is unhashable)."""
    R = 5
    ran = 0

    def helper(self, x):
        return x * 2

    @cash.cache
    def f(self, x):
        _HitC.ran += 1
        return self.helper(x) + self.R


class _StateD:
    def __init__(self, base):
        self.base = base

    def h(self, x):
        return x + self.base

    @cash.cache
    def g(self, x):
        return self.h(x)


def test_method_hit_is_stable_and_instances_are_isolated():
    """No over-invalidation, and per-instance state still distinguishes."""
    _HitC.ran = 0
    c = _HitC()
    c.f(1); c.f(1); c.f(1)
    assert _HitC.ran == 1, "a class-dep method that did not change should HIT"

    assert _StateD(100).g(1) == 101
    assert _StateD(200).g(1) == 201, "instance state must still distinguish entries"


def test_plain_function_with_self_named_param_is_not_treated_as_method():
    """A plain function whose first arg is named ``self`` must not fold a
    look-alike class member, and must not crash."""
    from cash import Cash
    from cash.backends import InMemoryBackend

    inst = Cash(backend=InMemoryBackend(), register_magic=False)

    @inst.cache
    def notamethod(self, x):
        return self + x

    assert notamethod(5, 3) == 8
    assert notamethod(5, 3) == 8  # HIT, no crash
