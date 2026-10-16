"""A function that holds itself in its own defaults is keyed, not recursed into.

networkx's ``argmap`` stores every function it wraps in that function's own
``__kwdefaults__``. When such a function cannot be pickled by reference (its
module name now names another object), cash keys it by its code surface, which
folds its defaults, which hold the function, whose surface folds its
defaults... The walk recursed until ``RecursionError`` and no call that
reached the function cached. The function is now keyed as a back-reference to
the frame already folding it, so nothing is left out of the key.
"""

from __future__ import annotations

import itertools
import sys
import textwrap
import types
from pathlib import Path

import pytest

from cash import Cash
from tests._scripts import run_python

pytestmark = [pytest.mark.core]

_modules = itertools.count()


def _module(body: str) -> types.ModuleType:
    """*body* run as a registered module of its own, as user code is."""
    mod = types.ModuleType(f"_cash_self_ref_{next(_modules)}")
    sys.modules[mod.__name__] = mod
    exec(textwrap.dedent(body), mod.__dict__)
    return mod


SELF_REF = """
    def helper(x, *, _self=None):
        return x + {n}

    helper.__kwdefaults__["_self"] = helper
    real = helper
    helper = None  # the name no longer names it: pickle by reference fails
"""

#: Two classes, each holding the other in a method's defaults. A class's
#: digest is remembered, so this is the cycle where a remembered answer
#: could stand in for a walked one.
CYCLE = """
    class A:
        def m(self, *, _other=None):
            return {n}

    class B:
        def m(self, *, _other=None):
            return 2

    A.m.__kwdefaults__["_other"] = B
    B.m.__kwdefaults__["_other"] = A
    real_a, real_b = A, B
    A = B = None
"""


def test_a_function_holding_itself_caches_across_processes(tmp_path: Path):
    (tmp_path / "job.py").write_text(
        textwrap.dedent(
            """
            import cash


            class Dispatch:  # what networkx binds under the function's name
                def __init__(self, f):
                    self.f = f

                def __call__(self, *a, **k):
                    return self.f(*a, **k)


            def helper(x, *, _self=None):
                return x + 1


            helper.__kwdefaults__["_self"] = helper
            helper = Dispatch(helper)


            @cash.cache
            def g(n):
                return helper(n)


            print(g(1), g(1), g.cache_info()["miss_reasons"], len(g.cache_info()["warnings"]))
            """
        ),
        encoding="utf-8",
    )
    first = run_python("job.py", cwd=tmp_path).stdout.strip().splitlines()[-1]
    second = run_python("job.py", cwd=tmp_path).stdout.strip().splitlines()[-1]
    assert first == "2 2 {'no entry yet': 1} 0"
    assert second == "2 2 {} 0"


def test_editing_a_function_holding_itself_moves_its_digest():
    c = Cash(register_magic=False)
    one = c._code.code_surface_hash(_module(SELF_REF.format(n=1)).real)
    again = c._code.code_surface_hash(_module(SELF_REF.format(n=1)).real)
    other = c._code.code_surface_hash(_module(SELF_REF.format(n=2)).real)
    assert one is not None
    assert one == again
    assert one != other


def test_a_cycle_is_keyed_the_same_whichever_member_is_keyed_first():
    """Inside *a*'s walk, *b* folds a back-reference to *a*; on its own, *b*
    folds *a* in full. Neither answer is remembered for the next walk, or
    *b*'s key would depend on whether *a* was keyed before it."""
    c = Cash(register_magic=False)
    mod = _module(CYCLE.format(n=1))
    c._code.code_surface_hash(mod.real_a)
    b_after_a = c._code.code_surface_hash(mod.real_b)
    b_alone = Cash(register_magic=False)._code.code_surface_hash(mod.real_b)
    assert b_after_a == b_alone

    edited = _module(CYCLE.format(n=2))
    assert Cash(register_magic=False)._code.code_surface_hash(edited.real_b) != b_alone
