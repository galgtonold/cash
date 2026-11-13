"""Functions holding themselves in their defaults are walked once per key build.

networkx's ``argmap`` stores every function it wraps in that function's own
``__kwdefaults__``, and the functions reach each other. A function holding a
function in its defaults is never kept in the code-surface memo (its defaults
can be rebound), and one holding itself sits on a cycle, so each was walked
again from every function whose defaults or code reached it: work that grew
exponentially with the chain, and the first networkx call spent minutes
building its key. Within one key build such a surface is now walked once,
and served only where walking it again would give the same digest.
"""

from __future__ import annotations

import itertools
import sys
import textwrap
import types

import pytest

from cash import Cash
from cash.decorator.global_values import KEY_BUILD_DIGESTS

pytestmark = [pytest.mark.core]

_modules = itertools.count()

#: f0 .. f{n-1}: each holds itself, as argmap does, and every function after
#: it; the names are rebound so pickling by reference fails, and each is
#: keyed by its code surface. The last one's body adds *k*.
CHAIN = """
    fs = []
    for i in range({n}):
        def f(x, *, _self=None, _next=()):
            return x + 1
        f.__kwdefaults__["_self"] = f
        fs.append(f)
    def last(x, *, _self=None, _next=()):
        return x + {k}
    last.__kwdefaults__["_self"] = last
    fs[-1] = last
    for i, f in enumerate(fs):
        f.__kwdefaults__["_next"] = tuple(fs[i + 1:])
    f = last = None
"""


def _chain(n: int, k: int = 1) -> list:
    mod = types.ModuleType(f"_cash_self_chain_{next(_modules)}")
    sys.modules[mod.__name__] = mod
    exec(textwrap.dedent(CHAIN.format(n=n, k=k)), mod.__dict__)
    return mod.fs


def _in_one_key_build(c: Cash, fn) -> str | None:
    token = KEY_BUILD_DIGESTS.set({})
    try:
        return c._code.code_surface_hash(fn)
    finally:
        KEY_BUILD_DIGESTS.reset(token)


def test_each_function_is_walked_once_in_a_key_build(monkeypatch):
    c = Cash(register_magic=False)
    fs = _chain(16)
    walked: list = []
    own_digest = c._code._own_digest

    def counting(obj, is_type):
        walked.append(obj)
        return own_digest(obj, is_type)

    monkeypatch.setattr(c._code, "_own_digest", counting)
    assert _in_one_key_build(c, fs[0]) is not None
    # 2**(n-1) walks before; each function once now.
    assert len(walked) == len(fs)


def test_the_digest_is_the_one_walking_everything_gives():
    """The memo serves what the walk without it computes, so the key does
    not depend on it."""
    fs = _chain(8)
    walked = Cash(register_magic=False)._code.code_surface_hash(fs[0])
    remembered = _in_one_key_build(Cash(register_magic=False), fs[0])
    assert walked is not None
    assert remembered == walked
    # And the same for each function on its own, entered mid-chain.
    for fn in fs[1:]:
        assert _in_one_key_build(Cash(register_magic=False), fn) == Cash(register_magic=False)._code.code_surface_hash(
            fn
        )


def test_editing_a_function_down_the_chain_moves_the_digest():
    c = Cash(register_magic=False)
    one = _in_one_key_build(c, _chain(12, k=1)[0])
    again = _in_one_key_build(c, _chain(12, k=1)[0])
    other = _in_one_key_build(c, _chain(12, k=2)[0])
    assert one == again
    assert one != other
