"""cash's own wrapper code is never part of a key.

`@cash.cache` returns a wrapper whose ``__module__`` and ``__qualname__``
are the user function's. Every key channel that meets one keys the
function it wraps (code) or its state (data), never the wrapper's code,
so a change inside cash cannot move a user's keys.
"""

from __future__ import annotations

import functools
import os
import warnings

import cash
from cash import Cash
from cash.backends import InMemoryBackend
from cash.decorator.code_identity import CodeIdentity

CASH_DIR = os.path.dirname(os.path.abspath(cash.__file__))

c = Cash(backend=InMemoryBackend(), register_magic=False)


@c.cache
def f(x):
    return x + 1


class Holder:
    fn = f

    def run(self, x):
        return f(x)


TABLE = {"f": f}


@c.cache
def takes(fn, x):
    return fn(x)


@c.cache
def by_table(x):
    return TABLE["f"](x)


@c.cache
def by_default(x, fn=f):
    return fn(x)


def test_no_key_channel_identifies_cash_code(monkeypatch):
    identified: list[str] = []
    real = CodeIdentity._code_identity

    def spy(self, fn):
        code = getattr(fn, "__code__", None)
        if code is not None:
            identified.append(code.co_filename)
        return real(self, fn)

    monkeypatch.setattr(CodeIdentity, "_code_identity", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explained = [
            takes.explain(f, 1),
            takes.explain(Holder().run, 1),
            takes.explain(functools.partial(f), 1),
            by_table.explain(1),
            by_default.explain(1),
        ]
    assert all(e.cache_key for e in explained), [e.reason for e in explained]

    assert identified, "the spy saw no code: the channels it guards never ran"
    from_cash = sorted({p for p in identified if os.path.abspath(p).startswith(CASH_DIR)})
    assert not from_cash, f"cash's own code was keyed: {from_cash}"


def test_a_cached_function_s_surface_is_the_function_it_wraps():
    assert c._code.code_surface_hash(f) == c._code.code_surface_hash(f.__wrapped__)
