"""CAS-224: a statement calling @cash.cache(ttl=0) must inherit that TTL.

Under %cash_on, ``x = f()`` is cached as an ordinary statement. When ``f`` is
decorated ``@cash.cache(ttl=0)`` — "recompute every call" — the statement cache
had no TTL and froze ``x`` at the first result, silently overriding the freshness
the decorator promised. ``_ttl_floor_from_called_functions`` lowers the
statement's effective TTL to the smallest TTL of any cash-wrapped function it
calls, so ttl=0 rides the immediate-expiry path (CAS-221) and the body runs every
run.

These unit-test the helper directly with stub wrappers, so no kernel is needed.
The end-to-end behaviour is pinned by the real-driver reproducer in r11p2.
"""
from __future__ import annotations

from cash.notebook.statement import StatementProcessor


class _Shell:
    def __init__(self, ns):
        self.user_ns = ns


class _Stub:
    """Only ``self.shell.user_ns`` is read by the method under test."""
    def __init__(self, ns):
        self.shell = _Shell(ns)


def _wrapper(ttl):
    def f():  # pragma: no cover - never called
        return 1
    f._cash_cached = True
    f._cash_declared_ttl = ttl
    return f


def _floor(ns, inputs, effective_ttl):
    return StatementProcessor._ttl_floor_from_called_functions(_Stub(ns), set(inputs), effective_ttl)


def test_ttl_zero_function_floors_the_statement():
    """The regression: a ttl=0 call drags the statement TTL to 0."""
    ns = {'f': _wrapper(0)}
    assert _floor(ns, ['f'], None) == 0


def test_plain_decorated_call_is_untouched():
    """ttl=None wrapper must leave the statement TTL exactly as it was."""
    ns = {'f': _wrapper(None)}
    assert _floor(ns, ['f'], None) is None


def test_non_wrapper_input_is_ignored():
    """An ordinary variable in inputs must not change the TTL."""
    ns = {'df': [1, 2, 3]}
    assert _floor(ns, ['df'], None) is None
    assert _floor(ns, ['df'], 300) == 300


def test_only_lowers_never_raises_the_ttl():
    """A ttl=3600 function must not extend a statement that was tighter."""
    ns = {'f': _wrapper(3600)}
    assert _floor(ns, ['f'], 60) == 60          # keep the tighter statement TTL
    assert _floor(ns, ['f'], None) == 3600       # but adopt it when unset


def test_minimum_across_several_called_functions():
    ns = {'f': _wrapper(3600), 'g': _wrapper(0), 'h': _wrapper(None)}
    assert _floor(ns, ['f', 'g', 'h'], None) == 0


def test_absent_name_is_safe():
    """A called name not yet in the namespace must not raise."""
    assert _floor({}, ['not_here'], 120) == 120


def test_missing_declared_ttl_attr_is_safe():
    """A cash wrapper without the TTL stamp (legacy) is skipped, not crashed."""
    def f():  # pragma: no cover
        return 1
    f._cash_cached = True  # no _cash_declared_ttl
    assert _floor({'f': f}, ['f'], None) is None
