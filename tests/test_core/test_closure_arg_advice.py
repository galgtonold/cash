"""KEY-UNHASHABLE-ARG must not tell you to register a hasher for every function.

CAS-117, round-17 tester r17s4 (F8). Following cash's own warning produced a
wrong answer:

    fit(make_model(3.0), d)
    # Fix: register a hasher with cash.register_hasher(function, ...)
    cash.register_hasher(types.FunctionType, lambda f: f.__qualname__)
    fit(make_model(5.0), d)      # -> make_model(3.0)'s answer

The fix line was built from the argument's type name, which for a closure is
`function` -- a hasher for every function in the process, and the obvious one
(by name) collides every closure a factory makes.
"""
from __future__ import annotations

import functools
import types
import warnings

import pytest

from cash import Cash

pytestmark = pytest.mark.core


def make_scaler(k):
    def scale(x):
        return x * k
    return scale


class Session:
    def __reduce__(self):
        raise TypeError("a live session cannot be pickled")


def _messages(rec, code):
    return [str(w.message) for w in rec if f"[{code}]" in str(w.message)]


@pytest.fixture
def c(tmp_path):
    # The partial arm trips a once-per-process notice (KEY-OPAQUE-CALLABLE);
    # hand the dedup set back as it was, so no later test finds it spent.
    saved = set(Cash._WARNED_UNHASHABLE)
    yield Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    Cash._WARNED_UNHASHABLE.clear()
    Cash._WARNED_UNHASHABLE.update(saved)


@pytest.mark.parametrize("arg", [
    make_scaler(2),
    lambda x: x,
    functools.partial(make_scaler(2), 1),
], ids=["closure", "lambda", "partial-of-closure"])
def test_a_code_argument_is_not_told_to_register_a_type_wide_hasher(c, arg):
    """THE BUG: the fix line said `register_hasher(function, ...)`."""
    @c.cache
    def apply_to(fn, x):
        return x

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        apply_to(arg, 3)

    found = _messages(rec, "KEY-UNHASHABLE-ARG")
    assert found, "the call cached, so this test no longer exercises the warning"
    assert "register_hasher(" not in found[0]
    assert "plain arguments" in found[0]


def test_an_ordinary_unhashable_type_still_gets_the_hasher_advice(c):
    """The control: for a live object the type-wide hasher IS the right fix."""
    @c.cache
    def query(session, n):
        return n

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        query(Session(), 3)

    found = _messages(rec, "KEY-UNHASHABLE-ARG")
    assert found
    assert "cash.register_hasher(Session, ...)" in found[0]


def test_explain_gives_the_same_advice(c):
    @c.cache
    def apply_to(fn, x):
        return x

    hint = apply_to.explain(make_scaler(2), 3).details["hint"]
    assert "register_hasher(" not in hint


@pytest.mark.parametrize("type_", [types.FunctionType, types.MethodType, functools.partial])
def test_registering_a_hasher_for_every_callable_warns(c, type_):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.register_hasher(type_, lambda f: getattr(f, "__qualname__", "?"))
    assert _messages(rec, "KEY-CALLABLE-HASHER")


def test_registering_a_hasher_for_an_ordinary_type_is_silent(c):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.register_hasher(Session, lambda s: "one")
    assert not _messages(rec, "KEY-CALLABLE-HASHER")


# ---------------------------------------------------------------------------
# Round 18 (r18s5, F4): the argument NAMED was the first one of a non-built-in
# type, not the one that failed. `score(df, lambda d: ...)` blamed the
# DataFrame and advised a DataFrame hasher -- rejected by cash, and with
# override=True a re-key of every DataFrame function -- while the lambda was
# the culprit. Each candidate is now hashed on its own.
# ---------------------------------------------------------------------------

def test_a_frame_next_to_a_lambda_blames_the_lambda(c):
    pd = pytest.importorskip("pandas")

    @c.cache
    def score(df, fn):
        return float(fn(df).sum().sum())

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        score(pd.DataFrame({"a": [1, 2]}), lambda d: d * 2)
    found = _messages(rec, "KEY-UNHASHABLE-ARG")
    assert found
    assert "of type function" in found[0]
    assert "DataFrame" not in found[0]


def test_an_unpicklable_object_after_a_frame_is_named(c):
    pd = pytest.importorskip("pandas")

    @c.cache
    def use(df, session):
        return len(df)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        use(pd.DataFrame({"a": [1]}), Session())
    found = _messages(rec, "KEY-UNHASHABLE-ARG")
    assert found and "of type Session" in found[0]
