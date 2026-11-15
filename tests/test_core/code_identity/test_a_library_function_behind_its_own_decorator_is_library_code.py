"""A library function its module holds behind a decorator is the library's.

Reported against scanpy checked out as the user's code: a global its code
reads held ``typing``'s functions as they are before ``_tp_cache`` wraps
them. ``typing._GenericAlias.__getitem__`` finds the wrapper, not the
function, so the function's name did not find it in its module, and it was
taken for code the user wrote (the exec'd-code case). Its body's
``compile``/``eval`` of forward references then refused every call
("code an argument carries picks code by a run-time name"). Found through
``__wrapped__`` under its own name, the module it is in says whose it is.
"""

from __future__ import annotations

import inspect
import textwrap
import typing

import pytest

from cash.decorator.user_code import is_user_code_object
from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

MAIN = textwrap.dedent("""
    import inspect, sys, typing
    import cash

    HANDLERS = {inspect.unwrap(typing._GenericAlias.__getitem__): "subscript"}


    def wrapped(fn):
        def inner(*args):
            return fn(*args)

        inner.__wrapped__ = fn
        inner.__qualname__ = fn.__qualname__
        return inner


    @wrapped
    def scale(x):
        return x * {FACTOR}


    OWN = {inspect.unwrap(scale): "scale"}


    @cash.cache
    def f(x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return (sorted(HANDLERS.values()), [fn(x) for fn in OWN])


    print(f(1))
""")


def test_the_original_behind_a_library_decorator_is_not_user_code():
    original = inspect.unwrap(typing._GenericAlias.__getitem__)
    assert original is not typing._GenericAlias.__getitem__
    assert not is_user_code_object(original)


def test_a_global_holding_it_is_cached_and_an_own_one_still_keyed(tmp_path):
    def run(factor):
        (tmp_path / "main.py").write_text(MAIN.replace("{FACTOR}", str(factor)), encoding="utf-8")
        proc = run_python("main.py", cwd=tmp_path)
        return proc.stdout.strip(), proc.stderr

    out, err = run(2)
    assert out == "(['subscript'], [2])" and "[RUN]" in err
    out, err = run(2)
    assert "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"
    # The user's own function behind its own decorator is still the user's.
    out, err = run(7)
    assert out == "(['subscript'], [7])", err
