"""A function reached by a constant name is a dependency; one reached by a runtime
value is warned about.

Reported against the decorator: a cached function calling a method of its
argument, which called ``getattr(MOD, "fun1")()``. Editing ``fun1`` served the
old result, silently -- the string is a constant, not a name the code loads, so
neither the function's analysis nor the argument's code followed it. With a
name held in a variable (``getattr(MOD, NAME)()``) cash cannot follow it at all;
the cached function's own body refuses that line, but a method of an argument's
class was never analysed, and said nothing.

Each case runs in fresh interpreters on one cache: the edit (``fun1`` returns 10
-> 20) must recompute, and a run without an edit must hit.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

HELPERS = "def fun1():\n    return {RET}\n"

IN_FUNCTION = textwrap.dedent('''
    import sys, time
    import cash
    import helpers

    @cash.cache
    def g(y):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)  # @cash:assume-safe
        return y + getattr(helpers, "fun1")()

    print(g(2))
''')

IN_ARGUMENT_METHOD = textwrap.dedent('''
    import sys, time
    import cash
    import helpers

    class A:
        def __init__(self, x):
            self.x = x

        def f(self, y):
            return self.x + y + getattr(helpers, "fun1")()

    @cash.cache
    def f(a, y):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)  # @cash:assume-safe
        return a.f(y)

    print(f(A(1), 2))
''')

SAME_MODULE_ARGUMENT_METHOD = textwrap.dedent('''
    import sys, time
    import cash

    def fun1():
        return {RET}

    MOD = sys.modules[__name__]

    class A:
        def __init__(self, x):
            self.x = x

        def f(self, y):
            return self.x + y + getattr(MOD, "fun1")()

    @cash.cache
    def f(a, y):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)  # @cash:assume-safe
        return a.f(y)

    print(f(A(1), 2))
''')

RUNTIME_NAME = textwrap.dedent('''
    import sys, time
    import cash

    def fun1():
        return 10

    MOD = sys.modules[__name__]
    NAME = "fun1"

    class A:
        def __init__(self, x):
            self.x = x

        def f(self, y):
            return self.x + y + getattr(MOD, NAME)(){WAIVER}

    @cash.cache
    def f(a, y):
        time.sleep(0.2)  # @cash:assume-safe
        return a.f(y)

    print(f(A(1), 2))
''')


def _run(tmp_path, script, **fmt):
    (tmp_path / "main.py").write_text(script.format(**fmt) if fmt else script)
    # No .pyc: Python trusts one whose source has the same size and the same
    # whole-second mtime, and `return 10` -> `return 20` keeps the size, so an
    # edit landing in the previous run's second imported the old helpers.
    env = dict(os.environ, CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONWARNINGS="always",
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "main.py"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip(), proc.stderr


@pytest.mark.parametrize("script, base, edited", [
    (IN_FUNCTION, "12", "22"),
    (IN_ARGUMENT_METHOD, "13", "23"),
], ids=["in_the_cached_function", "in_an_argument_method"])
def test_editing_a_function_named_by_a_constant_string_recomputes(tmp_path, script, base, edited):
    (tmp_path / "helpers.py").write_text(HELPERS.format(RET=10))
    out, err = _run(tmp_path, script)
    assert out == base and "[RUN]" in err
    out, err = _run(tmp_path, script)
    assert out == base and "[RUN]" not in err, "an unedited run must hit"

    (tmp_path / "helpers.py").write_text(HELPERS.format(RET=20))
    out, err = _run(tmp_path, script)
    assert out == edited, f"served the result from before the edit: {out}"


def test_the_same_through_the_modules_own_globals(tmp_path):
    out, _ = _run(tmp_path, SAME_MODULE_ARGUMENT_METHOD.replace("{RET}", "10"))
    assert out == "13"
    out, _ = _run(tmp_path, SAME_MODULE_ARGUMENT_METHOD.replace("{RET}", "20"))
    assert out == "23"


def test_a_name_held_in_a_variable_in_an_argument_method_is_warned_about(tmp_path):
    out, err = _run(tmp_path, RUNTIME_NAME.replace("{WAIVER}", ""))
    assert out == "13"
    assert "KEY-DYNAMIC-DEPENDENCY" in err and "A.f" in err, err


def test_a_waiver_on_that_line_silences_it(tmp_path):
    out, err = _run(tmp_path, RUNTIME_NAME.replace("{WAIVER}", "  # @cash:assume-safe"))
    assert out == "13"
    assert "KEY-DYNAMIC-DEPENDENCY" not in err, err


# Reported next: the object whose method reaches the function is not the
# argument but held BY it (`A(1, B())`, `A.f` calling `self.b.f()`). The
# argument's class was folded; what the instance held was not looked into, so
# editing `B.f`, or `fun1`/`fun2` behind it, served the old result.
NESTED = textwrap.dedent('''
    import sys, time
    import cash

    def fun1():
        return fun2() + {F1}

    def fun2():
        return {F2}

    FUNS = [fun1, fun2]

    class B:
        def f(self):
            return FUNS[0](){B_EXTRA}

    class A:
        def __init__(self, x, b):
            self.x = x
            self.b = b

        def f(self, y):
            return self.x + y + self.b.f()

    @cash.cache
    def f(a, y):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)  # @cash:assume-safe
        return a.f(y)

    print(f(A(1, B()), 2))
''')


def _nested(**over):
    fmt = {"F1": "1", "F2": "90", "B_EXTRA": ""}
    fmt.update(over)
    return NESTED.replace("{F1}", fmt["F1"]).replace("{F2}", fmt["F2"]).replace("{B_EXTRA}", fmt["B_EXTRA"])


@pytest.mark.parametrize("edit, want", [
    ({"F2": "100"}, "104"),
    ({"F1": "5"}, "98"),
    ({"B_EXTRA": " + 1000"}, "1094"),
], ids=["function_behind_the_held_object", "function_it_calls", "held_objects_method"])
def test_code_of_an_object_the_argument_holds_is_a_dependency(tmp_path, edit, want):
    out, err = _run(tmp_path, _nested())
    assert out == "94" and "[RUN]" in err
    out, err = _run(tmp_path, _nested())
    assert out == "94" and "[RUN]" not in err, "an unedited run must hit"

    out, err = _run(tmp_path, _nested(**edit))
    assert out == want, f"served the result from before the edit: {out}"
