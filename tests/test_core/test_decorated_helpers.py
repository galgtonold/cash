"""A decorated helper is every function it runs: the wrapper AND the wrapped.

Round 18 (r18s1): with ``functools.wraps`` only the wrapped function was
followed, so an edit to the wrapper's body was served stale; without it, only
the wrapper was, so an edit to the wrapped function was. The follow-up matrix
found the same hole in every shape tried -- stacks, class-based decorators,
decorator arguments, library wrappers that hold the user's function
(``np.vectorize``, ``toolz.curry``, ``wrapt``, the ``decorator`` package,
``singledispatch``) -- and three worse ones underneath:

* two helpers wrapped by the SAME decorator shared one digest (it was
  memoized under the wrapper's code object, which both share);
* the function inside a ``wraps`` decorator was analysed in the DECORATOR
  module's namespace, so nothing it called was followed (``@cash.cache`` over
  ``@timed`` lost every helper);
* what the wrapped function reads (``K`` in ``return x * K``) never reached the
  key, because the object bound to the name was the wrapper.

Each case: write v1, run cached twice (cold, then warm -- which must not
execute), apply one edit, then compare a cached run with ``CASH_DISABLE=1``.
Every step is a fresh process; the body sleeps past the persistence floor.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

# Several fresh interpreters per case: past the suite-wide 30 s on a loaded runner.
pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

D = textwrap.dedent

JOB = D('''
    import os, sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cash
    from helpers import h

    @cash.cache
    def f(x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)  # @cash:assume-safe
        return h(x)

    print(float(f(2)))
''')

DECOS = D('''
    import functools
    FACTOR = 1

    def add1(fn):
        @functools.wraps(fn)
        def wrapper(x):
            return fn(x) + 1
        return wrapper

    def times10(fn):
        @functools.wraps(fn)
        def wrapper(x):
            return fn(x) * 10
        return wrapper

    def scale(k):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(x):
                return fn(x) * k
            return wrapper
        return deco

    def scale_nowraps(k):
        def deco(fn):
            def wrapper(x):
                return fn(x) * k
            return wrapper
        return deco

    def by_factor(fn):
        @functools.wraps(fn)
        def wrapper(x):
            return fn(x) * FACTOR
        return wrapper

    class Traced:
        def __init__(self, fn):
            self.fn = fn
            functools.update_wrapper(self, fn)
        def __call__(self, x):
            return self.fn(x) * 1
''')

INNER_DEP = "def dep(x):\n    return x + 1\n"

STACK = '''
    from decos import add1, times10
    @add1
    @times10
    def h(x):
        return x + 1
'''
TRACED = '''
    from decos import Traced
    @Traced
    def h(x):
        return x + 1
'''
WRAPT = '''
    import wrapt
    @wrapt.decorator
    def doubled(wrapped, instance, args, kwargs):
        return wrapped(*args, **kwargs) * 2
    @doubled
    def h(x):
        return x + 1
'''
DECORATOR = '''
    from decorator import decorator
    @decorator
    def doubled(fn, *args, **kwargs):
        return fn(*args, **kwargs) * 2
    @doubled
    def h(x):
        return x + 1
'''
ROOT_JOB = JOB.replace("@cash.cache\n", "@cash.cache\n@add1\n").replace(
    "from helpers import h", "from helpers import h\nfrom decos import add1")
TWO_JOB = JOB.replace("from helpers import h", "from helpers import h, h_other").replace(
    "    return h(x)\n",
    "    return h(x)\n\n\n@cash.cache\ndef g(x):\n    time.sleep(0.2)  # @cash:assume-safe\n"
    "    return h_other(x)\n",
).replace("print(float(f(2)))", "g(2)\nprint(float(f(2)))")

# (id, helpers.py, (file, old, new), job, required module)
CASES = [
    ("stack-helper-body", STACK, ("helpers.py", "x + 1\n", "x + 100\n"), JOB, None),
    ("stack-outer-wrapper", STACK, ("decos.py", "fn(x) + 1", "fn(x) + 5"), JOB, None),
    ("stack-inner-wrapper", STACK, ("decos.py", "fn(x) * 10", "fn(x) * 20"), JOB, None),
    ("stack-order", STACK, ("helpers.py", "@add1\n@times10", "@times10\n@add1"), JOB, None),
    ("arg-at-def-wraps", '''
        from decos import scale
        @scale(10)
        def h(x):
            return x + 1
    ''', ("helpers.py", "@scale(10)", "@scale(100)"), JOB, None),
    ("arg-at-def-nowraps", '''
        from decos import scale_nowraps
        @scale_nowraps(10)
        def h(x):
            return x + 1
    ''', ("helpers.py", "@scale_nowraps(10)", "@scale_nowraps(100)"), JOB, None),
    ("arg-is-module-constant", '''
        from decos import scale
        K = 10
        @scale(K)
        def h(x):
            return x + 1
    ''', ("helpers.py", "K = 10", "K = 100"), JOB, None),
    ("nowraps-helper-body", '''
        from decos import scale_nowraps
        @scale_nowraps(10)
        def h(x):
            return x + 1
    ''', ("helpers.py", "x + 1\n", "x + 100\n"), JOB, None),
    ("nowraps-two-layers", '''
        from decos import scale_nowraps
        @scale_nowraps(2)
        @scale_nowraps(10)
        def h(x):
            return x + 1
    ''', ("helpers.py", "x + 1\n", "x + 100\n"), JOB, None),
    ("wrapper-reads-its-global", '''
        from decos import by_factor
        @by_factor
        def h(x):
            return x + 1
    ''', ("decos.py", "FACTOR = 1", "FACTOR = 7"), JOB, None),
    ("wrapped-reads-its-global", '''
        from decos import add1
        K = 10
        @add1
        def h(x):
            return x * K
    ''', ("helpers.py", "K = 10", "K = 100"), JOB, None),
    ("wrapped-calls-a-helper", '''
        from decos import add1
        from inner_dep import dep
        @add1
        def h(x):
            return dep(x)
    ''', ("inner_dep.py", "x + 1\n", "x + 100\n"), JOB, None),
    ("class-decorator-helper-body", TRACED, ("helpers.py", "x + 1\n", "x + 100\n"), JOB, None),
    ("class-decorator-call", TRACED, ("decos.py", "self.fn(x) * 1", "self.fn(x) * 3"), JOB, None),
    ("same-decorator-two-helpers", '''
        from decos import add1
        @add1
        def h_other(x):
            return x + 7
        @add1
        def h(x):
            return x + 1
    ''', ("helpers.py", "return x + 1\n", "return x + 100\n"), TWO_JOB, None),
    ("root-over-wrapper-body", '''
        def h(x):
            return x + 1
    ''', ("decos.py", "fn(x) + 1", "fn(x) + 5"), ROOT_JOB, None),
    ("root-over-wrapper-helper", '''
        def h(x):
            return x + 1
    ''', ("helpers.py", "x + 1\n", "x + 100\n"), ROOT_JOB, None),
    ("lru-cache", '''
        import functools
        @functools.lru_cache(maxsize=None)
        def h(x):
            return x + 1
    ''', ("helpers.py", "x + 1\n", "x + 100\n"), JOB, None),
    ("singledispatch-impl", '''
        from functools import singledispatch
        @singledispatch
        def h(x):
            return -1
        @h.register(int)
        def _(x):
            return x + 1
    ''', ("helpers.py", "x + 1\n", "x + 100\n"), JOB, None),
    ("np-vectorize-body", '''
        import numpy as np
        @np.vectorize
        def h(x):
            return x / 3
    ''', ("helpers.py", "x / 3\n", "x / 5\n"), JOB, "numpy"),
    ("np-vectorize-arg", '''
        import numpy as np
        @np.vectorize(otypes=[float])
        def h(x):
            return x / 3
    ''', ("helpers.py", "otypes=[float]", "otypes=[int]"), JOB, "numpy"),
    ("np-vectorize-arg-constant", '''
        import numpy as np
        OT = [float]
        @np.vectorize(otypes=OT)
        def h(x):
            return x / 3
    ''', ("helpers.py", "OT = [float]", "OT = [int]"), JOB, "numpy"),
    ("tenacity-body", '''
        from tenacity import retry, stop_after_attempt
        @retry(stop=stop_after_attempt(3))
        def h(x):
            return x + 1
    ''', ("helpers.py", "x + 1\n", "x + 100\n"), JOB, "tenacity"),
    ("wrapt-body", WRAPT, ("helpers.py", "x + 1\n", "x + 100\n"), JOB, "wrapt"),
    ("wrapt-user-wrapper", WRAPT, ("helpers.py", "* 2\n", "* 3\n"), JOB, "wrapt"),
    ("decorator-pkg-body", DECORATOR, ("helpers.py", "x + 1\n", "x + 100\n"), JOB, "decorator"),
    ("decorator-pkg-caller", DECORATOR, ("helpers.py", "* 2\n", "* 3\n"), JOB, "decorator"),
    ("toolz-curry", '''
        from toolz import curry
        @curry
        def h(x, y=1):
            return x + y
    ''', ("helpers.py", "x + y\n", "x + y * 100\n"), JOB, "toolz"),
    ("pydantic-validate-call", '''
        from pydantic import validate_call
        @validate_call
        def h(x: int):
            return x + 1
    ''', ("helpers.py", "x + 1\n", "x + 100\n"), JOB, "pydantic"),
]


def _run(proj, *, disable=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    if disable:
        env["CASH_DISABLE"] = "1"
    p = subprocess.run([sys.executable, str(proj / "job.py")], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.strip(), "[RUN]" in p.stderr


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_an_edit_to_any_layer_of_a_decorated_helper_reaches_the_key(tmp_path, case):
    _, helpers, (edited, old, new), job, needs = case
    if needs is not None and importlib.util.find_spec(needs) is None:
        pytest.skip(f"{needs} is not installed")
    files = {"job.py": job, "helpers.py": D(helpers), "decos.py": DECOS, "inner_dep.py": INNER_DEP}
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    before, _ = _run(tmp_path)
    warm, warm_ran = _run(tmp_path)
    assert warm == before and not warm_ran, "the unedited second run did not hit"

    assert old in files[edited]
    (tmp_path / edited).write_text(files[edited].replace(old, new), encoding="utf-8")
    got, _ = _run(tmp_path)
    want, _ = _run(tmp_path, disable=True)
    assert want != before, "the edit should change the answer"
    assert got == want, f"served {got} after the edit, the code now computes {want}"
