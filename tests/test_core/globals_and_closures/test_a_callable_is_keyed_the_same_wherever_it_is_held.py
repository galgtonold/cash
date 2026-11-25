"""A callable is keyed by what it was built with, wherever it is held.

"What was this callable built with?" has one answer
(`HelperIdentity.callable_state`): its code, its defaults, what its closure
captured, the instance a method is bound to, what a ``functools.partial``
binds, what a wrapper wraps, the attributes stored on a function, and the
callables inside a captured list or dict, however deep. Every place that
can hold a callable asks it: an argument, the cached function's own closure,
a module global, a parameter default, an object, a helper.

The matrix below is every holder against every shape. A cell builds the
callable with one number and calls the cached function, in three processes
over one cache folder:

1. each shape with its own number: a wrong value here is one shape served
   another shape's entry in the same process;
2. other numbers: a wrong value is the old result served in a new process
   (the callable's state never reached the key);
3. the first numbers again: every cell must be a hit, so no cell passes by
   running uncached.

A new holder is one ``@holder`` function in `SCRIPT`, a new shape one
``@shape`` function.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

SCRIPT = textwrap.dedent("""
    import functools
    import json
    import sys
    import warnings

    import cash

    HOLDERS, SHAPES = {}, {}


    def holder(fn):
        HOLDERS[fn.__name__] = fn
        return fn


    def shape(fn):
        SHAPES[fn.__name__] = fn
        return fn


    class Model:
        def __init__(self, w):
            self.w = w

        def predict(self, x):
            return x * self.w

        def __call__(self, x):
            return x * self.w


    def logged(fn):
        @functools.wraps(fn)
        def wrapper(x):
            return fn(x)

        return wrapper


    def times(k, x):
        return x * k


    def call_with(fn, x):
        return fn(x)


    # --- what the callable was built with: one function per shape --------

    @shape
    def closure(k):
        def a(x):
            return x * k

        return a


    @shape
    def bound_method(k):
        return Model(k).predict


    @shape
    def partial(k):
        return functools.partial(times, k)


    @shape
    def callable_object(k):
        return Model(k)


    @shape
    def lambda_default(k):
        return lambda x, k=k: x * k


    @shape
    def function_attribute(k):
        def a(x):
            return x * a.k

        a.k = k
        return a


    @shape
    def decorated(k):
        @logged
        def a(x):
            return x * k

        return a


    @shape
    def lru_cached(k):
        @functools.lru_cache
        def a(x):
            return x * k

        return a


    @shape
    def closure_over_closure(k):
        a = closure(k)

        def b(x):
            return a(x)

        return b


    @shape
    def three_levels(k):
        b = closure_over_closure(k)

        def c(x):
            return b(x)

        return c


    @shape
    def partial_of_closure(k):
        return functools.partial(closure(k))


    @shape
    def partial_binding_a_closure(k):
        return functools.partial(call_with, closure(k))


    @shape
    def closure_on_an_object(k):
        m = Model(0)
        m.predict2 = closure(k)
        return lambda x: m.predict2(x)


    @shape
    def list_of_closures(k):
        fns = [closure(k)]
        return lambda x: fns[0](x)


    @shape
    def dict_of_closures(k):
        table = {"f": closure(k)}
        return lambda x: table["f"](x)


    @shape
    def closure_over_dict(k):
        cfg = {"w": k}

        def a(x):
            return x * cfg["w"]

        def b(x):
            return a(x)

        return b


    @shape
    def closure_as_a_default(k):
        def a(x, fn=closure(k)):
            return fn(x)

        return a


    @shape
    def bound_method_in_a_closure(k):
        predict = Model(k).predict
        return lambda x: predict(x)


    @shape
    def partial_in_a_closure(k):
        mul = functools.partial(times, k)
        return lambda x: mul(x)


    # --- where the callable is held: one function per holder --------------
    # Each returns the cached function and a call of it.

    def wrap(c):
        def outer(x):
            return c(x)

        return outer


    class Box:
        def __init__(self, fn):
            self.fn = fn


    C = W = T = None


    @cash.cache
    def takes_function(fn, x):
        return fn(x)


    @cash.cache
    def takes_list(fns, x):
        return fns[0](x)


    @cash.cache
    def takes_object(box, x):
        return box.fn(x)


    @cash.cache
    def reads_global(x):
        return C(x)


    @cash.cache
    def reads_wrapped_global(x):
        return W(x)


    @cash.cache
    def reads_global_dict(x):
        return T["f"](x)


    def helper(x):
        return C(x)


    @cash.cache
    def calls_helper(x):
        return helper(x)


    class Pipe:
        def __init__(self, fn):
            self.fn = fn

        @cash.cache
        def run(self, x):
            return self.fn(x)


    @holder
    def argument(c):
        return takes_function, lambda: takes_function(c, 3)


    @holder
    def argument_in_a_list(c):
        return takes_list, lambda: takes_list([c], 3)


    @holder
    def argument_wrapped(c):
        return takes_function, lambda: takes_function(wrap(c), 3)


    @holder
    def own_closure(c):
        @cash.cache
        def f(x):
            return c(x)

        return f, lambda: f(3)


    @holder
    def own_closure_wrapped(c):
        outer = wrap(c)

        @cash.cache
        def f(x):
            return outer(x)

        return f, lambda: f(3)


    @holder
    def module_global(c):
        global C
        C = c
        return reads_global, lambda: reads_global(3)


    @holder
    def module_global_wrapped(c):
        global W
        W = wrap(c)
        return reads_wrapped_global, lambda: reads_wrapped_global(3)


    @holder
    def default(c):
        @cash.cache
        def f(x, fn=c):
            return fn(x)

        return f, lambda: f(3)


    @holder
    def default_wrapped(c):
        @cash.cache
        def f(x, fn=wrap(c)):
            return fn(x)

        return f, lambda: f(3)


    @holder
    def object_argument(c):
        return takes_object, lambda: takes_object(Box(c), 3)


    @holder
    def object_argument_wrapped(c):
        return takes_object, lambda: takes_object(Box(wrap(c)), 3)


    @holder
    def method_self(c):
        return Pipe.run, lambda: Pipe(c).run(3)


    @holder
    def module_global_dict(c):
        global T
        T = {"f": c}
        return reads_global_dict, lambda: reads_global_dict(3)


    @holder
    def helper_reading_a_global(c):
        global C
        C = c
        return calls_helper, lambda: calls_helper(3)


    if __name__ == "__main__":
        row, numbers = HOLDERS[sys.argv[1]], json.loads(sys.argv[2])
        out = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for name, k in numbers.items():
                seen = len(caught)
                cached, call = row(SHAPES[name](k))
                before = cached.cache_info()
                value = call()
                after = cached.cache_info()
                out[name] = {
                    "value": value,
                    "hits": after["hits"] - before["hits"],
                    "misses": after["misses"] - before["misses"],
                    "warnings": sorted({str(w.message).split(":")[0][:60] for w in caught[seen:]}),
                }
        print(json.dumps(out))
""")

HOLDERS = re.findall(r"^@holder\ndef (\w+)", SCRIPT, re.M)
SHAPES = re.findall(r"^@shape\ndef (\w+)", SCRIPT, re.M)


def _numbers(base: int) -> dict[str, int]:
    """One number per shape, no two alike within a run or across runs."""
    return {name: base + 10 * i for i, name in enumerate(SHAPES)}


FIRST, SECOND = _numbers(2), _numbers(5)


def _row(holder: str, tmp_path_factory) -> list[dict]:
    """The three runs of *holder*'s row, made once however many tests and
    xdist workers ask: the first to ask runs them, the rest read the file."""
    base = tmp_path_factory.getbasetemp()
    shared = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    folder = Path(shared) / "callable-matrix" / holder
    done = folder / "rows.json"
    try:
        folder.mkdir(parents=True)
    except FileExistsError:
        deadline = time.monotonic() + 110
        while not done.exists():
            assert time.monotonic() < deadline, f"the runs of {holder} did not finish"
            time.sleep(0.1)
        return json.loads(done.read_text())
    rows: list = []
    try:
        (folder / "main.py").write_text(SCRIPT)
        for numbers in (FIRST, SECOND, FIRST):
            out = run_python("main.py", holder, json.dumps(numbers), cwd=folder).stdout
            rows.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        partial = folder / "rows.json.partial"
        partial.write_text(json.dumps(rows if len(rows) == 3 else {"failed": holder}))
        partial.replace(done)
    return rows


@pytest.mark.timeout(120)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("holder", HOLDERS)
def test_what_the_callable_was_built_with_reaches_the_key(holder, shape, tmp_path_factory):
    rows = _row(holder, tmp_path_factory)
    assert isinstance(rows, list), f"the runs of {holder} failed"
    first, second, again = (run[shape] for run in rows)
    assert first["value"] == 3 * FIRST[shape], "served another shape's entry in one process"
    assert second["value"] == 3 * SECOND[shape], "served the result of a callable built with another value"
    assert again["value"] == 3 * FIRST[shape]
    assert (again["hits"], again["misses"]) == (1, 0), f"not served from the cache: {again}"
    assert not any("KEY-" in w or "uncached" in w for w in first["warnings"]), first["warnings"]


def test_the_matrix_has_every_holder_and_shape():
    """A positive control for the parsing above: a holder or shape the
    regular expressions missed would silently leave the matrix."""
    assert len(HOLDERS) == SCRIPT.count("@holder\n") == 14
    assert len(SHAPES) == SCRIPT.count("@shape\n") == 19
