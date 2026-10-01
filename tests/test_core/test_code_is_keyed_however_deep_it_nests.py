"""Code reaches the key however deeply it is nested, wrapped or referenced.

Each walk that decides which code is in a decorator key stopped at a fixed
depth or count -- nested code objects four or eight levels down, eight
decorators, eight partials, four closures of closures, four class references,
two nested class attributes, 32 singledispatch implementations -- and keyed
what lay beyond by name, or not at all. Editing code past the cut served the
result from before the edit. Every walk now follows everything; a visited set
ends cycles.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

import cash
from cash import Cash
from cash.code_digest import opaque_identity
from cash.decorator.key_values import stabilize_for_global_hash

pytestmark = [pytest.mark.core]


# -- a project run in fresh processes, edited between runs --------------------


def _write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    past = time.time() - 30  # saved a while before the run, like an ordinary edit
    os.utime(path, (past, past))


def _run(project: Path) -> tuple[str, int]:
    """Run ``job.py``: its answer, and how many times the body ran."""
    src = str(Path(cash.__file__).resolve().parents[1])  # the cash under test
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        CASH_CACHE_DIR=str(project / ".cash"),
        PYTHONPATH=os.pathsep.join([src, str(project)]),
    )
    p = subprocess.run(
        [sys.executable, "job.py"], cwd=str(project), env=env, capture_output=True, text=True, timeout=120
    )
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout.strip().splitlines()[-1], p.stderr.count("[RUN]")


JOB = """
    import sys
    import time

    import cash
    import helper


    @cash.cache
    def run(arg):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return {body}


    print(run(helper.ARG))
"""


def _edited_answer(tmp_path: Path, helper: str, edit: tuple[str, str], body: str) -> list[tuple[str, int]]:
    _write(tmp_path / "helper.py", helper)
    _write(tmp_path / "job.py", JOB.format(body=body))
    runs = [_run(tmp_path), _run(tmp_path)]
    old, new = edit
    assert old in helper
    _write(tmp_path / "helper.py", helper.replace(old, new))
    runs.append(_run(tmp_path))
    return runs


#: A class passed as an argument whose code names the next class, six deep:
#: the reference walk stopped four classes down.
REFERENCE_CHAIN = """
    class F:
        def go(self):
            return 1

    class E:
        def go(self):
            return F().go()

    class D:
        def go(self):
            return E().go()

    class C:
        def go(self):
            return D().go()

    class B:
        def go(self):
            return C().go()

    class A:
        def go(self):
            return B().go()

    ARG = A
"""

#: Callable instances held as class attributes, four levels: the nested
#: class walk stopped two levels down.
NESTED_CALLABLE_ATTRIBUTES = """
    class Leaf:
        def __call__(self):
            return 1

    class Inner:
        deeper = Leaf()

        def __call__(self):
            return self.deeper()

    class Mid:
        inner = Inner()

        def __call__(self):
            return self.inner()

    class Outer:
        inner = Mid()

    ARG = Outer
"""

#: A method default that cannot be pickled, a lambda six lists deep: it was
#: identified as "<deep>" below four.
DEEP_UNPICKLABLE_DEFAULT = """
    class Model:
        def go(self, fns=[[[[[[lambda: 1]]]]]]):
            fns = fns[0][0][0][0][0]
            return fns[0]()

    ARG = Model
"""

#: A singledispatch function with 40 implementations: 32 were followed.
MANY_DISPATCH_IMPLEMENTATIONS = """
    import functools

    @functools.singledispatch
    def describe(x):
        return 0

    KINDS = [type(f"K{i}", (), {}) for i in range(40)]
    for kind in KINDS[:-1]:
        describe.register(kind, lambda x: -1)

    @describe.register(KINDS[-1])
    def _(x):
        return 1

    ARG = None
"""


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("helper", "edit", "body"),
    [
        pytest.param(REFERENCE_CHAIN, ("return 1", "return 2"), "arg().go()", id="class-reference-chain"),
        pytest.param(NESTED_CALLABLE_ATTRIBUTES, ("return 1", "return 2"), "arg.inner()", id="nested-class-attributes"),
        pytest.param(DEEP_UNPICKLABLE_DEFAULT, ("lambda: 1", "lambda: 2"), "arg().go()", id="unpicklable-default"),
        pytest.param(
            MANY_DISPATCH_IMPLEMENTATIONS,
            ("return 1", "return 2"),
            "helper.describe(helper.KINDS[-1]())",
            id="dispatch-registry",
        ),
    ],
)
def test_an_edit_to_deeply_reached_code_recomputes(tmp_path, helper, edit, body):
    first, hit, edited = _edited_answer(tmp_path, helper, edit, body)
    assert first == ("1", 1)
    assert hit == ("1", 0)  # the control: unedited, it is served
    assert edited == ("2", 1)


# -- in one process -----------------------------------------------------------


def _fileless_module(name: str, body: str) -> types.ModuleType:
    """Run *body* in a registered module with no file, as a notebook cell or
    ``exec`` does: its functions have no source to read, only bytecode."""
    module = sys.modules.get(name) or types.ModuleType(name)
    sys.modules[name] = module
    exec(textwrap.dedent(body), module.__dict__)
    return module


def _nested_lambdas(levels: int, value: int) -> str:
    return "lambda: " * levels + str(value)


def test_captured_lambdas_on_one_line_differing_ten_levels_down_do_not_collide():
    """Their source is one line; only the code tells them apart, and nested
    code was compared four (fingerprint) and eight (bytecode) levels down."""
    c = Cash()

    def make(fn):
        @c.cache
        def run():
            result = fn
            while callable(result):
                result = result()
            return result

        return run

    ten_1, ten_2 = _nested_lambdas(10, 1), _nested_lambdas(10, 2)
    # One line, so both lambdas have the same source text.
    a, b = eval(f"(make({ten_1}), make({ten_2}))", {"make": make})
    assert a() == 1
    assert b() == 2


def test_a_sourceless_helper_nested_ten_levels_deep_is_keyed_by_its_innermost_code():
    """Bytecode stood for a helper without source down to eight nested code
    objects; below that its constants were not seen."""
    name = "_cash_test_deep_nested_code"
    c = Cash()
    try:
        module = _fileless_module(
            name,
            f"""
            def helper():
                return ({_nested_lambdas(10, 1)}){"()" * 10}
            """,
        )
        run = c.cache(_fileless_module(name, "def run():\n    return helper()\n").run)
        assert run() == 1
        _fileless_module(
            name,
            f"""
            def helper():
                return ({_nested_lambdas(10, 2)}){"()" * 10}
            """,
        )
        assert module.helper() == 2
        assert run() == 2
    finally:
        sys.modules.pop(name, None)


def test_a_global_read_by_a_class_held_three_attributes_down_is_keyed():
    """A class's methods are keyed with the globals they read, and so are the
    methods of user objects its attributes hold -- two levels down only."""
    name = "_cash_test_nested_class_globals"
    c = Cash()
    try:
        module = _fileless_module(
            name,
            """
            FACTOR = 1

            class Leaf:
                def get(self):
                    return FACTOR

            class Inner:
                deeper = Leaf()

            class Mid:
                inner = Inner()

            class Outer:
                inner = Mid()
            """,
        )

        @c.cache
        def run(cls):
            return cls.inner.inner.deeper.get()

        assert run(module.Outer) == 1
        module.FACTOR = 2
        assert run(module.Outer) == 2
    finally:
        sys.modules.pop(name, None)


def test_a_function_found_in_a_library_object_after_a_large_table_is_keyed():
    """The search through a library object gave up after 2000 values, so a
    user function held after a fitted vocabulary was keyed by name."""
    import argparse

    name = "_cash_test_library_held_code"
    c = Cash()

    @c.cache
    def run(ns):
        return ns.fn()

    vocabulary = {f"word{i}": i for i in range(5000)}
    try:
        module = _fileless_module(name, "def fn():\n    return 1\n")
        assert run(argparse.Namespace(vocabulary=vocabulary, fn=module.fn)) == 1
        module = _fileless_module(name, "def fn():\n    return 2\n")
        assert run(argparse.Namespace(vocabulary=vocabulary, fn=module.fn)) == 2
    finally:
        sys.modules.pop(name, None)


def test_a_callable_ten_containers_deep_in_a_global_is_replaced_by_its_code():
    def fn():
        return 1

    value: object = fn
    for _ in range(10):
        value = {"k": value}
    stabilized = stabilize_for_global_hash(value, lambda f: f"code-of-{f.__name__}")
    for _ in range(10):
        stabilized = stabilized["k"]
    assert stabilized == ("__cash_callable__", "code-of-fn")


def test_a_container_holding_itself_is_stabilized():
    value: list = [len]
    value.append(value)
    stabilized = stabilize_for_global_hash(value, lambda f: f.__name__)
    assert stabilized[0] == ("__cash_callable__", "len")
    assert stabilized[1] == ("__cash_cycle__", "list")


class _Partial(functools.partial):
    """A partial subclass: CPython does not flatten one that has attributes."""


def test_a_chain_of_twelve_partials_is_named_by_what_it_finally_calls():
    def target(*args):
        return args

    chain: object = target
    for i in range(12):
        chain = _Partial(chain, i)
        chain.step = i  # keeps it from being flattened into the next one
    assert isinstance(chain.func, _Partial)
    assert opaque_identity(chain) == f"{target.__module__}.{target.__qualname__}"


DECORATED = """
import functools

def deco(fn):
    @functools.wraps(fn)
    def wrapper(*args):
        return fn(*args)
    return wrapper

@deco
@deco
def core():
    return {value}

class Model:
    @deco
    @deco
    def go(self):
        return {value}
"""


def test_a_decorated_function_or_method_passed_as_an_argument_is_keyed_by_its_body():
    """Code passed as an argument was keyed by its outermost wrapper -- the
    same for everything that decorator wraps -- and one ``__wrapped__`` below
    it, so under two decorators an edit to the body served the old result."""
    name = "_cash_test_decorated_code_args"
    c = Cash()

    @c.cache
    def call(fn):
        return fn()

    @c.cache
    def use(cls):
        return cls().go()

    try:
        module = _fileless_module(name, DECORATED.format(value=1))
        assert (call(module.core), use(module.Model)) == (1, 1)
        module = _fileless_module(name, DECORATED.format(value=2))
        assert (call(module.core), use(module.Model)) == (2, 2)
    finally:
        sys.modules.pop(name, None)
