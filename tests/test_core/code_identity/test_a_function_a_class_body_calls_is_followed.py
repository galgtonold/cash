"""Editing a function reached from a class through other functions invalidates the call.

A class is followed for its key without its body being audited, and so is
every module function its methods call by bare name. Such a function was
keyed by its own code only: what IT called was never followed, so the chain
stopped at the first function. ``Front.__init__`` calling ``_common_init``,
which calls ``_get_sampler``, which builds ``RWalk``, whose ``run`` calls
``generic_walk`` in another module of the package: an edit to
``generic_walk`` served the result from before it. Found on dynesty:
``DynamicNestedSampler`` -> ``_common_sampler_init`` ->
``_get_internal_sampler`` -> ``RWalkSampler`` -> ``generic_random_walk``.

Checked with the trace off (the key alone must see the edit) and on, with a
lambda the cached function reads from a global dict, through a module
attribute call (from a function and from a method), and for class -> function -> class -> method -> function.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core]

MODULE_A = textwrap.dedent("""
    from pkg import b, c
    from pkg.b import RWalk


    class Base:
        def __init__(self):
            self.ready = True


    class Front(Base):
        def __init__(self):
            super().__init__()
            _common_init(self)


    def _common_init(s):
        s.sampler = _get_sampler()


    def _get_sampler():
        return RWalk()


    class ByAttr:
        def __init__(self):
            self.sampler = _attr_sampler()


    def _attr_sampler():
        return b.Doubler()


    class Outer:
        def __init__(self):
            self.inner = _make_inner()

        def run(self, x):
            return self.inner.step(x)


    def _make_inner():
        return b.Inner()


    class ViaFront:
        def __init__(self):
            self.via = _make_via()


    def _make_via():
        return c.ViaModule()
""")

MODULE_B = textwrap.dedent("""
    class RWalk:
        def run(self, x):
            return generic_walk(x)


    def generic_walk(x):
        return x * MULT


    class Doubler:
        def run(self, x):
            return _walk_twice(x)


    def _walk_twice(x):
        return generic_walk(x) + generic_walk(x)


    class Inner:
        def step(self, x):
            return _inner_step(x)


    def _inner_step(x):
        return generic_walk(x) + 1
""")

MODULE_C = textwrap.dedent("""
    from pkg import b


    class ViaModule:
        def run(self, x):
            return b.generic_walk(x) - 1
""")

DRIVERS = {
    "function_chain": """
        from pkg.a import Front

        @cash.cache
        def go(x):
            f = Front()
            return f.sampler.run(x)
    """,
    "lambda_in_a_global_dict": """
        from pkg.a import Front

        PROBLEMS = {"a": lambda v: v + 1}

        @cash.cache
        def go(x):
            f = Front()
            return f.sampler.run(PROBLEMS["a"](x))
    """,
    "module_attribute_call": """
        from pkg.a import ByAttr

        @cash.cache
        def go(x):
            return ByAttr().sampler.run(x)
    """,
    "method_calls_through_a_module": """
        from pkg.a import ViaFront

        @cash.cache
        def go(x):
            return ViaFront().via.run(x)
    """,
    "class_function_class_method_function": """
        from pkg.a import Outer

        @cash.cache
        def go(x):
            return Outer().run(x)
    """,
}


@pytest.mark.parametrize("trace", ["0", "1"], ids=["trace_off", "trace_on"])
@pytest.mark.parametrize("shape", sorted(DRIVERS))
def test_an_edit_down_a_chain_from_a_class_body_is_seen(tmp_path, shape, trace):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text(MODULE_A)
    (pkg / "c.py").write_text(MODULE_C)
    body = textwrap.dedent(DRIVERS[shape])
    (tmp_path / "main.py").write_text(f"import cash\n{body}\nprint(go(5))\n")
    env = {"CASH_TRACE": trace}
    seen = []
    for mult in (2, 3):
        (pkg / "b.py").write_text(MODULE_B.replace("MULT", str(mult)))
        seen.append(int(run_python("main.py", cwd=tmp_path, env=env).stdout.split()[-1]))
    assert seen == [_plain(shape, mult) for mult in (2, 3)]


def _plain(shape: str, mult: int) -> int:
    """What an uncached run prints with ``generic_walk`` multiplying by *mult*."""
    if shape == "function_chain":
        return 5 * mult
    if shape == "lambda_in_a_global_dict":
        return 6 * mult
    if shape == "module_attribute_call":
        return 10 * mult
    if shape == "method_calls_through_a_module":
        return 5 * mult - 1
    return 5 * mult + 1
