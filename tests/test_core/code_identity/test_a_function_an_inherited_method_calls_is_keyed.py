"""Editing a module function that an inherited method calls invalidates the call.

``class Child(Base)`` with only an ``__init__`` of its own, and ``Base.run``
calling the module function ``_configure`` by bare name: the walk queued
``Child`` for its own source and never its bases, so ``_configure`` never
became a helper. The class digest holds ``Base.run``'s code, so an edit to
``run`` itself was seen, but an edit to ``_configure`` served the result from
before it. Found on dynesty: ``DynamicNestedSampler(DynamicSampler)`` running
``sample_batch``, which calls ``_configure_batch_sampler`` of its module.

Checked with the trace off (the key alone must see the edit) and on, for a
direct base, a lambda the cached function reads from a global dict, and a
base two levels up beside a mixin.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core]

MODULE = textwrap.dedent("""
    def _configure(x):
        return x * MULT


    class Base:
        def run(self, v):
            return _configure(v)


    class Child(Base):
        def __init__(self):
            super().__init__()


    class Mixin:
        def mixed(self, v):
            return _configure(v) + 1000


    class Mid(Base):
        pass


    class Deep(Mixin, Mid):
        def __init__(self):
            super().__init__()
""")

DRIVERS = {
    "direct_base": """
        from pkg.mod import Child

        @cash.cache
        def go(v):
            return Child().run(v)
    """,
    "lambda_in_a_global_dict": """
        from pkg.mod import Child

        PROBLEMS = {"a": lambda v: v + 1}

        @cash.cache
        def go(v):
            return Child().run(PROBLEMS["a"](v))
    """,
    "two_levels_and_a_mixin": """
        from pkg.mod import Deep

        @cash.cache
        def go(v):
            d = Deep()
            return d.run(v) + d.mixed(v)
    """,
}


@pytest.mark.parametrize("trace", ["0", "1"], ids=["trace_off", "trace_on"])
@pytest.mark.parametrize("shape", sorted(DRIVERS))
def test_an_edit_to_a_function_a_base_method_calls_is_seen(tmp_path, shape, trace):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    body = textwrap.dedent(DRIVERS[shape])
    (tmp_path / "main.py").write_text(f"import cash\n{body}\nprint(go(5))\n")
    env = {"CASH_TRACE": trace}
    seen = []
    for mult in (2, 3):
        (pkg / "mod.py").write_text(MODULE.replace("MULT", str(mult)))
        seen.append(int(run_python("main.py", cwd=tmp_path, env=env).stdout.split()[-1]))
    assert seen == [_plain(shape, mult) for mult in (2, 3)]


def _plain(shape: str, mult: int) -> int:
    """What an uncached run prints with ``_configure`` multiplying by *mult*."""
    if shape == "direct_base":
        return 5 * mult
    if shape == "lambda_in_a_global_dict":
        return 6 * mult
    return 5 * mult + 5 * mult + 1000
