"""The code a class with no source forwards to is keyed.

xclim builds each indicator class with ``type(...)`` and sets its
``compute`` to a function ``boltons.funcutils.wraps`` compiled: a stub with
no source file whose globals are a private dict holding the decorator's
wrapper (``_call``), and whose ``__wrapped__`` skips that wrapper. A class
with no source was keyed by its ``__call__`` alone, and the stub by its
bytecode, the same for every function it wraps: an edit to the function an
indicator computes with, or to the decorator's wrapper, served the result
computed before it. The class's own functions are keyed now, and a stub by
the functions it calls through its private globals.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(300)]

OPS = textwrap.dedent("""
    def forward(func):
        def wrapper(x):
            return func(x) + {offset}

        stub_globals = {{"_call": wrapper}}
        exec(compile("def compute(x):\\n    return _call(x)\\n", "<stub>", "exec"), stub_globals)
        stub = stub_globals["compute"]
        stub.__wrapped__ = func
        stub.__module__ = func.__module__
        return stub


    def double(x):
        return x * {factor}
""")

TOOLS = textwrap.dedent("""
    import ops


    class Base:
        def __call__(self, x):
            return self.compute(x)


    Tool = type("Tool", (Base,), {"compute": staticmethod(ops.forward(ops.double))})
    TOOL = Tool()
""")

MAIN = textwrap.dedent("""
    import cash
    import tools

    c = cash.Cash(cache_dir=".cash")


    @c.cache
    def run(x):
        return tools.TOOL(x)


    print(run(5), run.cache_info()["hits"])
""")


@pytest.mark.parametrize("trace", ["1", "0"])
def test_an_edit_behind_the_stub_is_not_served(tmp_path, trace):
    (tmp_path / "tools.py").write_text(TOOLS, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")

    def run(offset, factor):
        (tmp_path / "ops.py").write_text(OPS.format(offset=offset, factor=factor), encoding="utf-8")
        proc = run_python("main.py", cwd=tmp_path, env={"CASH_TRACE": trace})
        return tuple(map(int, proc.stdout.split()))

    assert run(1, 2) == (11, 0)
    assert run(1, 2) == (11, 1)
    # The wrapper the stub calls through its private globals.
    assert run(7, 2) == (17, 0)
    # The function the stub's ``__wrapped__`` names.
    assert run(7, 3) == (22, 0)
