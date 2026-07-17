"""CAS-198: a top-level ``await`` inside a control-structure BODY must not
SyntaxError under ``%cash_on``.

``for x in xs: r = await fetch(x)`` -- the canonical async-batch pattern --
reached the sync ``ControlStructureProcessor``, whose unflagged ``compile()``
raises ``SyntaxError: 'await' outside function``. The CAS-164 top-level-await
support had landed on the regular-statement path but not the control-body path.
The fix routes an await-bearing control structure through the
``PyCF_ALLOW_TOP_LEVEL_AWAIT``-capable async statement path as ONE awaited unit.

Two things are pinned here:

  1. ``contains_top_level_await`` -- the routing decision -- classifies every
     shape correctly, INCLUDING the exclusion of an ``await`` nested inside a
     ``def`` / ``async def`` / ``lambda`` (a separate coroutine scope).
  2. ``ControlStructureProcessor.process_await_unit`` actually executes an
     await-loop as one awaited unit against a real event loop -- the flagged
     compile + await mechanism the fix introduces.

The full autoawait ROUTING (ipykernel dispatching the cell to ``run_cell_async``)
only reproduces against a LIVE Jupyter server, which the nbclient unit harness
lacks (CAS-136/190); that end-to-end path is covered by ``scripts/wheel_gate.py``
scenario S6.
"""
import ast
import asyncio
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.control_structures import contains_top_level_await
from cash.notebook.ipython.magics import CashMagics


class MockShell(Configurable):
    """Minimal IPython-compatible shell (mirrors test_single_unit_caching)."""

    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


# --------------------------------------------------------------------------
# (1) the routing decision
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [
    ("for x in xs:\n    r = await f(x)", True),
    ("while c:\n    await g()", True),
    ("if c:\n    y = await h()", True),
    ("with ctx:\n    await k()", True),
    ("for x in xs:\n    async for y in z:\n        pass", True),
    ("with a:\n    async with b:\n        pass", True),
    # await nested inside a def / async def / lambda is a SEPARATE scope:
    ("for x in xs:\n    def worker():\n        return await q()", False),
    ("if c:\n    async def job():\n        await q()", False),
    # plain control structures with no top-level await:
    ("for x in xs:\n    acc.append(x)", False),
    ("while n:\n    n -= 1", False),
    ("with open('f') as fh:\n    data = fh.read()", False),
])
def test_contains_top_level_await(code, expected):
    node = ast.parse(code).body[0]
    assert contains_top_level_await(node) is expected


# --------------------------------------------------------------------------
# (2) the async twin executes the await-loop as one unit
# --------------------------------------------------------------------------

def test_process_await_unit_runs_await_loop(magics_fixture):
    """The fix: an await-bearing for-loop runs via the flag-capable async unit,
    producing correct results -- before the fix it SyntaxError'd at compile."""
    magics, shell, _backend = magics_fixture

    calls = []

    async def fetch(x):
        calls.append(x)
        await asyncio.sleep(0)
        return x * 10

    shell.user_ns.update(fetch=fetch, xs=[1, 2, 3], results=[])

    code = "for x in xs:\n    r = await fetch(x)\n    results.append(r)"
    node = ast.parse(code).body[0]
    assert contains_top_level_await(node)

    result = asyncio.run(
        magics._control_structure_processor.process_await_unit(node, silent=True)
    )

    assert result.success, getattr(result, "error", None)
    assert shell.user_ns["results"] == [10, 20, 30]
    assert calls == [1, 2, 3]


def test_process_await_unit_reports_body_error(magics_fixture):
    """A genuine runtime error inside the awaited body surfaces as a failed
    result (not swallowed) -- the awaited unit still routes errors."""
    magics, shell, _backend = magics_fixture

    async def boom(_x):
        await asyncio.sleep(0)
        raise ValueError("kaboom")

    shell.user_ns.update(boom=boom, xs=[1])

    code = "for x in xs:\n    r = await boom(x)"
    node = ast.parse(code).body[0]

    result = asyncio.run(
        magics._control_structure_processor.process_await_unit(node, silent=True)
    )
    assert not result.success
