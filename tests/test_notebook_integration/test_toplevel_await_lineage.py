"""Top-level-await (IPython autoawait) cells participate in cash's pipeline.

Regression for CAS-92 stage 1 (correctness only — caching the awaited result
is stage 2 and deliberately out of scope here).

ipykernel dispatches cells containing top-level ``await`` through
``shell.run_cell_async``, NOT the sync ``shell.run_cell`` that cash
monkey-patches.  Before the stage-1 fix those cells bypassed cash entirely:
no upstream reconstruction, no self-modifying-input reset.  Cash now also
wraps ``run_cell_async`` so the same pre-execution handling runs around a
single delegation to the real coroutine.

Covered here:
    (i)   ``x = await bump(x)`` — a self-modifying reassignment must reset to
          its from-start value on an isolated re-run (idempotence), not
          accumulate.
    (ii)  editing an ``async def`` body then isolated-re-running ONLY the
          awaiting cell must pick up the new function body (no stale value).
    (iii) an ``asyncio.run`` sync-bridge control still behaves idempotently.

Note: the cell is executed EXACTLY ONCE (ipykernel drives the coroutine on
its live loop); cash only runs its non-execution pre/post phases around it.
"""
import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def test_toplevel_await_selfmod_isolated_rerun_is_idempotent(nb_runner):
    """(i) ``x = await bump(x)`` resets to its from-start value on re-run."""
    nb_runner.create_notebook([
        textwrap.dedent("""\
            import asyncio
            x = 1
            async def bump(v):
                await asyncio.sleep(0)
                return v + 1
        """),
        "x = await bump(x)\nprint(f'x={x}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "x=2" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"

    # Isolated re-run of only the awaiting cell must be idempotent: cash resets
    # ``x`` to its cell-entry base (re-running the producer) before the cell
    # runs, so it does not accumulate to 3.
    nb_runner.run_cell(2)
    assert "x=2" in nb_runner.get_output(2), (
        f"top-level-await self-mod not idempotent on isolated re-run: "
        f"{nb_runner.get_output(2)!r}"
    )


def test_async_def_edit_then_isolated_rerun_of_await_cell_picks_up_new_body(nb_runner):
    """(ii) editing the async def, then re-running only the awaiting cell,
    must use the new function body (upstream reconstruction)."""
    nb_runner.create_notebook([
        "import asyncio",
        textwrap.dedent("""\
            async def compute(x):
                await asyncio.sleep(0)
                return x + 1
        """),
        "result = await compute(10)\nprint(f'result={result}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "result=11" in nb_runner.get_output(3), f"first run: {nb_runner.get_output(3)!r}"

    nb_runner.set_cell_source(2, textwrap.dedent("""\
        async def compute(x):
            await asyncio.sleep(0)
            return x + 100
    """))
    # Isolated re-run of ONLY the awaiting cell: its upstream check sees the
    # edited ``compute`` source and re-executes the def cell, so the await
    # runs against the new body.
    nb_runner.run_cell(3)
    assert "result=110" in nb_runner.get_output(3), (
        f"await cell used a stale async def after an upstream edit: "
        f"{nb_runner.get_output(3)!r}"
    )


def test_asyncio_run_bridge_selfmod_isolated_rerun_is_idempotent(nb_runner):
    """(iii) sync-bridge control: ``asyncio.run`` via a thread pool, self-mod,
    isolated re-run stays idempotent (this cell has no top-level await, so it
    exercises the sync path and must keep working)."""
    nb_runner.create_notebook([
        textwrap.dedent("""\
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            total = 5
            async def add_ten(v):
                return v + 10
        """),
        textwrap.dedent("""\
            with ThreadPoolExecutor(max_workers=1) as _ex:
                total = _ex.submit(asyncio.run, add_ten(total)).result()
            print(f'total={total}')
        """),
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=15" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"

    nb_runner.run_cell(2)
    assert "total=15" in nb_runner.get_output(2), (
        f"asyncio.run bridge self-mod not idempotent on isolated re-run: "
        f"{nb_runner.get_output(2)!r}"
    )
