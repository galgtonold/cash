"""Integration tests for caching top-level-await (autoawait) cell results.

CAS-116 (CAS-92 stage 2): a cell that runs ``x = await slow()`` through
IPython's autoawait path must CACHE the awaited result, so an identical second
run is a cache HIT that skips the await entirely — not a recompute.

Stage 1 (CAS-92) only restored *correctness* (upstream reconstruction +
self-modifying-reassignment reset) for these cells while still recomputing on
every run.  These tests pin the stage-2 behaviour:

1. ``test_toplevel_await_result_is_cached_second_run`` — a side-effect counter
   that lives OUTSIDE the awaiting cell increments on run 1 (COMPUTE) but NOT
   on an identical run 2 (cache HIT skips the await).  The debug markers must
   also show cache activity on the second run.
2. ``test_toplevel_await_selfmod_stays_idempotent`` — a self-modifying await
   cell (``x = await bump(x)``) re-run in isolation must reprint its from-start
   value, guarding against a stage-1 regression once caching is in play.
"""
import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def test_toplevel_await_result_is_cached_second_run(nb_runner):
    """An expensive ``x = await slow()`` cell COMPUTES on run 1 and is
    RESTORED (await skipped) on an identical run 2.

    Detection is by a side-effect counter (``call_log``) defined in an upstream
    cell: ``slow()`` appends to it every time the coroutine actually runs.  A
    cache HIT restores ``result`` without calling ``slow()``, so the counter
    stays at 1.  We also assert the debug markers show cache activity on run 2.
    """
    nb_runner.create_notebook([
        textwrap.dedent("""\
            import asyncio
            call_log = []
            async def slow():
                await asyncio.sleep(0)
                call_log.append('ran')
                return sum(range(300000))
        """),
        "result = await slow()\nprint(f'result={result}')",
        "print(f'calls={len(call_log)}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()

    # Run 1: the coroutine runs once (COMPUTE).
    nb_runner.run_all()
    assert "result=44999850000" in nb_runner.get_output(2), (
        f"run 1 wrong result: {nb_runner.get_output(2)!r}"
    )
    assert "calls=1" in nb_runner.get_output(3), (
        f"coroutine did not run exactly once on first run: {nb_runner.get_output(3)!r}"
    )

    # Run 2 (identical): a cache HIT restores ``result`` and MUST NOT re-run the
    # coroutine, so the side-effect counter stays at 1.
    nb_runner.run_all()
    assert "result=44999850000" in nb_runner.get_output(2), (
        f"run 2 wrong result: {nb_runner.get_output(2)!r}"
    )
    assert "calls=1" in nb_runner.get_output(3), (
        f"await cell recomputed on identical 2nd run (counter grew): "
        f"{nb_runner.get_output(3)!r}"
    )

    raw = nb_runner.get_raw_output(2)
    has_cache = "CACHE_HIT" in raw or "Cache hit: True" in raw
    assert has_cache, (
        f"top-level-await cell shows no cache activity on identical 2nd run: {raw[:500]}"
    )


def test_toplevel_await_selfmod_stays_idempotent(nb_runner):
    """A self-modifying await cell (``x = await bump(x)``) re-run in isolation
    reprints its from-start value — caching must not turn an isolated re-run
    into an accumulation (stage-1 regression guard)."""
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
    nb_runner.enable_persist()

    nb_runner.run_all()
    assert "x=2" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"

    # Isolated re-run of only the self-modifying await cell must be idempotent.
    nb_runner.run_cell(2)
    assert "x=2" in nb_runner.get_output(2), (
        f"self-mod await cell accumulated instead of idempotent: "
        f"{nb_runner.get_output(2)!r}"
    )
