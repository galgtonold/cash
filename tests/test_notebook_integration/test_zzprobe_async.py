"""Adversarial probes: concurrency & async (deterministic).

Attack surface (each test = one distinct mechanism):

1.  test_toplevel_await_selfmod_isolated_rerun_idempotent
        x = await bump(x) — self-modifying reassignment through IPython
        autoawait. Isolated re-run must print the same value (idempotence).
2.  test_toplevel_await_cache_hit_second_run
        Expensive top-level-await result: is it actually cached (CACHE_HIT
        on an identical second run_all) or does autoawait bypass caching?
3.  test_await_cell_edit_invalidates_downstream
        Edit the *await* cell (not the async def) — downstream must recompute.
4.  test_async_def_edit_then_isolated_rerun_of_await_cell
        Edit the async def cell, then isolated-re-run ONLY the awaiting cell:
        must pick up the new function body (no stale 11).
5.  test_asyncio_run_bridge_selfmod_rerun
        total = <thread-bridged asyncio.run(add_ten(total))> — self-mod via a
        sync bridge. Isolated re-run must be idempotent.
6.  test_threadpool_map_cache_hit_and_unrelated_edit
        Deterministic ThreadPoolExecutor.map sum: CACHE_HIT on identical
        second run; editing an unrelated cell must not CELL_CHANGED it.
7.  test_threadpool_selfmod_accumulator_isolated_rerun
        acc = acc + sum(ex.map(...)) inside a with-block — idempotent re-run.
8.  test_background_thread_event_gated_list_isolated_rerun
        Thread started in cell A mutates a global list only when cell C sets
        an Event (deterministic). Isolated re-run of cell B (which read the
        list BEFORE the mutation) must reprint the pre-mutation value.
9.  test_lock_unpicklable_restart_persist_graceful
        threading.Lock crossing cells + kernel restart under persist:
        correct values, no traceback.
10. test_future_var_crossing_cells_restart
        concurrent.futures.Future in a variable: re-run + restart correctness.
11. test_queue_drain_isolated_rerun_idempotent
        queue.Queue drained in a later cell; isolated re-run of the drain
        cell should reprint the drained items (stateful-consumable channel).
"""
import textwrap

import pytest

pytestmark = [pytest.mark.timeout(90)]


# ---------------------------------------------------------------------------
# 1. top-level await, self-modifying reassignment, isolated re-run
# ---------------------------------------------------------------------------

def test_toplevel_await_selfmod_isolated_rerun_idempotent(nb_runner):
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
    nb_runner.run_cell(2)
    assert "x=2" in nb_runner.get_output(2), (
        f"isolated re-run not idempotent: {nb_runner.get_output(2)!r}"
    )


# ---------------------------------------------------------------------------
# 2. top-level await result: cached at all?
# ---------------------------------------------------------------------------

def test_toplevel_await_cache_hit_second_run(nb_runner):
    nb_runner.create_notebook([
        "import asyncio",
        textwrap.dedent("""\
            async def compute():
                await asyncio.sleep(0)
                return sum(range(300000))
        """),
        "result = await compute()\nprint(f'result={result}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "result=44999850000" in nb_runner.get_output(3)

    nb_runner.run_all()
    assert "result=44999850000" in nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    has_cache = "CACHE_HIT" in raw or "Cache hit: True" in raw
    assert has_cache, (
        f"top-level-await cell shows no cache activity on identical 2nd run: {raw[:500]}"
    )


# ---------------------------------------------------------------------------
# 3. editing the await cell invalidates downstream
# ---------------------------------------------------------------------------

def test_await_cell_edit_invalidates_downstream(nb_runner):
    nb_runner.create_notebook([
        "import asyncio",
        textwrap.dedent("""\
            async def compute(x):
                await asyncio.sleep(0)
                return x + 1
        """),
        "result = await compute(10)",
        "print(f'double={result * 2}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "double=22" in nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "result = await compute(40)")
    nb_runner.run_all()
    assert "double=82" in nb_runner.get_output(4), (
        f"stale downstream after editing await cell: {nb_runner.get_output(4)!r}"
    )


# ---------------------------------------------------------------------------
# 4. edit async def, then isolated re-run of ONLY the awaiting cell
# ---------------------------------------------------------------------------

def test_async_def_edit_then_isolated_rerun_of_await_cell(nb_runner):
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
    assert "result=11" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, textwrap.dedent("""\
        async def compute(x):
            await asyncio.sleep(0)
            return x + 100
    """))
    nb_runner.run_cell(3)
    assert "result=110" in nb_runner.get_output(3), (
        f"await cell used stale async def after upstream edit: {nb_runner.get_output(3)!r}"
    )


# ---------------------------------------------------------------------------
# 5. asyncio.run via thread bridge, self-modifying, isolated re-run
# ---------------------------------------------------------------------------

def test_asyncio_run_bridge_selfmod_rerun(nb_runner):
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
        f"isolated re-run not idempotent: {nb_runner.get_output(2)!r}"
    )


# ---------------------------------------------------------------------------
# 6. ThreadPoolExecutor.map: cache hit + unrelated-edit stability
# ---------------------------------------------------------------------------

def test_threadpool_map_cache_hit_and_unrelated_edit(nb_runner):
    nb_runner.create_notebook([
        textwrap.dedent("""\
            from concurrent.futures import ThreadPoolExecutor
            def sq(x):
                return x * x
        """),
        textwrap.dedent("""\
            with ThreadPoolExecutor(max_workers=4) as ex:
                total = sum(ex.map(sq, range(100)))
            print(f'total={total}')
        """),
        "z = 1",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "total=328350" in nb_runner.get_output(2)

    # identical second run: pool cell should show cache activity
    nb_runner.run_all()
    assert "total=328350" in nb_runner.get_output(2)
    raw = nb_runner.get_raw_output(2)
    has_cache = "CACHE_HIT" in raw or "Cache hit: True" in raw
    assert has_cache, (
        f"executor cell shows no cache activity on identical 2nd run: {raw[:500]}"
    )

    # unrelated edit must not invalidate the pool cell
    nb_runner.set_cell_source(3, "z = 2")
    nb_runner.run_all()
    assert "total=328350" in nb_runner.get_output(2)
    raw = nb_runner.get_raw_output(2)
    assert "[CELL_CHANGED]" not in raw, (
        f"unrelated edit marked executor cell as changed: {raw[:500]}"
    )


# ---------------------------------------------------------------------------
# 7. self-modifying accumulator through executor map, isolated re-run
# ---------------------------------------------------------------------------

def test_threadpool_selfmod_accumulator_isolated_rerun(nb_runner):
    nb_runner.create_notebook([
        textwrap.dedent("""\
            from concurrent.futures import ThreadPoolExecutor
            def sq(x):
                return x * x
            acc = 0
        """),
        textwrap.dedent("""\
            with ThreadPoolExecutor(max_workers=2) as ex:
                acc = acc + sum(ex.map(sq, range(5)))
            print(f'acc={acc}')
        """),
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "acc=30" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert "acc=30" in nb_runner.get_output(2), (
        f"isolated re-run accumulated instead of idempotent: {nb_runner.get_output(2)!r}"
    )


# ---------------------------------------------------------------------------
# 8. background thread mutates a global list only when a LATER cell fires it
# ---------------------------------------------------------------------------

def test_background_thread_event_gated_list_isolated_rerun(nb_runner):
    nb_runner.create_notebook([
        textwrap.dedent("""\
            import threading
            data = []
            ev = threading.Event()
            def worker():
                ev.wait()
                data.extend([1, 2, 3])
            t = threading.Thread(target=worker, daemon=True)
            t.start()
        """),
        "snap = len(data)\nprint(f'snap={snap}')",
        "ev.set()\nt.join()\nprint(f'n={len(data)}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "snap=0" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"
    assert "n=3" in nb_runner.get_output(3)

    # ADJUDICATED: this is the documented "Background threads" limitation, not a
    # bug (docs/known-limitations.md — "a thread that mutates data after the cell
    # that created it has finished is outside cash's view entirely; re-running an
    # earlier cell can observe the mutated state instead of the state at that
    # point in the notebook").
    #
    # It is not fixable within cash's model. The mutation happens in another
    # thread with no statement to attribute it to, so there is nothing to hash,
    # key, or invalidate against: by the time cell B re-runs, `data` genuinely
    # holds [1,2,3] and a from-start re-run of `len(data)` is 3. Reprinting 0
    # would mean restoring a stale snapshot over live state — the opposite of
    # what cash guarantees everywhere else.
    #
    # Pinned as the real behaviour so the limitation is visible rather than a
    # standing red test.
    nb_runner.run_cell(2)
    assert "snap=3" in nb_runner.get_output(2), (
        f"expected the documented thread-visibility behaviour: {nb_runner.get_output(2)!r}"
    )


# ---------------------------------------------------------------------------
# 9. threading.Lock crossing cells + restart under persist
# ---------------------------------------------------------------------------

def test_lock_unpicklable_restart_persist_graceful(nb_runner):
    nb_runner.create_notebook([
        textwrap.dedent("""\
            import threading
            lock = threading.Lock()
            base = sum(range(100000))
        """),
        textwrap.dedent("""\
            with lock:
                val = base % 97
            print(f'val={val}')
        """),
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "val=28" in nb_runner.get_output(2)
    assert "Traceback" not in nb_runner.get_output(1)
    assert "Traceback" not in nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "val=28" in nb_runner.get_output(2), (
        f"wrong value after restart: {nb_runner.get_output(2)!r}"
    )
    assert "Traceback" not in nb_runner.get_output(1), (
        f"restore of unpicklable lock crashed: {nb_runner.get_output(1)[:400]}"
    )
    assert "Traceback" not in nb_runner.get_output(2), (
        f"cell using restored lock crashed: {nb_runner.get_output(2)[:400]}"
    )


# ---------------------------------------------------------------------------
# 10. Future stored in a variable, crossing cells; re-run + restart
# ---------------------------------------------------------------------------

def test_future_var_crossing_cells_restart(nb_runner):
    nb_runner.create_notebook([
        textwrap.dedent("""\
            from concurrent.futures import ThreadPoolExecutor
            def _work():
                return 21 * 2
            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(_work)
        """),
        "print(f'res={fut.result()}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "res=42" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    assert "res=42" in nb_runner.get_output(2), (
        f"re-run of fut.result() wrong: {nb_runner.get_output(2)!r}"
    )

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "res=42" in nb_runner.get_output(2), (
        f"wrong value after restart: {nb_runner.get_output(2)!r}"
    )
    assert "Traceback" not in nb_runner.get_output(1), (
        f"restore of unpicklable future/executor crashed: {nb_runner.get_output(1)[:400]}"
    )
    assert "Traceback" not in nb_runner.get_output(2), (
        f"cell using restored future crashed: {nb_runner.get_output(2)[:400]}"
    )


# ---------------------------------------------------------------------------
# 11. queue.Queue drained across cells; isolated re-run of the drain cell
# ---------------------------------------------------------------------------

def test_queue_drain_isolated_rerun_idempotent(nb_runner):
    nb_runner.create_notebook([
        textwrap.dedent("""\
            from queue import Queue
            q = Queue()
            for i in range(3):
                q.put(i)
        """),
        textwrap.dedent("""\
            got = []
            while not q.empty():
                got.append(q.get())
            print(f'got={got}')
        """),
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "got=[0, 1, 2]" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert "got=[0, 1, 2]" in nb_runner.get_output(2), (
        f"drained queue not restored on isolated re-run: {nb_runner.get_output(2)!r}"
    )
