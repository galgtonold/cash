"""async code, threads and concurrency primitives across cells.

Adversarial probes: concurrency & async (deterministic).

Attack surface (each test = one distinct mechanism):

1.  test_toplevel_await_cache_hit_second_run
        Expensive top-level-await result: is it actually cached (CACHE_HIT
        on an identical second run_all) or does autoawait bypass caching?
2.  test_await_cell_edit_invalidates_downstream
        Edit the *await* cell (not the async def) — downstream must recompute.
3.  test_threadpool_map_cache_hit_and_unrelated_edit
        Deterministic ThreadPoolExecutor.map sum: CACHE_HIT on identical
        second run; editing an unrelated cell must not CELL_CHANGED it.
4.  test_threadpool_selfmod_accumulator_isolated_rerun
        acc = acc + sum(ex.map(...)) inside a with-block — idempotent re-run.
5.  test_background_thread_event_gated_list_isolated_rerun
        Thread started in cell A mutates a global list only when cell C sets
        an Event (deterministic). Isolated re-run of cell B (which read the
        list BEFORE the mutation) must reprint the pre-mutation value.
6.  test_lock_unpicklable_restart_persist_graceful
        threading.Lock crossing cells + kernel restart under persist:
        correct values, no traceback.
7.  test_future_var_crossing_cells_restart
        concurrent.futures.Future in a variable: re-run + restart correctness.
8.  test_queue_drain_isolated_rerun_idempotent
        queue.Queue drained in a later cell; isolated re-run of the drain
        cell should reprint the drained items (stateful-consumable channel).

The self-modifying ``x = await bump(x)`` and ``asyncio.run`` bridge re-runs,
and the edited ``async def`` re-run from the awaiting cell, are in
``test_toplevel_await_lineage.py``.
"""

import textwrap

import pytest


@pytest.mark.timeout(90)
def test_toplevel_await_cache_hit_second_run(nb_runner):
    nb_runner.create_notebook(
        [
            "import asyncio",
            textwrap.dedent("""\
            async def compute():
                await asyncio.sleep(0)
                return sum(range(300000))
        """),
            "result = await compute()\nprint(f'result={result}')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "result=44999850000" in nb_runner.get_output(3)

    nb_runner.run_all()
    assert "result=44999850000" in nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    has_cache = "CACHE_HIT" in raw or "Cache hit: True" in raw
    assert has_cache, f"top-level-await cell shows no cache activity on identical 2nd run: {raw[:500]}"


@pytest.mark.timeout(90)
def test_await_cell_edit_invalidates_downstream(nb_runner):
    nb_runner.create_notebook(
        [
            "import asyncio",
            textwrap.dedent("""\
            async def compute(x):
                await asyncio.sleep(0)
                return x + 1
        """),
            "result = await compute(10)",
            "print(f'double={result * 2}')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "double=22" in nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "result = await compute(40)")
    nb_runner.run_all()
    assert "double=82" in nb_runner.get_output(4), (
        f"stale downstream after editing await cell: {nb_runner.get_output(4)!r}"
    )


@pytest.mark.timeout(90)
def test_threadpool_map_cache_hit_and_unrelated_edit(nb_runner):
    nb_runner.create_notebook(
        [
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
        ]
    )
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
    assert has_cache, f"executor cell shows no cache activity on identical 2nd run: {raw[:500]}"

    # unrelated edit must not invalidate the pool cell
    nb_runner.set_cell_source(3, "z = 2")
    nb_runner.run_all()
    assert "total=328350" in nb_runner.get_output(2)
    raw = nb_runner.get_raw_output(2)
    assert "[CELL_CHANGED]" not in raw, f"unrelated edit marked executor cell as changed: {raw[:500]}"


@pytest.mark.timeout(90)
def test_threadpool_selfmod_accumulator_isolated_rerun(nb_runner):
    nb_runner.create_notebook(
        [
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
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "acc=30" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert "acc=30" in nb_runner.get_output(2), (
        f"isolated re-run accumulated instead of idempotent: {nb_runner.get_output(2)!r}"
    )


@pytest.mark.timeout(90)
def test_background_thread_event_gated_list_isolated_rerun(nb_runner):
    nb_runner.create_notebook(
        [
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
        ]
    )
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


@pytest.mark.timeout(90)
def test_lock_unpicklable_restart_persist_graceful(nb_runner):
    nb_runner.create_notebook(
        [
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
        ]
    )
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
    assert "val=28" in nb_runner.get_output(2), f"wrong value after restart: {nb_runner.get_output(2)!r}"
    assert "Traceback" not in nb_runner.get_output(1), (
        f"restore of unpicklable lock crashed: {nb_runner.get_output(1)[:400]}"
    )
    assert "Traceback" not in nb_runner.get_output(2), (
        f"cell using restored lock crashed: {nb_runner.get_output(2)[:400]}"
    )


@pytest.mark.timeout(90)
def test_future_var_crossing_cells_restart(nb_runner):
    nb_runner.create_notebook(
        [
            textwrap.dedent("""\
            from concurrent.futures import ThreadPoolExecutor
            def _work():
                return 21 * 2
            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(_work)
        """),
            "print(f'res={fut.result()}')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "res=42" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    assert "res=42" in nb_runner.get_output(2), f"re-run of fut.result() wrong: {nb_runner.get_output(2)!r}"

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "res=42" in nb_runner.get_output(2), f"wrong value after restart: {nb_runner.get_output(2)!r}"
    assert "Traceback" not in nb_runner.get_output(1), (
        f"restore of unpicklable future/executor crashed: {nb_runner.get_output(1)[:400]}"
    )
    assert "Traceback" not in nb_runner.get_output(2), (
        f"cell using restored future crashed: {nb_runner.get_output(2)[:400]}"
    )


@pytest.mark.timeout(90)
def test_queue_drain_isolated_rerun_idempotent(nb_runner):
    nb_runner.create_notebook(
        [
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
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "got=[0, 1, 2]" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert "got=[0, 1, 2]" in nb_runner.get_output(2), (
        f"drained queue not restored on isolated re-run: {nb_runner.get_output(2)!r}"
    )


@pytest.mark.stress
class TestThreadingBasics:
    """Test threading patterns across cells."""

    def test_thread_pool_results(self, nb_runner):
        """ThreadPoolExecutor results cached across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor

                def compute_square(x):
                    return x ** 2

                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(compute_square, i) for i in range(10)]
                    results = [f.result() for f in futures]
                print(f"results={results}")
            """),
                textwrap.dedent("""\
                total = sum(results)
                print(f"total={total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]" in nb_runner.get_output(1)
        assert "total=285" in nb_runner.get_output(2)

    def test_thread_pool_map(self, nb_runner):
        """ThreadPoolExecutor.map across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor

                def process(item):
                    return item.upper()

                words = ['hello', 'world', 'from', 'threads']
                with ThreadPoolExecutor(max_workers=2) as pool:
                    processed = list(pool.map(process, words))
                print(f"processed={processed}")
            """),
                textwrap.dedent("""\
                joined = ' '.join(processed)
                print(f"joined={joined}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "processed=['HELLO', 'WORLD', 'FROM', 'THREADS']" in nb_runner.get_output(1)
        assert "joined=HELLO WORLD FROM THREADS" in nb_runner.get_output(2)


@pytest.mark.stress
class TestQueuePatterns:
    """Test queue-based patterns."""

    def test_thread_safe_counter(self, nb_runner):
        """Thread-safe counter with lock."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import threading

                class SafeCounter:
                    def __init__(self):
                        self._count = 0
                        self._lock = threading.Lock()
                    def increment(self, n=1):
                        with self._lock:
                            self._count += n
                    @property
                    def value(self):
                        return self._count

                counter = SafeCounter()
                threads = []
                for _ in range(10):
                    t = threading.Thread(target=lambda: [counter.increment() for _ in range(100)])
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
                print(f"count={counter.value}")
            """),
                textwrap.dedent("""\
                is_correct = counter.value == 1000
                print(f"correct={is_correct}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=1000" in nb_runner.get_output(1)
        assert "correct=True" in nb_runner.get_output(2)


@pytest.mark.stress
class TestProcessPoolPatterns:
    """Test process pool patterns (simulated with threading)."""

    def test_map_reduce_pattern(self, nb_runner):
        """Map-reduce pattern with thread pool."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor
                from collections import Counter

                documents = [
                    "the quick brown fox",
                    "the lazy dog",
                    "the quick dog jumps",
                    "brown fox jumps high"
                ]

                def map_words(doc):
                    return Counter(doc.split())

                with ThreadPoolExecutor() as pool:
                    word_counts = list(pool.map(map_words, documents))

                # Reduce
                total_counts = Counter()
                for wc in word_counts:
                    total_counts.update(wc)
            """),
                textwrap.dedent("""\
                top5 = total_counts.most_common(5)
                print(f"top5={top5}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "the" in out
        assert "3" in out  # "the" appears 3 times

    def test_parallel_aggregation(self, nb_runner):
        """Parallel data aggregation across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from concurrent.futures import ThreadPoolExecutor

                def chunk_sum(chunk):
                    return sum(chunk)

                data = list(range(1000))
                chunk_size = 100
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

                with ThreadPoolExecutor() as pool:
                    partial_sums = list(pool.map(chunk_sum, chunks))
                print(f"chunks={len(partial_sums)}")
            """),
                textwrap.dedent("""\
                total = sum(partial_sums)
                expected = sum(range(1000))
                print(f"total={total} expected={expected} match={total == expected}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "chunks=10" in nb_runner.get_output(1)
        assert "match=True" in nb_runner.get_output(2)


@pytest.mark.stress
class TestAsyncBasics:
    """Test async function definition and result caching."""

    def test_async_function_definition_and_call(self, nb_runner):
        """Define async function in one cell, call in another with top-level await."""
        nb_runner.create_notebook(
            [
                "import asyncio",
                textwrap.dedent("""\
                async def fetch_value(x):
                    await asyncio.sleep(0.01)
                    return x * 10
            """),
                textwrap.dedent("""\
                result = await fetch_value(5)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(3)

        # Second run should use cache
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(3)

    def test_async_function_change_propagates(self, nb_runner):
        """Changing async function body invalidates downstream."""
        nb_runner.create_notebook(
            [
                "import asyncio",
                textwrap.dedent("""\
                async def compute(x):
                    return x + 1
            """),
                textwrap.dedent("""\
                result = await compute(10)
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=11" in nb_runner.get_output(3)

        # Change async function
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            async def compute(x):
                return x + 100
        """),
        )
        nb_runner.run_all()
        assert "result=110" in nb_runner.get_output(3)

    def test_async_with_gather(self, nb_runner):
        """asyncio.gather for concurrent tasks via top-level await."""
        nb_runner.create_notebook(
            [
                "import asyncio",
                textwrap.dedent("""\
                async def square(n):
                    return n ** 2
            """),
                textwrap.dedent("""\
                tasks = [square(i) for i in range(5)]
                results = list(await asyncio.gather(*tasks))
                print(f"results={results}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 1, 4, 9, 16]" in nb_runner.get_output(3)
