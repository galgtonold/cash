"""Consumable / producer-re-execution engine: isolated re-run of a consumer cell.

A *consumable* input (generator, ``queue.Queue``, open file handle) is drained
IN PLACE and cannot be faithfully snapshot-restored — the cache store falls back
to keeping a reference, so "restoring" it hands back the already-drained object.
On an ISOLATED re-run of the consumer cell the producer does not re-run, so the
cell reads leftovers from its own previous run: ``got=[]`` instead of
``got=[0, 1, 2]``, ``total=0`` instead of ``total=55``.

``run_all`` is already correct for both because the producer cell re-runs first.
The fix makes an isolated re-run do the same, so the oracle throughout is:
**an isolated re-run must agree with run_all.**

The guard matrix (i–vi below) pins both directions: the bug is fixed AND a
consumable that is merely *inspected* rather than drained does not drag its
producer along for the ride.
"""

import textwrap

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.timeout(90)]


# ---------------------------------------------------------------------------
# (i) drained queue genuinely consumed -> isolated re-run re-runs the producer
# ---------------------------------------------------------------------------


def test_drained_queue_consumed_isolated_rerun(nb_runner):
    """The drain cell re-run alone must refill from the producer."""
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
    assert "got=[0, 1, 2]" in nb_runner.get_output(2), f"isolated re-run of the drain cell: {nb_runner.get_output(2)!r}"
    # Stable across repeated isolated re-runs (the base must be re-recorded
    # after each producer re-execution, not left at the drained state).
    nb_runner.run_cell(2)
    assert "got=[0, 1, 2]" in nb_runner.get_output(2), f"second isolated re-run: {nb_runner.get_output(2)!r}"


# ---------------------------------------------------------------------------
# (ii) exhausted STORED generator consumed -> isolated re-run
# ---------------------------------------------------------------------------


def test_exhausted_generator_consumed_isolated_rerun(nb_runner):
    """A generator stored in a var (not an inline genexpr, which is
    re-evaluated fresh and already worked) must be re-produced."""
    nb_runner.create_notebook(
        [
            "g = (i * i for i in range(6))",
            "total = 0\nfor x in g:\n    total += x\nprint(f'total={total}')",
            "print(f'seen={total}')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total=55" in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"

    nb_runner.run_cell(2)
    assert "total=55" in nb_runner.get_output(2), (
        f"isolated re-run of the generator consumer: {nb_runner.get_output(2)!r}"
    )
    nb_runner.run_cell(3)
    assert "seen=55" in nb_runner.get_output(3), f"downstream reader after re-run: {nb_runner.get_output(3)!r}"


# ---------------------------------------------------------------------------
# (iii) NO OVER-INVALIDATION: consumable present but NOT consumed
# ---------------------------------------------------------------------------
#
# The probes must self-disable: an unchanged qsize and a GEN_CREATED generator
# are evidence that nothing was drawn, so the producer must be left alone.


def test_inspected_but_not_consumed_queue_not_reexecuted(nb_runner):
    """``q.qsize()`` reports on the queue without drawing from it."""
    nb_runner.create_notebook(
        [
            textwrap.dedent("""\
            from queue import Queue
            import itertools
            _counter = itertools.count()
            q = Queue()
            for i in range(3):
                q.put(i)
            produced = next(_counter)
            print(f'produced={produced}')
        """),
            "n = q.qsize()\nprint(f'n={n}')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "n=3" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "produced=0" in nb_runner.get_output(1), nb_runner.get_output(1)

    nb_runner.run_cell(2)
    assert "n=3" in nb_runner.get_output(2), f"inspecting cell re-run: {nb_runner.get_output(2)!r}"
    # The producer must NOT have been re-executed: the side-effecting counter in
    # cell 1 would have advanced to produced=1 if it had.
    assert "produced=0" in nb_runner.get_output(1), (
        f"over-invalidation: producer re-ran for a non-consuming read: {nb_runner.get_output(1)!r}"
    )


def test_inspected_but_not_consumed_generator_not_reexecuted(nb_runner):
    """``type(g)`` leaves the generator at GEN_CREATED -> nothing to restore."""
    nb_runner.create_notebook(
        [
            textwrap.dedent("""\
            import itertools
            _counter = itertools.count()
            g = (i * i for i in range(6))
            produced = next(_counter)
            print(f'produced={produced}')
        """),
            "print(type(g).__name__)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "generator" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "produced=0" in nb_runner.get_output(1), nb_runner.get_output(1)

    nb_runner.run_cell(2)
    assert "generator" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "produced=0" in nb_runner.get_output(1), (
        f"over-invalidation: producer re-ran though the generator was untouched: {nb_runner.get_output(1)!r}"
    )


# ---------------------------------------------------------------------------
# (iv) cross-cell-built queue: the WHOLE producer chain must replay
# ---------------------------------------------------------------------------


def test_cross_cell_built_queue_full_chain_replays(nb_runner):
    """A: create + put(0); B: put(1); C: drain. The isolated re-run of C must
    replay A *and* B -> [0, 1], not just A -> [0]."""
    nb_runner.create_notebook(
        [
            "from queue import Queue\nq = Queue()\nq.put(0)",
            "q.put(1)",
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
    assert "got=[0, 1]" in nb_runner.get_output(3), f"first run: {nb_runner.get_output(3)!r}"

    nb_runner.run_cell(3)
    assert "got=[0, 1]" in nb_runner.get_output(3), (
        f"isolated re-run must replay the whole producer chain: {nb_runner.get_output(3)!r}"
    )


# ---------------------------------------------------------------------------
# (v) a slow CACHED consumer stays cache-rescued
# ---------------------------------------------------------------------------


def test_slow_cached_consumer_still_rescued(nb_runner):
    """A consumer whose statement is a cache hit must keep being restored from
    cache rather than dragged through a producer re-execution."""
    nb_runner.create_notebook(
        [
            "import time\nvals = list(range(200))",
            textwrap.dedent("""\
            import time
            time.sleep(0.05)
            agg = sum(vals)
            print(f'agg={agg}')
        """),
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "agg=19900" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.run_cell(2)
    assert "agg=19900" in nb_runner.get_output(2), f"cached consumer re-run: {nb_runner.get_output(2)!r}"


# ---------------------------------------------------------------------------
# (vi) run_all stays correct (producer re-runs first -> probe reports fresh)
# ---------------------------------------------------------------------------


def test_run_all_unchanged_for_queue_and_generator(nb_runner):
    nb_runner.create_notebook(
        [
            textwrap.dedent("""\
            from queue import Queue
            q = Queue()
            for i in range(3):
                q.put(i)
            g = (i * i for i in range(6))
        """),
            textwrap.dedent("""\
            got = []
            while not q.empty():
                got.append(q.get())
            total = sum(g)
            print(f'got={got} total={total}')
        """),
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "got=[0, 1, 2] total=55" in nb_runner.get_output(2), nb_runner.get_output(2)

    nb_runner.run_all()
    assert "got=[0, 1, 2] total=55" in nb_runner.get_output(2), f"second run_all diverged: {nb_runner.get_output(2)!r}"
    nb_runner.run_all()
    assert "got=[0, 1, 2] total=55" in nb_runner.get_output(2), f"third run_all diverged: {nb_runner.get_output(2)!r}"


# ---------------------------------------------------------------------------
# Restorable look-alikes must NOT be classified (over-invalidation guard)
# ---------------------------------------------------------------------------


def test_deepcopyable_iterators_left_alone(nb_runner):
    """``map``/``zip``/``filter``/``enumerate``/``iter(list)`` are self-iterators
    that drain in place too, but they ARE deep-copyable, so the store snapshots
    them fresh at set time and they restore correctly. Classifying them would
    re-execute their producers for nothing."""
    nb_runner.create_notebook(
        [
            textwrap.dedent("""\
            import itertools
            _c = itertools.count()
            m = map(str.upper, ['a', 'b'])
            z = zip([1, 2], 'ab')
            it = iter([7, 8, 9])
            produced = next(_c)
            print(f'produced={produced}')
        """),
            "print(list(m), list(z), list(it))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "['A', 'B']" in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "produced=0" in nb_runner.get_output(1), nb_runner.get_output(1)

    nb_runner.run_cell(2)
    # These restore correctly on their own, so the producer must stay put.
    assert "produced=0" in nb_runner.get_output(1), (
        f"over-invalidation: producer re-ran for deep-copyable iterators: {nb_runner.get_output(1)!r}"
    )


# ---------------------------------------------------------------------------
# (vii) several cells read one handle in turn: a forward run, not a re-run
# ---------------------------------------------------------------------------
#
# Each reader cell is compared with its OWN previous entry. Comparing cell C
# with the position cell B started from took C for a re-run of B on the very
# first Run All: the file was re-opened and C read the header again.


def test_cells_reading_one_file_in_turn_continue_where_the_last_stopped(nb_runner):
    (nb_runner.work_dir / "data.csv").write_text("name,score\nann,1\nbob,2\n", encoding="utf-8")
    nb_runner.create_notebook(
        [
            "fh = open('data.csv')",
            "header = fh.readline()",
            "rows = [line.strip() for line in fh]",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("header") == repr("name,score\n")
    assert nb_runner.peek("rows") == repr(["ann,1", "bob,2"])

    nb_runner.run_all()
    assert nb_runner.peek("rows") == repr(["ann,1", "bob,2"]), "second Run All"

    # An isolated re-run of the last reader still re-opens the file and replays
    # the header read, as Run All would.
    nb_runner.run_cell(3)
    assert nb_runner.peek("rows") == repr(["ann,1", "bob,2"]), "isolated re-run"


def test_cells_drawing_from_one_generator_through_a_slow_call_get_successive_items(nb_runner):
    nb_runner.create_notebook(
        [
            textwrap.dedent(f"""\
            import time
            def gen():
                yield from range(10)
            def work(x):
                time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
                return x
            g = gen()
        """),
            "a = work(next(g))",
            "b = work(next(g))",
            "c = work(next(g))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("(a, b, c)") == repr((0, 1, 2))


# ---------------------------------------------------------------------------
# (viii) a handle open for writing: its producer never re-runs
# ---------------------------------------------------------------------------
#
# Re-opening an append-mode log and replaying the writes above repeats side
# effects Run All performs once. Re-running a writer cell alone writes again,
# as in plain Jupyter, and nothing more.


@pytest.mark.parametrize(
    "cells, expected",
    [
        (
            [
                "log = open('run.log', 'a')",
                "print('start', file=log)",
                "print('end', file=log)",
                "log.close()\ntxt = open('run.log').read()",
            ],
            "start\nend\nend\n",
        ),
        (
            [
                "import gzip\nz = gzip.open('out.txt.gz', 'wt')",
                "z.write('a\\n')",
                "z.write('b\\n')",
                "z.close()\ntxt = gzip.open('out.txt.gz', 'rt').read()",
            ],
            "a\nb\nb\n",
        ),
    ],
    ids=["append-mode log", "gzip writer"],
)
def test_re_running_a_writer_cell_does_not_reopen_its_file(nb_runner, cells, expected):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert nb_runner.peek("txt") == repr(expected)
