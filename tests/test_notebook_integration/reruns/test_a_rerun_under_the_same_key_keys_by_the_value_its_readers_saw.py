"""Two copies of a statement in one cell hand each reader its own value.

The digests of a value-keyed run were filed by source and input lineages
alone, so two copies of a statement in one cell shared them:
``start = time.time()`` before and after the work ran the work again for the
reader below, and a repeated ``t = next(G)`` handed its second draw to the
reader of the first.

Calls are counted with ``os.open``/``os.write``: an ``open`` would be a read
the file tracker keys.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

PIN = "import cash\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n%cash_on"


def _count(nb_runner):
    path = nb_runner.work_dir / "calls.log"
    return len(path.read_bytes()) if path.exists() else 0


def _log(nb_runner):
    calls = (nb_runner.work_dir / "calls.log").as_posix()
    return (
        f"    fd = os.open({calls!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
    )


def test_the_work_between_two_timestamps_runs_once(nb_runner):
    nb_runner.create_notebook(
        [
            PIN,
            "import time, os\ndef work(x):\n" + _log(nb_runner) + "    return 1",
            "start = time.time()\nr = work(start)\nstart = time.time()",
            "z = r + 1",
        ]
    )
    nb_runner.start_kernel()
    counts = []
    for _ in range(2):
        nb_runner.run_all()
        counts.append(_count(nb_runner))
    assert nb_runner.peek("z") == "2"
    assert counts[0] == 1, counts
    # After a restart too: the work ran twice there (4 against 2).
    nb_runner.restart()
    nb_runner._init_cash()
    before = _count(nb_runner)
    nb_runner.run_all()
    assert nb_runner.peek("z") == "2"
    assert _count(nb_runner) - before == 1, (before, _count(nb_runner))


def test_a_draw_repeated_in_a_cell_hands_each_reader_its_own(nb_runner):
    (nb_runner.work_dir / "twice_helpers.py").write_text(
        "def gen():\n    i = 0\n    while True:\n        yield i\n        i += 1\nG = gen()\n", encoding="utf-8"
    )
    nb_runner.create_notebook(
        [
            PIN,
            "from twice_helpers import G",
            "t = next(G)\nu = t + 1\nt = next(G)\nv = t * 10",
            "w = (u, v)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("w") == "(1, 10)"


#: Each copy of the draw inside its own block: a statement in a block had
#: no occurrence index of its own, so the copies still shared one digest.
BLOCKS = {
    "try": "try:\n    t = next(G)\nexcept StopIteration:\n    pass\n",
    "if": "if True:\n    t = next(G)\n",
    "with": "with contextlib.nullcontext():\n    t = next(G)\n",
}


@pytest.mark.parametrize("block", list(BLOCKS))
def test_a_draw_repeated_in_two_blocks_hands_each_reader_its_own(nb_runner, block):
    (nb_runner.work_dir / "twice_helpers.py").write_text(
        "def gen():\n    i = 0\n    while True:\n        yield i\n        i += 1\nG = gen()\n", encoding="utf-8"
    )
    draw = BLOCKS[block]
    nb_runner.create_notebook(
        [
            PIN,
            "import contextlib\nfrom twice_helpers import G",
            draw + "u = t + 1\n" + draw + "v = t * 10",
            "w = (u, v)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("w") == "(1, 10)"
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert nb_runner.peek("w") == "(1, 10)"
