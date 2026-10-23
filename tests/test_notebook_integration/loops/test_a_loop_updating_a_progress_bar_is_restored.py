"""A loop that updates a progress bar on ``sys.stdout`` is restored.

``pbar = pyprind.ProgBar(n, stream=sys.stdout)`` holds the stream and its
``write``; ``pickle`` refuses the stream, so the loop calling
``pbar.update()`` ran again after every restart. The bar is now stored with
the stream as ``sys.stdout``, and restored onto the stdout of the process
(`live_state`): the count is what plain Python gives, and the bar writes
where a bar made in that process would. A bar on any other file, and
tqdm's, whose own parts hold it, still run, with the reason in the badge.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

BAR = (
    "import sys, time\n"
    "class Bar:\n"
    "    def __init__(self, stream):\n"
    "        self.stream = stream\n"
    "        self.out = stream.write\n"
    "        self.cnt = 0\n"
    "    def update(self):\n"
    "        self.cnt += 1\n"
    "        if self.cnt % 1000 == 0:\n"
    "            self.out('.')\n"
)
LOOP = "for i in range(3000):\n    time.sleep(0.0003)\n    pbar.update()"
STATUS = "[s['status'] for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
REASONS = (
    "[r for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements'] "
    "for r in [s.get('skipped_reason')] if r]"
)


def _printed(nb_runner, cell):
    """What the cell printed to stdout."""
    outputs = nb_runner.get_cell(cell).get("outputs", [])
    return "".join(o.get("text", "") for o in outputs if o.get("name") == "stdout")


def _restart_and_run(nb_runner, peek=STATUS):
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    seen = nb_runner.peek(peek)
    nb_runner.run_cell(3)
    return seen


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("stream", ["sys.stdout", "sys.stderr"])
def test_the_loop_is_restored_and_the_bar_writes_to_the_stream_of_the_process(nb_runner, stream):
    read = "pbar.out('|')\nprint(pbar.cnt, pbar.stream is " + stream + ")"
    nb_runner.create_notebook([BAR + f"pbar = Bar({stream})", LOOP, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip().endswith("3000 True")
    assert "RESTORED" in _restart_and_run(nb_runner)
    out = nb_runner.get_output(3)
    assert out.strip().endswith("3000 True")
    assert "|" in out


@pytest.mark.fresh_kernel
def test_what_the_bar_printed_is_shown_again_on_a_hit(nb_runner):
    """The bar was made in an earlier cell, while that cell's output was
    being recorded; what it prints in the loop's cell is the loop's, as in
    plain Python, and a hit shows it again."""
    nb_runner.create_notebook([BAR + "pbar = Bar(sys.stdout)", LOOP, "print(pbar.cnt)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _printed(nb_runner, 2) == "..."
    assert "RESTORED" in _restart_and_run(nb_runner)
    assert _printed(nb_runner, 2) == "..."


@pytest.mark.fresh_kernel
def test_a_free_lock_in_the_object_is_restored_free(nb_runner):
    make = BAR + "import threading\npbar = Bar(sys.stdout)\npbar.lock = threading.Lock()"
    read = "print(pbar.cnt, pbar.lock.acquire(blocking=False))"
    nb_runner.create_notebook([make, LOOP, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3000 True"
    assert "RESTORED" in _restart_and_run(nb_runner)
    assert nb_runner.get_output(3).strip() == "3000 True"


@pytest.mark.fresh_kernel
def test_a_bar_on_a_file_runs_the_loop(nb_runner):
    make = BAR + "log = open('bar.log', 'w')\npbar = Bar(log)"
    nb_runner.create_notebook([make, LOOP, "print(pbar.cnt)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3000"
    assert "RESTORED" not in _restart_and_run(nb_runner)
    assert nb_runner.get_output(3).strip() == "3000"


@pytest.mark.fresh_kernel
def test_a_tqdm_bar_runs_the_loop_and_says_why(nb_runner):
    make = "import time\nfrom tqdm import tqdm\npbar = tqdm(total=3000)"
    nb_runner.create_notebook([make, LOOP, "print(pbar.n)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3000"
    reasons = _restart_and_run(nb_runner, f"({STATUS}, {REASONS})")
    assert "RESTORED" not in reasons
    assert "pbar" in reasons
    assert nb_runner.get_output(3).strip() == "3000"


@pytest.mark.fresh_kernel
def test_a_loop_over_tqdm_is_restored(nb_runner):
    make = "import time\nfrom tqdm import tqdm\nacc = []"
    loop = "for i in tqdm(range(3000)):\n    time.sleep(0.0003)\n    acc.append(i)"
    nb_runner.create_notebook([make, loop, "print(len(acc), acc[-1])"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "3000 2999"
    assert "RESTORED" in _restart_and_run(nb_runner)
    assert nb_runner.get_output(3).strip() == "3000 2999"
