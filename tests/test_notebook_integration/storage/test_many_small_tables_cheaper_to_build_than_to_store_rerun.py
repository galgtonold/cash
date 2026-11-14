"""A loop building many small tables, cheaper to run than to store, runs again.

A loop appending 6,000 three-row frames to a list in 1.2 s had them copied
into the RAM tier and pickled for disk, 2.3 s in all, the pickling holding
the interpreter lock while the next statement ran. The store is priced
before anything is copied (`many_frames_cost`) and refused when it costs
more than a hit of the loop saves: the loop runs again, with the reason in its badge, and
what it built is the same. Under ``@cash:persist``, the control, it is
stored and restored however dear its store.

The sample is measured in the kernel as the store runs; its price is pinned
there (60 s), so a loaded machine, which slows the loop and the sample
unevenly, cannot move the verdict. A fresh kernel: the pin is the kernel's.
"""

import ast

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

SETUP = "import pandas as pd\nt = pd.DataFrame({'user': ['s10'] * 3, 'n': [1, 2, 3]})"
LOOP = "frames = []\nfor i in range(1500):{}\n    frames.append(t.copy())"
READ = "print(len(frames), sum(int(f['n'].sum()) for f in frames))"
PIN = """
import cash.notebook.statement.store as _store
from cash.backends import sampled_cost as _sampled
_store._test_found = []
def _pinned(value):
    measured = _sampled.many_frames_cost(value)
    if measured is None:
        return None
    _store._test_found.append(measured.items)
    return measured._replace(copy=30.0, dump=30.0)
_store.many_frames_cost = _pinned
"""
FOUND = "__import__('cash.notebook.statement.store', fromlist=['_'])._test_found"
METRICS = "get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']"
ROWS = f"[(s['code'][:14], str(s['status']), s.get('skipped_reason') or '') for s in {METRICS}]"


def _loop(nb_runner):
    """The last cell's statements, and the loop's ``(status, reason)``."""
    rows = ast.literal_eval(nb_runner.peek(ROWS))
    loop = [(status, reason) for code, status, reason in rows if code.startswith("for i in")]
    assert len(loop) == 1, rows
    return rows, loop[0]


@pytest.mark.parametrize(
    ("annotation", "stored"), [("", False), ("  # @cash:persist", True)], ids=["reruns", "persist"]
)
def test_the_loop_is_stored_only_when_storing_is_cheaper(nb_runner, annotation, stored):
    nb_runner.create_notebook([SETUP, LOOP.format(annotation), READ])
    nb_runner.start_kernel()
    nb_runner.peek(f"exec({PIN!r}, {{}})")
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    rows, (_status, reason) = _loop(nb_runner)
    assert ("tables would take" in reason) is not stored, rows
    if not stored:
        assert nb_runner.peek(FOUND) == "[1500]", rows
    nb_runner.run_cell(3)
    assert "1500 9000" in nb_runner.get_output(3)

    nb_runner.peek("frames.clear()")
    nb_runner.run_cell(2)
    rows, (status, _reason) = _loop(nb_runner)
    assert ("RESTORED" in status) is stored, rows
    nb_runner.run_cell(3)
    assert "1500 9000" in nb_runner.get_output(3)
