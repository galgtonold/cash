"""A loop building many small tables, cheaper to run than to store, runs again.

A loop appending 6,000 three-row frames to a list in 1.2 s had them copied
into the RAM tier and pickled for disk, 2.3 s in all, the pickling holding
the interpreter lock while the next statement ran. The store is priced
before anything is copied (`many_frames_cost`) and refused when it costs
more than the loop: the loop runs again, with the reason in its badge, and
what it built is the same. Under ``@cash:persist``, the control, it is
stored and restored.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import pandas as pd\nt = pd.DataFrame({'user': ['s10'] * 3, 'n': [1, 2, 3]})"
LOOP = "frames = []\nfor i in range(1500):{}\n    frames.append(t.copy())"
READ = "print(len(frames), sum(int(f['n'].sum()) for f in frames))"
METRICS = "get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']"
STATUS = f"[s['status'] for s in {METRICS}]"
REASONS = f"[r for s in {METRICS} for r in [s.get('skipped_reason')] if r]"


@pytest.mark.parametrize(
    ("annotation", "stored"), [("", False), ("  # @cash:persist", True)], ids=["reruns", "persist"]
)
def test_the_loop_is_stored_only_when_storing_is_cheaper(nb_runner, annotation, stored):
    nb_runner.create_notebook([SETUP, LOOP.format(annotation), READ])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    reasons = nb_runner.peek(REASONS)
    assert ("tables would take" in reasons) is not stored, reasons
    nb_runner.run_cell(3)
    assert "1500 9000" in nb_runner.get_output(3)

    nb_runner.peek("frames.clear()")
    nb_runner.run_cell(2)
    status = nb_runner.peek(STATUS)
    assert ("RESTORED" in status) is stored, status
    nb_runner.run_cell(3)
    assert "1500 9000" in nb_runner.get_output(3)
