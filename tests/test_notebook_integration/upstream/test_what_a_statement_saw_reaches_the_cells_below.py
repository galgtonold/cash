"""A slow statement built on a value that changed is computed again, as a
plain kernel computes it, though the value's own statement did not change.

The value changes without its source or inputs changing: the clock or a fresh
id read on every run, the drawing a helper counts on pyplot's current figure,
a settings dict a function reads, changed in place. The statement making it
ran again (or was keyed anew), but the value kept its lineage, so the slow
statement below kept its key and was served the first run's result, from RAM
and after a restart from disk.
"""

import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

REPORT = "import time, uuid\nfrom datetime import datetime\n\ndef build_report(run_id):\n    time.sleep(0.2)\n    return {'run': run_id}"


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    "statement",
    ["run_id = datetime.now().strftime('%H%M%S%f')", "run_id = uuid.uuid4().hex"],
    ids=["clock", "uuid"],
)
def test_a_report_on_this_runs_id(nb_runner, statement):
    nb_runner.create_notebook([REPORT, statement, "report = build_report(run_id)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.peek("run_id")
    time.sleep(0.01)
    nb_runner.run_all()
    assert nb_runner.peek("run_id") != first, "setup: the id must change"
    assert nb_runner.peek("report['run']") == nb_runner.peek("run_id")

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert nb_runner.peek("report['run']") == nb_runner.peek("run_id")


def test_a_message_on_what_a_helper_counts_on_the_figure(nb_runner):
    helpers = (
        "import time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n"
        "def n_lines():\n    return len(plt.gca().lines)\n\n"
        "def describe(n):\n    time.sleep(0.2)\n    return f'{n} lines drawn'"
    )
    nb_runner.create_notebook([helpers, "plt.figure()\nplt.plot([1, 2])", "k = n_lines()\nmsg = describe(k)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("msg") == "'1 lines drawn'"
    nb_runner.set_cell_source(2, "plt.figure()\nplt.plot([1, 2])\nplt.plot([3, 4])")
    nb_runner.run_all()
    assert nb_runner.peek("msg") == "'2 lines drawn'"


def test_a_report_on_a_setting_changed_in_place(nb_runner, tmp_path):
    (tmp_path / "d.csv").write_text("k,v\nx,1.234\ny,2.345\nx,3.456\ny,4.567\n", newline="\n", encoding="utf-8")
    cells = [
        "import pandas as pd, time\nCFG = {'round': 2}",
        "def summarize(df):\n    time.sleep(0.05)\n    return df.groupby('k').v.mean().round(CFG['round']).to_dict()\n\n"
        "def render(d):\n    time.sleep(0.2)\n    return ', '.join(f'{k}={v}' for k, v in d.items())",
        "raw = pd.read_csv('d.csv')",
        "res = summarize(raw)\nreport = render(res)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("report") == "'x=2.34, y=3.46'"
    nb_runner.set_cell_source(3, cells[2] + "\nCFG['round'] = 1")
    nb_runner.run_all()
    assert nb_runner.peek("report") == "'x=2.3, y=3.5'"
