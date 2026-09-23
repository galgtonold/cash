"""A function that reads a loop's result is not rebuilt when the loop's cell re-runs.

With no edit at all: run the model-comparison cell again
(``results = {}`` filled by a loop over model families), then the report cell
that saves a figure drawn by functions reading ``results``. Cash raised
``UpstreamStateError: 'logreg'`` and left ``results`` empty.

The check for values built on an older input counted a ``def`` as
built on ``results``: a function records the globals it reads among its
inputs. It holds no copy of them -- it reads them when called -- but the
functions were rebuilt, and with them ``results = {}`` without the loop.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.loops]

CELLS = [
    "import cash\n%cash_on",
    "import time\n"
    "def evaluate(name):\n"
    "    time.sleep(0.05)\n"
    "    return {'score': len(name)}\n"
    "families = ['logreg', 'forest', 'boost']",
    "results = {}\nfor name in families:\n    results[name] = evaluate(name)",
    "def draw_best():\n"
    "    return max(results, key=lambda n: results[n]['score'])\n"
    "def draw_logreg():\n"
    "    return results['logreg']['score']",
    "report = (draw_best(), draw_logreg())",
    "print('REPORT', report)",
]


def test_rerunning_the_loop_cell_then_the_report_cell_works(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "REPORT ('logreg', 6)" in nb_runner.get_output(6)

    nb_runner.run_cell(3)
    nb_runner.run_cell(6)

    assert "REPORT ('logreg', 6)" in nb_runner.get_output(6), nb_runner.get_output(6)
