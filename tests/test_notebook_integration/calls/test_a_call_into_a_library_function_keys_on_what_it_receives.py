"""A call whose callee uses a library function is still keyed on what it receives.

Adding a column to ``docs`` (``docs["topic"] = ...``) or
fixing the titles re-ran the vectorise cell, as it should -- ``docs`` changed --
but ``fit_vectors(docs['text'], SEED)`` re-fitted TF-IDF and SVD on byte-
identical text (``0/1 hit``), and every ``cluster_k(Z, k, SEED)`` after it.

``fit_vectors`` calls ``sklearn.preprocessing.normalize``, which sklearn wraps
in a validating decorator: a plain Python function, so cash walked into
sklearn's own globals, found state that is not plain data, and keyed the call
on where its argument came from instead. A library's module state is no more
covered by that key than by the content key; its code is code.
"""

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = (
    "import os, time\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from sklearn.preprocessing import normalize\n"
    "def fit_vectors(texts, seed):\n"
    "    fd = os.open('fits.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'fit|')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.1)\n"
    "    lengths = np.array([[len(t), t.count(' ') + seed] for t in texts], dtype=float)\n"
    "    return normalize(lengths)\n"
    "SEED = 1"
)
DOCS = (
    "docs = pd.DataFrame({'text': ['alpha beta %d' % i for i in range(300)], 'title': ['t%d' % i for i in range(300)]})"
)
VECTORS = "Z = fit_vectors(docs['text'], SEED)\nprint('Z', round(float(Z.sum()), 6))"


def _fits(runner) -> int:
    log = Path(runner.work_dir) / "fits.log"
    return log.read_text(encoding="utf-8").count("fit|") if log.exists() else 0


def test_fixing_the_titles_does_not_refit_on_identical_text(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, DOCS, VECTORS])
    nb_runner.start_kernel()
    nb_runner.run_all()
    want = next(line for line in nb_runner.get_output(4).splitlines() if line.startswith("Z "))
    assert _fits(nb_runner) == 1

    nb_runner.set_cell_source(3, DOCS.replace("'t%d'", "'Title %d'"))  # titles only
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert want in nb_runner.get_output(4), nb_runner.get_output(4)
    assert _fits(nb_runner) == 1, "re-fitted on byte-identical text"


HELPER = (
    "import time\n"
    "LOOKUP = {'a': 1}\n"
    "def scale(v):\n"
    "    return v * LOOKUP['a']\n"
    "def work(v):\n"
    "    time.sleep(0.1)\n"
    "    return scale(v)"
)


def test_a_notebook_helper_is_still_walked(nb_runner):
    """Only a library's function is taken as code: a helper defined in the
    notebook reading a global keeps its global in the key."""
    nb_runner.create_notebook(
        ["import cash\n%cash_on", HELPER, "out = [work(v) for v in [1, 2, 3]]\nprint('OUT', out)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT [1, 2, 3]" in nb_runner.get_output(3)
    nb_runner.set_cell_source(2, HELPER.replace("'a': 1", "'a': 10"))
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "OUT [10, 20, 30]" in nb_runner.get_output(3), nb_runner.get_output(3)
