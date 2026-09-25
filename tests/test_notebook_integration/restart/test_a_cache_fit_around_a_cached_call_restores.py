"""A ``# @cash:cache-fit`` fit that takes a cached call's result restores, on a
rerun and after a restart, on the default backend.

The fit is the statement's own work -- well above every floor -- however much
of the statement's time the cached ``prep(...)`` call took, so the statement is
stored for it and comes back without fitting again.
"""

import pytest

pytest.importorskip("sklearn")

from tests._nbharness.badge import shows_cached, shows_executed

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import time\n"
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "def prep(n):\n"
    "    time.sleep(0.3)\n"
    "    return [f'the cat {i % 97} sat by dog {i % 89} and bird {i % 83}' for i in range(n)]",
    "vec = TfidfVectorizer()",
    "# @cash:cache-fit\nX = vec.fit_transform(prep(150_000))",
    "# @cash:no-cache\nprint('X', X.shape[0], 'fitted', hasattr(vec, 'vocabulary_'))",
]


def _restored(out: str) -> bool:
    return shows_cached(out) and not shows_executed(out)


def test_the_fit_restores_on_a_rerun_and_after_a_restart(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X 150000 fitted True" in nb_runner.get_output(5)

    nb_runner.run_all()
    assert _restored(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert "X 150000 fitted True" in nb_runner.get_output(5)

    nb_runner.restart()
    nb_runner.run_all()
    assert _restored(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert "X 150000 fitted True" in nb_runner.get_output(5)
