"""Run Alls that take no digest of an ensemble's estimators refit nothing.

E6 (homl ch02): ``cv_res.sort_values(..., inplace=True)`` on the frame built
from ``grid_search.cv_results_`` runs on every Run All (its share check
gives up, so it is never stored), and it changes ``grid_search`` in place,
and so, maybe, the variables holding the pipelines the search holds. On the
first Run All its digests saw them unchanged: they kept their lineages, and
the moves they would have taken were recorded as aliases of those lineages,
which the upstream simulation resolves. On a later Run All, with more
objects in the namespace, the run took no digest and moved every one of
them on. The simulation kept them, so the next cell re-ran upstream every
cell that fits on them: 130 s of refits on a same-code Run All, 1.83x plain
(``tests/test_notebook/upstream/test_a_share_check_out_of_time_moves_held_
estimators_as_a_digest_did.py`` pins that step).

Here ``vote.fit`` stands in, dropped from the cache before each later Run
All, which takes no digest and whose share check gives up: on a same-code
Run All and after a restart, only the ensemble's own clones are fitted
again, and nothing runs upstream.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\ncash.configure(min_execution_time_to_cache_seconds=0.0)",
    (
        "import os\n"
        "LOG = os.path.abspath('fits.log')\n"
        "def mark():\n"
        "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'f')\n"
        "    os.close(fd)"
    ),
    (
        "import time\n"
        "import numpy as np\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "class Tree(DecisionTreeClassifier):\n"
        "    def fit(self, X, y, sample_weight=None, check_input=True):\n"
        "        mark()\n"
        "        time.sleep(0.2)  # worth storing on disk too\n"
        "        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)"
    ),
    "X = np.array([[i % 7, i % 5] for i in range(60)])\ny = np.array([i % 2 for i in range(60)])",
    "a = Tree(max_depth=2, random_state=0)\nb = Tree(max_depth=3, random_state=0)",
    "a.fit(X, y)",
    "b.fit(X, y)",
    "from sklearn.ensemble import VotingClassifier\nvote = VotingClassifier([('a', a), ('b', b)])",
    # Runs on every later Run All (`_later_run_all`), as ``cv_res.sort_values(...)`` did.
    "vote.fit(X, y)",
    "from sklearn.base import clone\nagain = clone(a).fit(X, y)\nprint('depth', a.get_depth(), again.get_depth())",
]
NO_DIGEST = (
    "exec('from cash.notebook.statement import processor as p\\n"
    "p.SHARE_CHECK_FLOOR_S = 0.0\\np.SHARE_CHECK_FACTOR = 0.0\\np.SHARE_FALLBACK_FLOOR_S = 0.0\\n"
    "p.StatementProcessor._holder_digests = lambda self, run: {}\\n"
    "p.StatementProcessor._unchanged_holders = lambda self, run, holders: set()', {})"
)


FORGET_THE_FIT = (
    "(lambda backend: len([backend.delete(meta['key']) for meta in list(backend.list_entries() or ())"
    " if str(meta.get('code', '')).strip() == 'vote.fit(X, y)']))"
    "(get_ipython().magics_manager.registry['CashMagics']._cash_instance.backend)"
)


def _later_run_all(nb) -> int:
    """A Run All that takes no digest and finds no entry for the ensemble's
    fit; the fits it made."""
    nb.peek(NO_DIGEST)
    # Stored by the first Run All only: the later ones store nothing.
    nb.peek(FORGET_THE_FIT)
    nb.run_all()
    return _fits(nb)


def _fits(nb) -> int:
    log = nb.work_dir / "fits.log"
    n = len(log.read_bytes()) if log.exists() else 0
    log.unlink(missing_ok=True)
    return n


def _printed(nb) -> list[str]:
    return [line for line in nb.get_output(len(CELLS)).splitlines() if line.startswith("depth")]


@pytest.mark.timeout(300)
def test_run_alls_taking_no_digest_fit_only_the_ensembles_clones(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _fits(nb_runner) == 5  # a, b, the ensemble's two clones, and the clone of a
    printed = _printed(nb_runner)

    same_code = _later_run_all(nb_runner)
    assert _printed(nb_runner) == printed
    nb_runner.restart()
    after_restart = _later_run_all(nb_runner)
    again = _later_run_all(nb_runner)

    # The ensemble's two clones, which it fits again on each: nothing more.
    assert (same_code, after_restart, again) == (2, 2, 2)
    assert _printed(nb_runner) == printed
    for i in range(1, len(CELLS) + 1):
        assert "Upstream" not in nb_runner.get_output(i), nb_runner.get_output(i)
