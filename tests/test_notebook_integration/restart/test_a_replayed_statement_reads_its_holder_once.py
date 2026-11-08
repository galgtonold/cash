"""After a restart, a statement replayed upstream reads what holds its object once.

E8 (sklearn ensembles): after a restart, the statements above the cell run
again (``voting_clf.set_params(...)``, ``estimators.pop()``). After each,
the list holding the fitted forests was pickled twice: by the check that
its holders did not change (`_unchanged_holders`) and again for its session
hash (`replay_derivation_bumps`), with only cash's code between. The first
read was outside any `value_hash.one_read` window (the window closes before
the checks that count weak references), so the second could not reuse it.
"""

from __future__ import annotations

SETUP = "import cash\n%load_ext cash\n%cash_on"
MAKE = "import argparse\nimport numpy as np\nm = argparse.Namespace(w=np.zeros((300, 300)), n=0)\nmodels = [m]"
#: Each changes ``m`` in place, so each checks ``models``, which holds it.
CHANGES = "m.n += 1\nm.w[0, 0] += 1\nm.n += 1\nm.w[0, 0] -= 1\nm.n += 1"
COUNT = (
    "exec('''\n"
    "from cash import value_hash\n"
    "READS = []\n"
    "_real = value_hash._compute_hash\n"
    "def _counted(v):\n"
    "    if v is globals().get('models'):\n"
    "        READS.append(1)\n"
    "    return _real(v)\n"
    "value_hash._compute_hash = _counted\n"
    "''', globals()) or 0"
)


def test_each_replayed_statement_reads_the_list_holding_its_object_once(nb_runner):
    nb_runner.create_notebook([SETUP, MAKE, CHANGES, "print('N', models[0].n, m.w.sum())"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.peek(COUNT)
    nb_runner.run_cell(4)
    reads = int(nb_runner.peek("len(READS)"))
    nb_runner.peek("exec('value_hash._compute_hash = _real', globals()) or 0")
    assert "N 3 0.0" in nb_runner.get_output(4), nb_runner.get_output(4)
    # ``models = [m]`` and the five statements that change ``m``: each
    # replayed, each reads ``models`` once after it ran.
    assert reads <= 6, f"models was pickled {reads} times replaying 6 statements"
