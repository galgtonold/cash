"""A ``frozen=True`` result held by a notebook variable keys a cached call
given it by its producer's call, as a script does.

``model = lib.train(...)`` in a cell, then ``lib.importance(model, ...)``:
the notebook re-tagged ``model`` with the statement's lineage, dropping the
decorator's frozen tag, so the call keyed ``model`` differently from the
same call in a script, and the two never shared an entry (158 s of
permutation importance in one credit-risk project). A value restored from
the statement's entry after a restart carried no frozen tag at all.
"""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

LIB = """\
import time

import cash

RUNS = {runs!r}


class Model:
    def __init__(self, k):
        self.k = k


@cash.cache(frozen=True, assume_safe=True)
def train(k):
    time.sleep(0.4)
    return Model(k)


@cash.cache
def importance(model, n):
    with open(RUNS, "a") as fh:
        fh.write("run\\n")
    time.sleep(0.4)
    return model.k * n
"""

SCRIPT = "import lib\nm = lib.train(2)\nprint('R', lib.importance(m, 3))\n"


def _runs(path) -> int:
    return path.read_text().count("run") if path.exists() else 0


def test_the_notebook_and_a_script_share_the_call_and_it_restores_after_a_restart(nb_runner, tmp_path):
    work = nb_runner.work_dir
    runs = tmp_path / "runs.txt"
    (work / "lib.py").write_text(LIB.format(runs=str(runs)))
    (work / "score.py").write_text(SCRIPT)
    nb_runner.create_notebook(
        [
            "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
            "import lib",
            "model = lib.train(2)",
            "imp = lib.importance(model, 3)\nprint('R', imp)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 6" in nb_runner.get_output(4)
    assert _runs(runs) == 1, "control: the notebook computed it"
    nb_runner.settle_writes()

    script = subprocess.run(
        [sys.executable, "score.py"],
        cwd=work,
        # The kernel's cache: the default ``.cash`` in the folder it runs in.
        env={**{k: v for k, v in os.environ.items() if k != "CASH_CACHE_DIR"}, "CASH_VERBOSE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "R 6" in script.stdout, script.stdout + script.stderr
    assert _runs(runs) == 1, "the script recomputed the call the notebook stored\n" + script.stderr

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3])
    nb_runner.add_cell("again = lib.importance(model, 3)\nprint('A', again)")
    nb_runner.run_cells([5])
    assert "A 6" in nb_runner.get_output(5)
    assert _runs(runs) == 1, "a restored model keyed the call differently after the restart"


def test_a_frozen_result_changed_in_a_cell_is_keyed_as_changed(nb_runner, tmp_path):
    """The notebook's own lineage takes over once a cell changes the value."""
    runs = tmp_path / "runs.txt"
    (nb_runner.work_dir / "lib.py").write_text(LIB.format(runs=str(runs)))
    nb_runner.create_notebook(
        [
            "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
            "import lib",
            "model = lib.train(2)",
            "before = lib.importance(model, 3)\nprint('B', before)",
            "model.k = 5",
            "after = lib.importance(model, 3)\nprint('A', after)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "B 6" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert "A 15" in nb_runner.get_output(6), nb_runner.get_output(6)
