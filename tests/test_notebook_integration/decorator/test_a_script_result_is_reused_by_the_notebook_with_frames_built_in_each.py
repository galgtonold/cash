"""A script's cached calls are reused by a notebook that builds the same
frames itself, and a different fitted model still recomputes.

The credit-risk shape: ``score.py`` loads a CSV, slices ``train[cols]``
and passes it to a ``frozen=True`` fit, whose result goes on into a second
cached call. The notebook does the same in cells (``X_train =
train[features]``, ``model = lib.train(...)``, ``lib.importance(model, ...)``).
The notebook-first direction is covered by
test_a_frozen_result_held_by_a_notebook_variable_keys_by_its_producer.py;
this covers script first, then the notebook, then a restart, plus the
guard that a model fitted with other parameters never shares an entry.
"""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

LIB = """\
import time

import pandas as pd

import cash

RUNS = {runs!r}


def _log(what):
    with open(RUNS, "a") as fh:
        fh.write(what + "\\n")


@cash.cache(frozen=True)
def load(path):
    return pd.read_csv(path)


def split(df):
    return df[df["t"] < 30], df[df["t"] >= 30]


class Model:
    def __init__(self, w):
        self.w = w


@cash.cache(frozen=True, assume_safe=True)
def train(X, y, params):
    _log("train")
    time.sleep(0.3)
    return Model(float((X.sum(axis=1) * y).sum()) * params["k"])


@cash.cache
def importance(model, X, y):
    _log("importance")
    time.sleep(0.3)
    return model.w + float(X.to_numpy().sum()) + float(y.sum())
"""

SCRIPT = """\
import sys

import lib

k = int(sys.argv[1])
df = lib.load("data.csv")
cols = [c for c in df.columns if c.startswith("f")]
train, test = lib.split(df)
model = lib.train(train[cols], train["y"], {"k": k})
print("R", lib.importance(model, test[cols], test["y"]))
"""

CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import lib",
    'df = lib.load("data.csv")\nfeatures = [c for c in df.columns if c.startswith("f")]',
    'train, test = lib.split(df)\nX_train, y_train = train[features], train["y"]\n'
    'X_test, y_test = test[features], test["y"]',
    'params = {"k": 2}',
    "model = lib.train(X_train, y_train, params)",
    "imp = lib.importance(model, X_test, y_test)\nprint('R', imp)",
]


def _counts(path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.split()
    return lines.count("train"), lines.count("importance")


def _script(work, k):
    return subprocess.run(
        [sys.executable, "score.py", str(k)],
        cwd=work,
        # The kernel's cache: the default ``.cash`` in the folder it runs in.
        env={**{k: v for k, v in os.environ.items() if k != "CASH_CACHE_DIR"}, "CASH_VERBOSE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _write_project(work, runs):
    (work / "lib.py").write_text(LIB.format(runs=str(runs)), encoding="utf-8")
    (work / "score.py").write_text(SCRIPT, encoding="utf-8")
    rows = ["t,f1,f2,other,y"] + [f"{t},{t % 7},{(t * 3) % 5},{t % 2},{t % 3 == 0:d}" for t in range(40)]
    (work / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_a_script_run_first_is_reused_by_the_notebook_and_after_a_restart(nb_runner, tmp_path):
    work = nb_runner.work_dir
    runs = tmp_path / "runs.txt"
    _write_project(work, runs)

    first = _script(work, 2)
    assert first.returncode == 0, first.stdout + first.stderr
    expected = first.stdout.split("R ", 1)[1].split()[0]
    assert _counts(runs) == (1, 1), "control: the script computed both calls"

    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(7)
    assert f"R {expected}" in out, out
    assert _counts(runs) == (1, 1), "the notebook recomputed what the script stored"

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3, 4, 5, 6, 7])
    out = nb_runner.get_output(7)
    assert f"R {expected}" in out, out
    assert _counts(runs) == (1, 1), "the notebook recomputed the script's calls after a restart"


def test_a_model_fitted_with_other_parameters_never_shares_the_entry(nb_runner, tmp_path):
    """Equal frames, a different fitted model: both calls run again, in either place."""
    work = nb_runner.work_dir
    runs = tmp_path / "runs.txt"
    _write_project(work, runs)

    first = _script(work, 2)
    assert first.returncode == 0, first.stdout + first.stderr
    at_k2 = first.stdout.split("R ", 1)[1].split()[0]

    cells = list(CELLS)
    cells[4] = 'params = {"k": 3}'
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(7)
    assert f"R {at_k2}" not in out, out
    assert _counts(runs) == (2, 2), "the notebook served the k=2 entries for a k=3 model"
    at_k3 = out.split("R ", 1)[1].split()[0]
    nb_runner.settle_writes()

    again = _script(work, 3)
    assert again.returncode == 0, again.stdout + again.stderr
    assert f"R {at_k3}" in again.stdout, again.stdout + again.stderr
    assert _counts(runs) == (2, 2), "the script recomputed the notebook's k=3 calls\n" + again.stderr
