"""Whether a fast statement's share check finished leaves the lineages alone.

The share check of a statement may take a tenth of a second
(``SHARE_CHECK_FLOOR_S``). In handson-ml's ensembles chapter,
``estimators = [rf, et, svm, mlp]`` before ``for estimator in estimators:
estimator.fit(X, y)`` sometimes finished its check and was stored, and
sometimes ran past it and was not. Stored, a second Run All served it with
the four estimators restored from one entry, sharing their parameter-name
strings as the ones built in a cell do; not stored, it took them as their own
statements left them, restored apart, holding copies. The loop gave the list
a lineage from a digest of what it held, which the shared strings changed, so
on two runs out of five ``voting_clf.fit(X_train, y_train)`` (18.7 s) ran
again because "'estimators', stored with this statement's outputs, has
changed". Here the check of the list is made to run out every time, as a
busy machine makes it now and then.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

DEFINE = """import os, time
class Est:
    def __init__(self, criterion="gini", max_features="sqrt"):
        self.criterion = criterion
        self.max_features = max_features
        self.params = ("criterion", "max_features")
    def fit(self, X):
        self.n_seen_ = len(X)
        return self
def vote(models, log):
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)
    time.sleep(SLEEP)
    return [m.n_seen_ for m in models]"""

LIST = "ms = [a, b]"


@pytest.fixture
def store_everything(cash_instance):
    """Every statement stored, so ``a`` and ``b`` are served apart on the
    second run, as ``random_forest_clf`` was."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0


@pytest.fixture
def list_check_runs_out(monkeypatch):
    """The share check of ``ms = [a, b]`` runs out of time; every other
    statement's has time to spare."""
    real = processor.StatementProcessor._refuse_unrestorable_outputs

    def refuse(self, run, *args, **kwargs):
        if run.code.strip() != LIST:
            return real(self, run, *args, **kwargs)
        with monkeypatch.context() as m:
            m.setattr(processor, "SHARE_CHECK_FLOOR_S", 0.0)
            m.setattr(processor, "SHARE_CHECK_FACTOR", 0.0)
            return real(self, run, *args, **kwargs)

    monkeypatch.setattr(processor.StatementProcessor, "_refuse_unrestorable_outputs", refuse)


def _cells(log) -> list[str]:
    return [
        DEFINE.replace("SLEEP", repr(ABOVE_PERSISTENCE_FLOOR_S)),
        "X = list(range(50))",
        'a = Est()\nb = Est(criterion="entropy")',
        f"{LIST}\nfor m in ms:\n    m.fit(X)",
        f"out = vote(ms, {str(log)!r})",
    ]


def _run_all(cash_magics, cells) -> dict[str, str]:
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    lineage = cash_magics.tracking_state.variable_lineage
    return {name: lineage[name] for name in ("ms", "out")}


@pytest.mark.usefixtures("store_everything", "list_check_runs_out")
def test_a_second_run_all_serves_what_reads_the_fitted_list(cash_magics, tmp_path):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "votes.log"
    cells = _cells(log)

    first = _run_all(cash_magics, cells)
    second = _run_all(cash_magics, cells)

    assert second == first
    assert log.read_bytes() == b"x"
    assert cash_magics.shell.user_ns["out"] == [50, 50]


@pytest.mark.usefixtures("store_everything", "list_check_runs_out")
def test_changing_an_estimators_parameter_still_runs_the_reader_again(cash_magics, tmp_path):
    """Guard: blind to which strings are shared, not to what changed."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "votes.log"
    cells = _cells(log)

    first = _run_all(cash_magics, cells)
    cells[2] = 'a = Est()\nb = Est(criterion="log_loss")'
    second = _run_all(cash_magics, cells)

    assert second["ms"] != first["ms"]
    assert log.read_bytes() == b"xx"
    assert cash_magics.shell.user_ns["ms"][1].criterion == "log_loss"
