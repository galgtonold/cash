"""A fit that works on clones leaves the estimators the ensemble holds alone.

``VotingClassifier(named_estimators).fit(X, y)`` fits copies of the
estimators it holds. Cash stores those estimators with the ensemble's
outputs, as variables sharing its objects, and moved their lineages on at
its fit all the same: running the scoring cell above it again then refitted
every estimator (E8, homl ch07: 36 s of fits for a 0.3 s cell). It now takes
an in-place-edit digest of each of them just before the statement runs and
again after, and keeps the lineage of one the fit left as it was. Nothing is
known of the library: ``VotingClassifier`` fits clones, ``Pipeline`` fits
the steps it holds in place, and only the digests tell them apart.

Each case also runs with the share check forced to give up on every
statement, as a loaded machine makes it now and then: what is computed and
how often anything is fitted still match plain Python.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell

pytest.importorskip("sklearn")

DEFINE = """import os
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
LOG = LOG_PATH
def mark():
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"f")
    os.close(fd)
class Tree(DecisionTreeClassifier):
    def fit(self, X, y, sample_weight=None, check_input=True):
        mark()
        return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
class Scaler(StandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        mark()
        return super().fit(X, y, sample_weight=sample_weight)"""

DATA = "X = [[i % 7, i % 5] for i in range(60)]\ny = [i % 2 for i in range(60)]"
ENSEMBLE = [
    DATA,
    "a = Tree(max_depth=2, random_state=0)\nb = Tree(max_depth=3, random_state=0)",
    "ests = [a, b]\nfor e in ests:\n    e.fit(X, y)",
    "scores = [e.get_depth() for e in ests]",
    "named = [('a', a), ('b', b)]",
    "vote = VotingClassifier(named)",
    "vote.fit(X, y)",
    "out = vote.score(X, y)",
    "depth = a.get_depth()",
]
SCORES = 4  # index into the cells, DEFINE first

CHAIN = [
    DATA,
    "scaler = Scaler()\nchain = Pipeline([('scale', scaler), ('tree', Tree(random_state=0))])",
    "chain.fit(X, y)",
]


@pytest.fixture(autouse=True)
def _digests_a_busy_machine_cannot_cut_short(monkeypatch):
    """The holder digests run within a share of the statement's time
    (`HOLDER_DIGEST_FLOOR_S`); a loaded worker could run past it and move
    the estimators on. Pinned far above, unless a test sets it."""
    monkeypatch.setattr(processor, "HOLDER_DIGEST_FLOOR_S", 30.0)
    monkeypatch.setattr(processor, "HOLDER_DIGEST_BUDGET_S", 30.0)


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0


@pytest.fixture(params=["in time", "share check gives up"])
def share_check(request, monkeypatch):
    if request.param == "share check gives up":
        monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 0.0)
        monkeypatch.setattr(processor, "SHARE_CHECK_FACTOR", 0.0)
        monkeypatch.setattr(processor, "SHARE_FALLBACK_FLOOR_S", 0.0)
    return request.param


def _fits(log) -> int:
    n = len(log.read_bytes()) if log.exists() else 0
    log.unlink(missing_ok=True)
    return n


def _start(cash_magics, tmp_path, body, upto=None):
    """Run the notebook *body* follows ``DEFINE`` in, up to cell *upto*."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "fits.log"
    cells = [DEFINE.replace("LOG_PATH", repr(str(log))), *body]
    for cell in cells[:upto]:
        run_cash_cell(cash_magics, cell, cells=cells)
    return log, cells


def _lineage_across(cash_magics, cells, cell, name):
    """*name*'s lineage before and after *cell* runs."""
    lineage = cash_magics.tracking_state.variable_lineage
    before = lineage[name]
    run_cash_cell(cash_magics, cell, cells=cells)
    return before, lineage[name]


@pytest.mark.usefixtures("store_everything", "share_check")
def test_the_scoring_cell_run_again_fits_nothing(cash_magics, tmp_path):
    log, cells = _start(cash_magics, tmp_path, ENSEMBLE)
    ns = cash_magics.shell.user_ns
    assert _fits(log) == 4  # plain Python: a, b, and the ensemble's two clones
    scores, out = ns["scores"], ns["out"]

    depth = ns["depth"]

    run_cash_cell(cash_magics, cells[SCORES], cells=cells)
    run_cash_cell(cash_magics, cells[-2], cells=cells)
    # Below the ensemble's fit: the simulation keeps ``a``'s lineage there
    # as the run did, by the alias the run recorded, whether or not the
    # fit was stored.
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert _fits(log) == 0
    assert ns["scores"] == scores
    assert ns["out"] == out
    assert ns["depth"] == depth
    assert ns["vote"].estimators[0][1] is ns["a"]


@pytest.mark.usefixtures("store_everything", "share_check")
def test_the_ensembles_fit_keeps_the_lineages_of_what_it_holds(cash_magics, tmp_path):
    _, cells = _start(cash_magics, tmp_path, ENSEMBLE, upto=-3)

    before, after = _lineage_across(cash_magics, cells, cells[-3], "a")

    assert after == before


@pytest.mark.usefixtures("store_everything", "share_check")
def test_a_pipelines_fit_still_moves_the_step_it_fits(cash_magics, tmp_path):
    """Guard: ``Pipeline.fit`` fits the ``scaler`` it holds in place, so
    ``scaler`` moves on, and what read it before reads it again."""
    log, cells = _start(cash_magics, tmp_path, CHAIN, upto=-1)

    before, after = _lineage_across(cash_magics, cells, cells[-1], "scaler")

    assert after != before
    assert _fits(log) == 2
    assert hasattr(cash_magics.shell.user_ns["scaler"], "mean_")


@pytest.mark.usefixtures("store_everything")
def test_a_digest_not_taken_in_time_moves_the_estimators_on(cash_magics, tmp_path, monkeypatch):
    """Guard: with no time to digest them (a big forest), the estimators
    move on as before, and running the scoring cell again still shows what
    plain Python shows. The loop that fitted them stored them, so they have
    a session digest too: it runs out of time as well."""
    monkeypatch.setattr(processor, "HOLDER_DIGEST_BUDGET_S", 0.0)
    monkeypatch.setattr(processor, "HOLDER_DIGEST_FLOOR_S", 0.0)
    monkeypatch.setattr(processor, "HOLDER_DIGEST_FACTOR", 0.0)
    _, cells = _start(cash_magics, tmp_path, ENSEMBLE, upto=-3)
    ns = cash_magics.shell.user_ns
    scores = ns["scores"]

    def out_of_time(*args, **kwargs):
        raise TimeoutError("past its budget")

    with monkeypatch.context() as m:
        m.setattr(processor.value_hash, "budgeted_hash", out_of_time)
        before, after = _lineage_across(cash_magics, cells, cells[-3], "a")
    run_cash_cell(cash_magics, cells[-2], cells=cells)
    out = ns["out"]
    run_cash_cell(cash_magics, cells[SCORES], cells=cells)
    run_cash_cell(cash_magics, cells[-2], cells=cells)

    assert after != before
    assert ns["scores"] == scores
    assert ns["out"] == out


def _digests_out_of_time(m):
    """Give the holder digests no time, and have each change check run out
    of its budget (a big forest on a busy machine)."""
    m.setattr(processor, "HOLDER_DIGEST_BUDGET_S", 0.0)
    m.setattr(processor, "HOLDER_DIGEST_FLOOR_S", 0.0)
    m.setattr(processor, "HOLDER_DIGEST_FACTOR", 0.0)
    m.setattr(processor.value_hash, "budgeted_hash", lambda *args, **kwargs: None)


@pytest.mark.usefixtures("store_everything")
def test_a_repair_after_a_fit_digested_out_of_time_reads_what_a_run_gives(cash_magics, tmp_path, monkeypatch):
    """The first Run All gave the ensemble's fit no time to digest the
    estimators it holds, so the fit moved them on. Running the scoring
    cell again rebuilds them as a run from the top has them there (the
    loop fitting them is in the cell below the one binding them), and the
    cell reading ``a`` below the fit then rebuilds the rest. The repair
    of the scoring cell gave the snapshot of the cell binding ``a`` the
    fitted ``a`` the loop leaves, and the next repair, re-running ``a =
    Tree(...)`` and ``ests = [a, b]``, was refused (UpstreamRepairGap) as
    reading another ``a`` than a run from the top gives it there."""
    with monkeypatch.context() as m:
        _digests_out_of_time(m)
        log, cells = _start(cash_magics, tmp_path, ENSEMBLE)
    ns = cash_magics.shell.user_ns
    assert _fits(log) == 4
    scores, out, depth = ns["scores"], ns["out"], ns["depth"]

    run_cash_cell(cash_magics, cells[SCORES], cells=cells)
    run_cash_cell(cash_magics, cells[-2], cells=cells)
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert (ns["scores"], ns["out"], ns["depth"]) == (scores, out, depth)
    assert _fits(log) == 0
    assert ns["vote"].estimators[0][1] is ns["a"]


def _share_check_out_of_time(m):
    """Give the share check no time: the statement is not stored, and the
    holders of its outputs are not known."""
    m.setattr(processor, "SHARE_CHECK_FLOOR_S", 0.0)
    m.setattr(processor, "SHARE_CHECK_FACTOR", 0.0)
    m.setattr(processor, "SHARE_FALLBACK_FLOOR_S", 0.0)


@pytest.mark.usefixtures("store_everything")
def test_a_repair_whose_fit_moves_what_it_holds_reads_what_a_run_gives(cash_magics, tmp_path, monkeypatch):
    """The first Run All gave the ensemble's fit no time for its share
    check: not stored, and nothing known to hold its estimators. Run again
    with no time for the digests, the fit is stored with them and moves
    them on. The repair that ran it (for the cell reading ``a``) gave the
    snapshot of the cell binding ``a`` the lineage the fit left, and the
    repair of the scoring cell after it was refused (UpstreamRepairGap)."""
    with monkeypatch.context() as m:
        _share_check_out_of_time(m)
        log, cells = _start(cash_magics, tmp_path, ENSEMBLE)
    ns = cash_magics.shell.user_ns
    assert _fits(log) == 4
    scores, out, depth = ns["scores"], ns["out"], ns["depth"]

    with monkeypatch.context() as m:
        _digests_out_of_time(m)
        for i in (-3, -1, SCORES, -1):
            run_cash_cell(cash_magics, cells[i], cells=cells)

    assert (ns["scores"], ns["out"], ns["depth"]) == (scores, out, depth)
    assert ns["vote"].estimators[0][1] is ns["a"]


@pytest.mark.usefixtures("store_everything")
def test_a_fit_its_share_check_left_unstored_moves_nothing_it_left_alone(cash_magics, tmp_path, monkeypatch):
    """With no time for its share check, the ensemble's fit is not stored,
    and ``vote`` is linked with what the fit read (``X``, ``y``), which it
    may hold. Run again, the fit refits, as plain Python does; the digests
    taken around it show ``X`` and ``y`` left as they were, so they keep
    their lineages, as they do when the check finds in time that nothing
    holds them. Moved on, the cell reading ``a`` below rebuilt everything
    built on ``X`` and fitted the estimators again."""
    monkeypatch.setattr(processor, "SHARE_CHECK_FLOOR_S", 0.0)
    monkeypatch.setattr(processor, "SHARE_CHECK_FACTOR", 0.0)
    monkeypatch.setattr(processor, "SHARE_FALLBACK_FLOOR_S", 0.0)
    log, cells = _start(cash_magics, tmp_path, ENSEMBLE)
    ns = cash_magics.shell.user_ns
    lineage = cash_magics.tracking_state.variable_lineage
    assert _fits(log) == 4
    before = (lineage["X"], lineage["y"])

    run_cash_cell(cash_magics, cells[-3], cells=cells)
    assert _fits(log) == 2  # plain Python: the ensemble's two clones
    assert (lineage["X"], lineage["y"]) == before

    run_cash_cell(cash_magics, cells[-1], cells=cells)
    run_cash_cell(cash_magics, cells[SCORES], cells=cells)
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    assert _fits(log) == 0
    assert ns["vote"].estimators[0][1] is ns["a"]
