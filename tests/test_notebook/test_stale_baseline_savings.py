"""``%cash_stats`` must not claim a win it cannot verify (CAS-157).

Reported by independent testers in three consecutive rounds: a big-frame CSV
ETL notebook run across a kernel restart was ~4x SLOWER by wall clock than just
recomputing (unpickling 300MB frames costs more than re-parsing an OS-cached
CSV), and ``%cash_stats`` still reported "Net +5.7s..+9s saved".

The mechanism is the *baseline*, not the arithmetic. A restore is credited with
``saved_time`` — the ``execution_time`` recorded when the value was FIRST
computed, typically on a colder machine (cold page cache, cold imports). Cash
books today's real restore cost as overhead but credits yesterday's colder
compute cost as the saving, so the subtraction can print positive on a session
that lost wall time. No amount of estimating fixes this: the true recompute
cost is unknowable without doing the recompute.

So the headline NET is credited only from savings this session *verified* by
computing the same statement itself. The stale-baseline figure is still
reported, explicitly as an unverified upper bound. These tests pin:

* a session dominated by restores of stale baselines can NEVER print a positive
  net, however large the stale credit;
* the unverified upper bound is still shown, but flagged and never asserted as
  fact;
* a saving whose baseline this session re-measured DOES read as a verified win
  (the fix must not simply zero the metric out); and
* where the re-measured baseline is *cheaper* than the cache's, the cheaper —
  defensible — number is the one credited.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics


class _MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = _MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


def _stats_json(magics, capsys) -> dict:
    capsys.readouterr()  # drop anything buffered
    magics.cash_stats("json")
    return json.loads(capsys.readouterr().out.strip())


def _replay_retail_etl(magics) -> None:
    """The reported P1 session: run-all after a restart, every frame restored.

    Three ~300MB frames. Each was first cached on a cold run that took 25s to
    parse; each restore costs 8s of unpickling today. Cash's mediated wall time
    is 24s. Nothing is recomputed, so nothing corroborates the 25s baselines.
    """
    for i in range(3):
        magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 25.0,
              "execution_time": 0.0, "code": f"df{i} = pd.read_csv(f{i})"}],
            cell_total_time=8.0,
        )


class TestStaleBaselineCannotPrintAWin:
    """The headline net on the reported session. Fails on the CAS-157 baseline,
    which reports net = 75.0 − 24.0 = +51.0s for a session that lost time."""

    def test_restore_dominated_stale_session_nets_non_positive(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        _replay_retail_etl(magics)

        data = _stats_json(magics, capsys)
        # The stale credit is huge and the session still must not read as a win.
        assert data["total_time_saved"] == pytest.approx(75.0)
        assert data["total_overhead"] == pytest.approx(24.0)
        assert data["total_verified_saved"] == 0.0
        # THE BAR: nothing verified the 25s baselines, so the net cannot be
        # positive. It is exactly the overhead cash measured itself spending.
        assert data["net_time_saved"] <= 0.0
        assert data["net_time_saved"] == pytest.approx(-24.0)
        assert data["net_sign_verified"] is False

    def test_arbitrarily_large_stale_credit_still_cannot_net_positive(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        # A pathological stale baseline: an hour of "saving" credited from cache
        # metadata, on a cell that cost 5s of real wall time and verified nothing.
        magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 3600.0,
              "execution_time": 0.0, "code": "huge = load()"}],
            cell_total_time=5.0,
        )
        data = _stats_json(magics, capsys)
        # No quantity of unverified credit can buy a positive headline.
        assert data["net_time_saved"] <= 0.0
        assert data["net_sign_verified"] is False

    def test_human_output_does_not_assert_a_positive_net(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        _replay_retail_etl(magics)
        capsys.readouterr()
        magics.cash_stats("")
        out = capsys.readouterr().out

        net_line = next(line for line in out.splitlines() if "Net time saved:" in line)
        # The reported lie, verbatim in shape: "Net time saved:      51.0s".
        # The headline must not read as a bare positive figure.
        assert "at least -24.0s" in net_line
        # The unverified ceiling is still shown — as a claim, not a fact.
        assert "at best" in net_line
        assert "51.0s" in net_line
        # ... and the reason the ceiling is untrustworthy is spelled out.
        assert "when first cached" in out
        assert "caching off" in out


class TestVerifiedSavingsStillRead:
    """The fix must not just zero the metric out: a saving cash can stand
    behind must still print as a win."""

    def test_same_session_recompute_verifies_the_saving(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        # Cell computes a 30s fit HERE — that is a baseline measured under
        # today's conditions ...
        magics._update_session_stats(
            [{"status": CacheStatus.COMPUTED, "execution_time": 30.0, "code": "model = fit(X)"}],
            cell_total_time=30.2,
        )
        # ... and re-running the cell restores it in 0.2s.
        magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 30.0,
              "execution_time": 0.0, "code": "model = fit(X)"}],
            cell_total_time=0.2,
        )
        data = _stats_json(magics, capsys)
        assert data["total_verified_saved"] == pytest.approx(30.0)
        # Verified: overhead is the 0.4s of both cells, so the net is a real win.
        assert data["net_time_saved"] == pytest.approx(29.6)
        assert data["net_sign_verified"] is True

    def test_verified_win_is_labelled_and_not_hedged(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        magics._update_session_stats(
            [{"status": CacheStatus.COMPUTED, "execution_time": 30.0, "code": "model = fit(X)"}],
            cell_total_time=30.2,
        )
        magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 30.0,
              "execution_time": 0.0, "code": "model = fit(X)"}],
            cell_total_time=0.2,
        )
        capsys.readouterr()
        magics.cash_stats("")
        out = capsys.readouterr().out
        net_line = next(line for line in out.splitlines() if "Net time saved:" in line)
        assert "(verified)" in net_line
        # A proven win is stated straight — no range, no hedge.
        assert "at best" not in net_line
        assert "cash cost you" not in out

    def test_stale_high_baseline_is_credited_at_the_remeasured_cost(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        # The P1 pathology in miniature: the cache says the frame cost 25s (cold),
        # but this session re-parsed it in 6s (warm). Only the 6s is defensible.
        magics._update_session_stats(
            [{"status": CacheStatus.COMPUTED, "execution_time": 6.0, "code": "df = pd.read_csv(p)"}],
            cell_total_time=6.1,
        )
        magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 25.0,
              "execution_time": 0.0, "code": "df = pd.read_csv(p)"}],
            cell_total_time=8.0,
        )
        data = _stats_json(magics, capsys)
        # Gross still carries the cache's stale 25s ...
        assert data["total_time_saved"] == pytest.approx(25.0)
        # ... but only the re-measured 6s is credited as verified.
        assert data["total_verified_saved"] == pytest.approx(6.0)
        # Restoring (8s) cost MORE than recomputing (6s): a real loss, and the
        # headline says so instead of booking +17s off the stale baseline.
        assert data["net_time_saved"] < 0.0
        assert data["net_time_saved"] == pytest.approx(6.0 - 8.1)


class TestVerificationFiresOnTheRealPipeline:
    """The verified channel is keyed on a statement's ``code``, so it only works
    if the COMPUTED and RESTORED paths agree on that key. They do — one
    ``metrics`` dict is built per statement and both branches mutate it — but
    nothing else pins it. If a refactor split the key, verification would
    silently never fire and every session would report ``-overhead`` forever:
    a silent understatement, which no other test here would catch.
    """

    def test_compute_then_restore_credits_a_verified_saving(self, magics_fixture):
        magics, shell, _backend = magics_fixture
        magics._badge_mode = 'off'
        # A statement expensive enough to clear the 10ms cache floor, with no
        # import (imports are cheap and separately cached).
        cell = "slow = sum(i * i for i in range(2_000_000))"

        magics.cash("", cell)
        stats = magics._session.stats
        assert stats["statements_computed"] == 1
        # The baseline was measured HERE, keyed by the statement source.
        assert magics._session.measured_compute, (
            "no baseline recorded for a COMPUTED statement"
        )

        magics.cash("", cell)
        assert stats["statements_restored"] == 1, "second run did not hit the cache"
        # THE CONTRACT: the restore found the baseline this session measured, so
        # the saving is verified rather than taken on faith from the cache.
        assert stats["total_verified_saved"] > 0.0
        assert stats["total_verified_saved"] == pytest.approx(stats["total_time_saved"])

    def test_stats_reset_forgets_the_measured_baselines(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        magics._badge_mode = 'off'
        magics.cash("", "slow2 = sum(i * i for i in range(2_000_000))")
        assert magics._session.measured_compute

        magics.cash_stats("reset")
        # A reset that kept the baselines would go on verifying savings against
        # measurements it claims to have forgotten.
        assert magics._session.measured_compute == {}
        data = _stats_json(magics, capsys)
        assert data["total_verified_saved"] == 0.0


class TestCertainLossStillReadsAsALoss:
    """CAS-143's negative case must survive: where even the most generous
    reading of the cache's baselines is a loss, say so plainly."""

    def test_no_savings_at_all_still_says_cash_cost_you(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        for i in range(6):
            magics.cash("", f"cheap_{i} = {i} + 1")
        capsys.readouterr()
        magics.cash_stats("")
        out = capsys.readouterr().out
        # Nothing was cached, so no baseline is in question and the sign is
        # certain without verifying anything — report it bluntly, not as a range.
        assert "cash cost you" in out
        assert "at best" not in out


class TestUpstreamComputeIsNotOverhead:
    """H1, rejected on the evidence and pinned so it is not 'fixed' later.

    Upstream re-execution is the user's own notebook code rebuilding state the
    user would otherwise have rebuilt by hand. Booking it as cash's overhead
    would make cash understate itself by the size of the user's ETL on exactly
    the sessions where it helps most. It is also demonstrably NOT the CAS-157
    mechanism: the reported session re-executes nothing upstream and still
    overstated.
    """

    def test_upstream_recompute_counts_as_user_compute(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        # 40s of upstream ETL re-run inside a 41s cell: cash's own tax is the 1s,
        # not the 41s.
        magics._update_session_stats(
            [
                {"status": CacheStatus.COMPUTED, "execution_time": 40.0,
                 "is_upstream": True, "code": "df = etl()"},
                {"status": CacheStatus.COMPUTED, "execution_time": 0.01,
                 "is_upstream": False, "code": "print(df.shape)"},
            ],
            cell_total_time=41.0,
        )
        data = _stats_json(magics, capsys)
        assert data["total_compute_time"] == pytest.approx(40.01)
        assert data["total_overhead"] == pytest.approx(0.99, abs=1e-6)

    def test_stale_overstatement_occurs_with_zero_upstream_compute(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        _replay_retail_etl(magics)
        data = _stats_json(magics, capsys)
        # The reported session ran no upstream compute at all, so reclassifying
        # upstream time could not have moved its net by a single millisecond.
        assert data["total_compute_time"] == 0.0
        assert data["net_time_saved_upper_bound"] == pytest.approx(51.0)
