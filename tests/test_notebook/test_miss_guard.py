"""Tests for the perpetual-miss guard (CAS-172).

The bug being bounded: an input that hashes unstably across runs gives the
statement a different cache key every run, so it never hits — yet cash
re-serialises the (large) value every run. The cache can never pay back and the
statement is net-negative forever. CAS-165 / CAS-166 / CAS-171 were three
independent instances; we conceded we cannot enumerate the causes, so the guard
bounds the consequence instead.

**The instability is injected through the input LINEAGE, which is what a cache
key is actually built from.** ``compute_cache_key`` resolves each input to
``variable_lineage[var]`` (or a content hash when there is none), so writing a
fresh lineage for ``big`` before each run reproduces the real shape — "some input
hashes unstably across runs" — without needing an 8 MiB frame or a live sklearn
fit. A statement whose input has NO lineage at all is refused by
``decide_cacheability`` before the guard is ever consulted, so that route cannot
model the bug.

``TestRestartSaverIsNeverGuarded`` is the load-bearing one. A statement that is
net-negative in-session but saves hugely across a restart is not a bug — it is
the product. If the guard ever fires on a stable key, it has eaten the core value
proposition, and that is a strictly worse outcome than the waste it prevents.
"""
from __future__ import annotations

import itertools
import json
import os
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import FileBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.statement.miss_guard import (
    GUARD_AFTER_CONSECUTIVE_CHURN_MISSES as N_CHURN,
)
from cash.notebook.statement.miss_guard import (
    REPROBE_EVERY_N_RUNS as N_PROBE,
)
from cash.notebook.statement.miss_guard import (
    MissGuard,
    resolve_cache_dir,
)

# ---------------------------------------------------------------------------
# State-machine unit tests — no processor, no backend.
# ---------------------------------------------------------------------------


class TestMissGuardStateMachine:
    def test_cold_run_always_serialises(self):
        """The first sighting of a source must serialise: that is the product."""
        guard = MissGuard(None)
        assert guard.should_serialise("src") is True
        guard.observe("src", "key-1", hit=False)
        assert guard.should_serialise("src") is True

    def test_guards_after_n_consecutive_churn_misses(self):
        guard = MissGuard(None)
        keys = (f"key-{i}" for i in itertools.count())
        guard.observe("src", next(keys), hit=False)  # baseline
        for i in range(N_CHURN - 1):
            guard.observe("src", next(keys), hit=False)
            assert guard.should_serialise("src") is True, f"guarded early at churn {i + 1}"
        guard.observe("src", next(keys), hit=False)  # the Nth churn miss
        assert guard.is_guarded("src") is True
        assert guard.should_serialise("src") is False

    def test_a_hit_resets_the_evidence(self):
        """One hit proves the statement can pay back; the counter must zero."""
        guard = MissGuard(None)
        keys = (f"key-{i}" for i in itertools.count())
        guard.observe("src", next(keys), hit=False)
        for _ in range(N_CHURN - 1):
            guard.observe("src", next(keys), hit=False)
        guard.observe("src", next(keys), hit=True)
        for _ in range(N_CHURN - 1):
            guard.observe("src", next(keys), hit=False)
        assert guard.is_guarded("src") is False
        assert guard.should_serialise("src") is True

    def test_repeated_key_missing_is_not_churn(self):
        """A STABLE key that keeps missing is a stale entry (TTL / changed file
        dep), not key instability — the legitimate recompute-and-recache
        workflow. It must never accumulate churn."""
        guard = MissGuard(None)
        for _ in range(N_CHURN * 4):
            guard.observe("src", "same-key", hit=False)
        assert guard.is_guarded("src") is False
        assert guard.should_serialise("src") is True

    def test_edited_source_starts_from_zero_evidence(self):
        """Verdicts are keyed by source hash: an edit is a different statement."""
        guard = MissGuard(None)
        keys = (f"key-{i}" for i in itertools.count())
        for _ in range(N_CHURN + 1):
            guard.observe("src-v1", next(keys), hit=False)
        assert guard.is_guarded("src-v1") is True
        assert guard.should_serialise("src-v2") is True


def _drive_to_guarded(guard: MissGuard, keys) -> None:
    for _ in range(N_CHURN + 1):
        guard.observe("src", next(keys), hit=False)
    assert guard.is_guarded("src") is True


class TestReprobe:
    """A guard with no escape hatch is a new bug, not a fix.

    Once we stop writing, a key that later stabilises has no entry to hit — so
    without a re-probe the guard could permanently condemn a statement that would
    have cached fine.
    """

    def test_periodic_reprobe_allows_exactly_one_write_every_r_runs(self):
        """The run that fires the guard is guarded run #1 (its write is the
        first one suppressed), so probes land on guarded runs #R, #2R, ..."""
        guard = MissGuard(None)
        keys = (f"key-{i}" for i in itertools.count())
        _drive_to_guarded(guard, keys)
        assert guard.should_serialise("src") is False, "guarded run #1 must not write"

        probes = []
        for guarded_run in range(2, N_PROBE * 3 + 1):
            guard.observe("src", next(keys), hit=False)
            if guard.should_serialise("src"):
                probes.append(guarded_run)
        assert probes == [N_PROBE, N_PROBE * 2, N_PROBE * 3], (
            f"expected a probe every {N_PROBE} guarded runs, got {probes}"
        )

    def test_key_repeat_probes_immediately(self):
        """A guarded statement whose key equals last run's has visibly
        stabilised — probe at once rather than waiting out the blind interval."""
        guard = MissGuard(None)
        keys = (f"key-{i}" for i in itertools.count())
        _drive_to_guarded(guard, keys)

        guard.observe("src", "stable", hit=False)   # first stable run: still churn
        assert guard.should_serialise("src") is False
        guard.observe("src", "stable", hit=False)   # key repeats -> probe
        assert guard.should_serialise("src") is True

    def test_probe_write_leads_to_a_hit_which_unguards(self):
        guard = MissGuard(None)
        keys = (f"key-{i}" for i in itertools.count())
        _drive_to_guarded(guard, keys)

        guard.observe("src", "stable", hit=False)
        guard.observe("src", "stable", hit=False)   # probe writes the entry
        assert guard.should_serialise("src") is True
        guard.observe("src", "stable", hit=True)    # the write now pays off
        assert guard.is_guarded("src") is False
        assert guard.should_serialise("src") is True


class TestVerdictPersistence:
    def test_verdict_survives_a_new_instance(self, tmp_path):
        guard = MissGuard(str(tmp_path))
        keys = (f"key-{i}" for i in itertools.count())
        _drive_to_guarded(guard, keys)

        reloaded = MissGuard(str(tmp_path))
        assert reloaded.is_guarded("src") is True
        assert reloaded.should_serialise("src") is False

    def test_persists_only_when_the_verdict_flips_never_per_cell(self, tmp_path):
        """CAS-149 removed an 8-12 ms/cell fsync. This must not reintroduce one
        under a new name: the hot path stays in memory."""
        guard = MissGuard(str(tmp_path))
        writes = []
        real_persist = guard._persist

        def counting_persist():
            writes.append(1)
            return real_persist()

        guard._persist = counting_persist  # type: ignore[method-assign]

        keys = (f"key-{i}" for i in itertools.count())
        for _ in range(N_CHURN * 10):
            guard.observe("src", next(keys), hit=False)

        assert guard.is_guarded("src") is True
        assert len(writes) == 1, (
            f"verdict store written {len(writes)}x over {N_CHURN * 10} runs; "
            "it must be written only when the verdict flips"
        )

    def test_unguarding_flips_the_verdict_back_on_disk(self, tmp_path):
        guard = MissGuard(str(tmp_path))
        keys = (f"key-{i}" for i in itertools.count())
        _drive_to_guarded(guard, keys)
        assert json.loads((tmp_path / "_miss_guard.json").read_text())["guarded"] == ["src"]

        guard.observe("src", "stable", hit=True)
        assert json.loads((tmp_path / "_miss_guard.json").read_text())["guarded"] == []
        assert MissGuard(str(tmp_path)).is_guarded("src") is False

    def test_a_clean_notebook_never_writes_a_store(self, tmp_path):
        guard = MissGuard(str(tmp_path))
        for _ in range(50):
            guard.observe("src", "stable-key", hit=True)
        assert not (tmp_path / "_miss_guard.json").exists()

    @pytest.mark.parametrize(
        "content", ["not json at all", '{"version": 999, "guarded": ["src"]}', "[]"]
    )
    def test_unreadable_store_degrades_to_no_guard(self, tmp_path, content):
        """The guard is an optimisation. Its failure mode must be 'no
        optimisation', never 'no cache'."""
        (tmp_path / "_miss_guard.json").write_text(content)
        guard = MissGuard(str(tmp_path))
        assert guard.is_guarded("src") is False
        assert guard.should_serialise("src") is True

    def test_no_cache_dir_is_session_scoped_not_a_crash(self):
        guard = MissGuard(None)
        keys = (f"key-{i}" for i in itertools.count())
        _drive_to_guarded(guard, keys)  # must not raise
        assert MissGuard(None).is_guarded("src") is False


class TestResolveCacheDir:
    def test_finds_dir_through_a_tier_chain(self, tmp_path):
        from cash.backends import InMemoryBackend
        from cash.backends.tiered_backend import TieredBackend

        tiered = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path))])
        assert resolve_cache_dir(tiered) == os.path.abspath(str(tmp_path))

    def test_pure_memory_backend_has_nowhere_to_persist(self):
        from cash.backends import InMemoryBackend

        assert resolve_cache_dir(InMemoryBackend()) is None

    def test_a_mock_backend_does_not_masquerade_as_a_path(self):
        """A MagicMock answers every getattr with another mock; a good number of
        tests pass one as the cash instance."""
        assert resolve_cache_dir(MagicMock()) is None
        assert resolve_cache_dir(None) is None


# ---------------------------------------------------------------------------
# End-to-end through CashMagics + a real FileBackend.
# ---------------------------------------------------------------------------


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


CODE = "out = sum(big) * 2"


class _Session:
    """One kernel session against a cache dir. Re-making one on the same dir is
    the kernel restart."""

    def __init__(self, cache_dir: str):
        self.backend = FileBackend(cache_dir=cache_dir)
        self.cash = Cash(backend=self.backend, register_magic=False)
        # Isolate the guard from the unrelated cost-aware floor: these statements
        # are trivially cheap, and "too cheap to cache" would refuse the write
        # before the guard ever got a vote.
        self.cash.config.min_execution_time_to_cache_seconds = 0.0
        self.shell = _MockShell()
        self.magics = CashMagics(self.shell, self.cash)
        self.magics._auto_cache_enabled = True
        self.metrics: list[dict] = []
        self.magics._render_interactive_badge = (
            lambda metrics, **kw: self.metrics.extend(metrics)
        )

        # Every backend.set is one serialisation of the value — the cost the
        # guard exists to stop paying.
        self.serialised: list[str] = []
        real_set = self.backend.set

        def counting_set(key, value, metadata=None, *args, **kwargs):
            self.serialised.append(key)
            return real_set(key, value, metadata, *args, **kwargs)

        self.backend.set = counting_set  # type: ignore[method-assign]

    def run(self, lineage: str, code: str = CODE) -> dict:
        self.shell.user_ns["big"] = list(range(1000))
        self.magics._tracking_state.variable_lineage["big"] = lineage
        self.metrics.clear()
        self.magics.cash("", code)
        assert self.metrics, "no metrics captured"
        return self.metrics[-1]

    def close(self):
        self.backend.shutdown()


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    return str(d)


@pytest.fixture
def session(cache_dir):
    s = _Session(cache_dir)
    yield s
    s.close()


def _unstable():
    return (f"unstable-lineage-{i}" for i in itertools.count())


class TestUnstableKeyStopsSerialising:
    def test_serialisation_stops_after_n_churn_misses(self, session):
        keys = _unstable()
        for _ in range(N_CHURN + 1):
            session.run(next(keys))
        assert len(session.serialised) == N_CHURN, (
            "the cold run plus N-1 churn runs should serialise, then the guard fires"
        )

        # The guard fired on the run above (guarded run #1). It must stay silent
        # right up to the re-probe, and the re-probe must then actually fire —
        # a guard with no escape hatch is a new bug.
        before = len(session.serialised)
        for _ in range(N_PROBE - 2):  # guarded runs #2 .. #(R-1)
            session.run(next(keys))
        assert len(session.serialised) == before, (
            "guarded statement serialised before the re-probe was due"
        )
        session.run(next(keys))  # guarded run #R
        assert len(session.serialised) == before + 1, (
            "the periodic re-probe never let a write through"
        )

    def test_the_lookup_keeps_running_while_guarded(self, session):
        """Stop writing, keep hashing and keep looking up. The lookup is cheap,
        and it is the ONLY way a statement whose key later stabilises can notice
        an entry it could hit. Dropping it would make the guard terminal."""
        keys = _unstable()
        for _ in range(N_CHURN + 1):
            m = session.run(next(keys))
        assert m.get("skipped_reason"), "precondition: the guard should have fired"

        lookups = []
        real_check = session.magics._statement_processor._freshness.check_cache

        def counting_check(*args, **kwargs):
            lookups.append(1)
            return real_check(*args, **kwargs)

        session.magics._statement_processor._freshness.check_cache = counting_check
        serialised_before = len(session.serialised)
        for _ in range(3):
            session.run(next(keys))

        assert len(lookups) == 3, "a guarded statement must still be looked up"
        assert len(session.serialised) == serialised_before, (
            "...but must not be serialised"
        )

    def test_badge_says_the_statement_stopped_caching_and_why(self, session):
        keys = _unstable()
        for _ in range(N_CHURN + 1):
            m = session.run(next(keys))
        reason = m.get("skipped_reason")
        assert reason, "a guarded statement must tell the user it stopped caching"
        assert "Perpetual cache miss" in reason
        assert "different cache key" in reason  # the WHY: the key churns and never hits

    def test_persist_annotation_outranks_the_guard(self, session):
        """The guard is a default, not a veto: a user who explicitly asks for
        persistence gets it."""
        code = "# @cash:persist\nout = sum(big) * 2"
        keys = _unstable()
        for _ in range(N_CHURN * 3):
            session.run(next(keys), code=code)
        assert len(session.serialised) == N_CHURN * 3

    def test_verdict_store_is_not_written_per_cell(self, session):
        """CAS-149's fsync-per-cell must not come back under a new name."""
        writes = []
        guard = session.magics._statement_processor._miss_guard
        real_persist = guard._persist

        def counting_persist():
            writes.append(1)
            return real_persist()

        guard._persist = counting_persist  # type: ignore[method-assign]

        keys = _unstable()
        for _ in range(N_CHURN * 4):
            session.run(next(keys))
        assert len(writes) == 1, f"verdict persisted {len(writes)}x; expected 1 (the flip)"


class TestVerdictSurvivesKernelRestart:
    def test_a_restart_does_not_re_pay_the_learning(self, cache_dir):
        keys = _unstable()
        first = _Session(cache_dir)
        try:
            for _ in range(N_CHURN + 1):
                first.run(next(keys))
            assert len(first.serialised) == N_CHURN
        finally:
            first.close()

        assert os.path.exists(os.path.join(cache_dir, "_miss_guard.json"))

        # Kernel restart: brand-new shell, magics, processor and backend on the
        # same cache dir. Without persistence this session would re-pay all N
        # serialisations, and so would every restart after it.
        second = _Session(cache_dir)
        try:
            for _ in range(N_CHURN - 1):
                second.run(next(keys))
            assert second.serialised == [], (
                "the restarted kernel re-paid the serialisation the previous "
                "session already proved worthless"
            )
        finally:
            second.close()


class TestGuardedStatementCanRecover:
    def test_a_key_that_stabilises_recovers_on_its_own(self, session):
        """If a guard cannot be escaped it permanently condemns a statement that
        would eventually have cached fine."""
        keys = _unstable()
        for _ in range(N_CHURN + 1):
            m = session.run(next(keys))
        # Precondition: without this the test would pass on an un-guarded build,
        # where "recovery" is just the statement never having stopped caching.
        assert m.get("skipped_reason"), "precondition: the guard should have fired"
        assert len(session.serialised) == N_CHURN
        guarded_at = len(session.serialised)

        # The key stabilises (a seed gets pinned, a frame drops under the
        # sampling threshold, ...). Nothing tells the guard; it must notice.
        statuses = [session.run("now-stable")["status"] for _ in range(4)]

        assert len(session.serialised) == guarded_at + 1, (
            "the re-probe should have let exactly one write through"
        )
        assert CacheStatus.RESTORED in statuses, (
            f"a stabilised key never recovered to a cache hit: {statuses}"
        )
        assert session.run("now-stable").get("skipped_reason") is None, (
            "the statement recovered but is still badged as guarded"
        )


class TestRestartSaverIsNeverGuarded:
    """The regression test that protects the product's core value.

    An expensive statement with a STABLE key is net-negative in-session on its
    cold run and pays that back across a restart. It must never be guarded, no
    matter how many sessions it lives through.
    """

    def test_stable_key_statement_hits_across_restarts_and_is_never_guarded(
        self, cache_dir
    ):
        sessions_serialised = []
        for _ in range(N_CHURN + 3):  # more restarts than the guard's threshold
            s = _Session(cache_dir)
            try:
                statuses = [s.run("stable-lineage")["status"] for _ in range(3)]
                sessions_serialised.append(len(s.serialised))
                assert statuses[-1] == CacheStatus.RESTORED, (
                    f"a stable-key statement stopped hitting: {statuses}"
                )
                guard = s.magics._statement_processor._miss_guard
                assert not any(
                    guard.is_guarded(h) for h in guard._records
                ), "a legitimate restart-saver was guarded"
            finally:
                s.close()

        assert sessions_serialised[0] == 1, "the cold run must serialise once"
        assert sessions_serialised[1:] == [0] * (len(sessions_serialised) - 1), (
            "a warm restart-saver should hit from disk and never re-serialise"
        )
        assert not os.path.exists(os.path.join(cache_dir, "_miss_guard.json")), (
            "a healthy notebook must not write a verdict store at all"
        )

    def test_an_edited_upstream_between_runs_is_not_mistaken_for_churn(self, cache_dir):
        """Editing an upstream cell then re-running is normal work, not the
        perpetual-miss signature: the re-run repeats the key and hits, which
        zeroes the evidence long before the guard could fire."""
        s = _Session(cache_dir)
        try:
            for edit in range(N_CHURN * 3):
                s.run(f"edit-{edit}")          # upstream changed -> miss, cache it
                assert s.run(f"edit-{edit}")["status"] == CacheStatus.RESTORED
            guard = s.magics._statement_processor._miss_guard
            assert not any(guard.is_guarded(h) for h in guard._records)
        finally:
            s.close()
