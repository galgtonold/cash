"""A legitimately-empty cached value must be restorable.

A restore refused to restore ANY empty sized value whenever
the namespace happened to hold a non-empty one. The intent was sound — an empty
cache entry should not clobber live data — but the rule was unconditional, so a
filter that correctly matches nothing could never be served from cache and
re-executed on every run, forever.

The discriminator is lineage. When the cached lineage hash matches the expected
one, the empty value IS the correct current result and restoring it is right.
When lineage cannot be confirmed (file-dependency restores, or no expected
lineages supplied) an empty value is indistinguishable from a corrupt entry, and
the conservative refusal stands.

The pre-existing guard test in ``test_upstream_trusts_values_derived_from_loop_mutations.py`` covers the unconfirmed
direction and passes UNCHANGED — it supplies no expected lineages, so nothing is
confirmed. These tests cover the confirmed direction and the boundary.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cash.notebook.upstream import UpstreamChecker
from cash.notebook.upstream.cache_restore import lineage_confirmed_vars

METADATA = {"output_lineages": {"rows": "h1"}, "execution_time": 1.0}


def _make_simulator(shell_ns: dict, metadata=None, variables=None):
    cash_instance = MagicMock()
    cash_instance.backend.get.return_value = (metadata or METADATA, {"variables": variables or {}})
    shell = MagicMock()
    shell.user_ns = shell_ns
    checker = UpstreamChecker(shell, cash_instance)
    return checker.simulator, shell


def _restore(shell_ns, cached, confirmed, metadata=None):
    """Restore ``rows = ...`` whose entry holds *cached*; its lineage is
    confirmed when the simulation expects the lineage the entry has."""
    simulator, shell = _make_simulator(shell_ns, metadata, cached)
    expected = {name: "h1" for name in cached} if confirmed else None
    restored = simulator.restore_statement("rows = f()", set(cached), {"f"}, {}, expected_lineages=expected)
    return restored, shell


class TestLineageConfirmedVars:
    """The set that decides whether the empty-guard applies."""

    def test_confirmed_when_hash_matches(self):
        confirmed = lineage_confirmed_vars({"output_lineages": {"rows": "h1"}}, {}, {"rows": "h1"})
        assert confirmed == frozenset({"rows"})

    def test_not_confirmed_when_hash_differs(self):
        confirmed = lineage_confirmed_vars({"output_lineages": {"rows": "h1"}}, {}, {"rows": "DIFFERENT"})
        assert confirmed == frozenset()

    def test_not_confirmed_when_file_deps_present(self):
        """File-dep restores skip the strict lineage check, so nothing is proven."""
        confirmed = lineage_confirmed_vars({"output_lineages": {"rows": "h1"}}, {"data.csv": 123.0}, {"rows": "h1"})
        assert confirmed == frozenset()

    @pytest.mark.parametrize(
        "metadata, expected",
        [
            ({}, {"rows": "h1"}),  # no output_lineages
            ({"output_lineages": {"rows": "h1"}}, None),  # no expected lineages
            ({"output_lineages": {}}, {"rows": "h1"}),  # var absent from cache
        ],
    )
    def test_not_confirmed_without_both_sides(self, metadata, expected):
        assert lineage_confirmed_vars(metadata, {}, expected) == frozenset()


class TestEmptyRestoreRespectsConfirmation:
    """The guard itself, through a restore."""

    def test_confirmed_empty_value_is_restored(self):
        """A correctly-empty result reaches the namespace."""
        restored, shell = _restore({"rows": [1, 2, 3]}, {"rows": []}, confirmed=True)

        assert "rows" in restored
        assert shell.user_ns["rows"] == []

    def test_unconfirmed_empty_value_is_blocked(self):
        """The original safety property, unchanged."""
        restored, shell = _restore({"rows": [1, 2, 3]}, {"rows": []}, confirmed=False)

        assert "rows" not in restored
        assert shell.user_ns["rows"] == [1, 2, 3], "live data must not be clobbered"

    def test_confirmation_only_affects_the_empty_case(self):
        """A non-empty cached value restores either way — confirmation is not
        a general gate on restoring, only on the empty-over-non-empty case."""
        for confirmed in (False, True):
            restored, shell = _restore({"rows": [1, 2, 3]}, {"rows": [9, 9]}, confirmed=confirmed)
            assert "rows" in restored
            assert shell.user_ns["rows"] == [9, 9]

    def test_empty_over_empty_restores_without_confirmation(self):
        """The guard needs a NON-empty incumbent; empty-over-empty is harmless."""
        restored, _ = _restore({"rows": []}, {"rows": []}, confirmed=False)

        assert "rows" in restored

    def test_var_absent_from_namespace_restores(self):
        """Nothing to protect when the name is not bound yet."""
        restored, shell = _restore({}, {"rows": []}, confirmed=False)

        assert "rows" in restored
        assert shell.user_ns["rows"] == []

    def test_unsized_values_are_unaffected(self):
        """Scalars have no len(); the guard must not choke on them."""
        restored, shell = _restore({"x": 42}, {"x": 0}, confirmed=False, metadata={"output_lineages": {"x": "h1"}})

        assert "x" in restored
        assert shell.user_ns["x"] == 0


class TestEndToEndThroughVirtualRestore:
    """With the statement a real filter, as the backward scan restores it."""

    def test_confirmed_lineage_restores_empty_result(self):
        simulator, shell = _make_simulator({"rows": list(range(1000))}, variables={"rows": []})

        restored = simulator.restore_statement(
            "rows = [r for r in data if r.matches(q)]",
            {"rows"},
            {"data", "q"},
            {},
            expected_lineages={"rows": "h1"},
        )

        assert "rows" in restored, "confirmed-empty result should restore"
        assert shell.user_ns["rows"] == []

    def test_mismatched_lineage_does_not_restore(self):
        simulator, shell = _make_simulator({"rows": list(range(1000))}, variables={"rows": []})

        restored = simulator.restore_statement(
            "rows = [r for r in data if r.matches(q)]",
            {"rows"},
            {"data", "q"},
            {},
            expected_lineages={"rows": "STALE"},
        )

        assert "rows" not in restored
        assert len(shell.user_ns["rows"]) == 1000
