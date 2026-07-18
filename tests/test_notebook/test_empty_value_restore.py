"""A legitimately-empty cached value must be restorable (CAS-101).

``_restore_vars_from_cache`` refused to restore ANY empty sized value whenever
the namespace happened to hold a non-empty one. The intent was sound — an empty
cache entry should not clobber live data — but the rule was unconditional, so a
filter that correctly matches nothing could never be served from cache and
re-executed on every run, forever.

The discriminator is lineage. When the cached lineage hash matches the expected
one, the empty value IS the correct current result and restoring it is right.
When lineage cannot be confirmed (file-dependency restores, or no expected
lineages supplied) an empty value is indistinguishable from a corrupt entry, and
the conservative refusal stands.

The pre-existing guard test in ``test_issue_fixes.py`` covers the unconfirmed
direction and passes UNCHANGED — it supplies no expected lineages, so nothing is
confirmed. These tests cover the confirmed direction and the boundary.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cash.notebook.upstream import UpstreamChecker


def _make_lineage(shell_ns: dict):
    cash_instance = MagicMock()
    shell = MagicMock()
    shell.user_ns = shell_ns
    checker = UpstreamChecker(shell, cash_instance, debug=False)
    checker.variable_lineage = {}
    return checker.simulator._virtual_lineage, shell, cash_instance


class TestLineageConfirmedVars:
    """The set that decides whether the empty-guard applies."""

    def test_confirmed_when_hash_matches(self):
        vl, _, _ = _make_lineage({})
        confirmed = vl._lineage_confirmed_vars(
            {'output_lineages': {'rows': 'h1'}}, {}, {'rows': 'h1'}
        )
        assert confirmed == frozenset({'rows'})

    def test_not_confirmed_when_hash_differs(self):
        vl, _, _ = _make_lineage({})
        confirmed = vl._lineage_confirmed_vars(
            {'output_lineages': {'rows': 'h1'}}, {}, {'rows': 'DIFFERENT'}
        )
        assert confirmed == frozenset()

    def test_not_confirmed_when_file_deps_present(self):
        """File-dep restores skip the strict lineage check, so nothing is proven."""
        vl, _, _ = _make_lineage({})
        confirmed = vl._lineage_confirmed_vars(
            {'output_lineages': {'rows': 'h1'}}, {'data.csv': 123.0}, {'rows': 'h1'}
        )
        assert confirmed == frozenset()

    @pytest.mark.parametrize(
        "metadata, expected",
        [
            ({}, {'rows': 'h1'}),                        # no output_lineages
            ({'output_lineages': {'rows': 'h1'}}, None),  # no expected lineages
            ({'output_lineages': {}}, {'rows': 'h1'}),    # var absent from cache
        ],
    )
    def test_not_confirmed_without_both_sides(self, metadata, expected):
        vl, _, _ = _make_lineage({})
        assert vl._lineage_confirmed_vars(metadata, {}, expected) == frozenset()


class TestEmptyRestoreRespectsConfirmation:
    """The guard itself, driven directly."""

    METADATA = {'output_lineages': {'rows': 'h1'}}

    def test_confirmed_empty_value_is_restored(self):
        """The CAS-101 fix: a correctly-empty result reaches the namespace."""
        vl, shell, _ = _make_lineage({'rows': [1, 2, 3]})

        restored = vl._restore_vars_from_cache(
            {'rows': []}, self.METADATA, frozenset({'rows'})
        )

        assert 'rows' in restored
        assert shell.user_ns['rows'] == []

    def test_unconfirmed_empty_value_is_blocked(self):
        """The original safety property, unchanged."""
        vl, shell, _ = _make_lineage({'rows': [1, 2, 3]})

        restored = vl._restore_vars_from_cache({'rows': []}, self.METADATA)

        assert 'rows' not in restored
        assert shell.user_ns['rows'] == [1, 2, 3], "live data must not be clobbered"

    def test_confirmation_only_affects_the_empty_case(self):
        """A non-empty cached value restores either way — confirmation is not
        a general gate on restoring, only on the empty-over-non-empty case."""
        for confirmed in (frozenset(), frozenset({'rows'})):
            vl, shell, _ = _make_lineage({'rows': [1, 2, 3]})
            restored = vl._restore_vars_from_cache(
                {'rows': [9, 9]}, self.METADATA, confirmed
            )
            assert 'rows' in restored
            assert shell.user_ns['rows'] == [9, 9]

    def test_empty_over_empty_restores_without_confirmation(self):
        """The guard needs a NON-empty incumbent; empty-over-empty is harmless."""
        vl, shell, _ = _make_lineage({'rows': []})

        restored = vl._restore_vars_from_cache({'rows': []}, self.METADATA)

        assert 'rows' in restored

    def test_var_absent_from_namespace_restores(self):
        """Nothing to protect when the name is not bound yet."""
        vl, shell, _ = _make_lineage({})

        restored = vl._restore_vars_from_cache({'rows': []}, self.METADATA)

        assert 'rows' in restored
        assert shell.user_ns['rows'] == []

    def test_unsized_values_are_unaffected(self):
        """Scalars have no len(); the guard must not choke on them."""
        vl, shell, _ = _make_lineage({'x': 42})

        restored = vl._restore_vars_from_cache(
            {'x': 0}, {'output_lineages': {'x': 'h1'}}
        )

        assert 'x' in restored
        assert shell.user_ns['x'] == 0


class TestEndToEndThroughVirtualRestore:
    """Through the real entry point, so the plumbing is covered too."""

    def test_confirmed_lineage_restores_empty_result(self):
        vl, shell, cash_instance = _make_lineage({'rows': list(range(1000))})
        cash_instance.backend.get.return_value = (
            {'output_lineages': {'rows': 'h1'}, 'execution_time': 1.0},
            {'variables': {'rows': []}},
        )

        restored, _, _ = vl._try_virtual_restore(
            "rows = [r for r in data if r.matches(q)]",
            {'rows'}, {'data', 'q'}, {},
            expected_lineages={'rows': 'h1'},
        )

        assert 'rows' in restored, "confirmed-empty result should restore"
        assert shell.user_ns['rows'] == []

    def test_mismatched_lineage_does_not_restore(self):
        vl, shell, cash_instance = _make_lineage({'rows': list(range(1000))})
        cash_instance.backend.get.return_value = (
            {'output_lineages': {'rows': 'h1'}, 'execution_time': 1.0},
            {'variables': {'rows': []}},
        )

        restored, _, _ = vl._try_virtual_restore(
            "rows = [r for r in data if r.matches(q)]",
            {'rows'}, {'data', 'q'}, {},
            expected_lineages={'rows': 'STALE'},
        )

        assert 'rows' not in restored
        assert len(shell.user_ns['rows']) == 1000
