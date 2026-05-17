"""Tests for storage column display in badges and TieredBackend storage propagation."""

import pytest
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.file_backend import FileBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.cache_status import CacheStatus


def _storage_html_for(metric_extras: dict) -> str:
    """Render a single COMPUTED metric to HTML and return that HTML for assertion.

    The current-cell storage cell is what these tests care about — the
    helper centralises the build/render boilerplate so each test reads as
    a single dict literal + assertion.
    """
    metric = {'code': 'x = 1', 'status': str(CacheStatus.COMPUTED), 'total_time': 0.05}
    metric.update(metric_extras)
    return render_html(build_interactive_badge([metric]))


class TestTieredBackendStoragePropagation:
    """Verify that TieredBackend propagates storage info back to the caller's metadata."""

    def test_storage_propagated_to_original_metadata(self, tmp_path):
        """The caller's original metadata dict should have storage set after .set()."""
        backend = TieredBackend([InMemoryBackend()])
        metadata = {'execution_time': 1.0}
        backend.set('key1', {'data': 'value'}, metadata)
        assert 'storage' in metadata
        assert 'RAM' in metadata['storage']

    def test_storage_propagated_tiered_ram_disk(self, tmp_path):
        """With RAM+DISK tiers and no promotion policy, both should appear."""
        l1 = InMemoryBackend()
        l2 = FileBackend(str(tmp_path / 'cache'))
        backend = TieredBackend([l1, l2])  # default policy: always promote
        metadata = {'execution_time': 1.0}
        backend.set('key2', {'data': 'value'}, metadata)
        assert 'storage' in metadata
        assert 'RAM' in metadata['storage']
        assert 'DISK' in metadata['storage']

    def test_storage_ram_only_when_not_promoted(self, tmp_path):
        """With a promotion policy that rejects, only RAM should appear."""
        l1 = InMemoryBackend()
        l2 = FileBackend(str(tmp_path / 'cache'))
        backend = TieredBackend([l1, l2], promotion_policy=lambda t, s: False)
        metadata = {'execution_time': 0.001}
        backend.set('key3', {'data': 'value'}, metadata)
        assert metadata['storage'] == ['RAM']

    def test_storage_propagated_when_metadata_is_none(self):
        """When metadata is None, the backend should still work without error."""
        backend = TieredBackend([InMemoryBackend()])
        # Should not raise
        backend.set('key4', {'data': 'value'}, None)

    def test_original_metadata_not_mutated_beyond_storage(self, tmp_path):
        """The copy should prevent sub-backends from polluting the original metadata
        with anything other than 'storage'."""
        l1 = InMemoryBackend()
        backend = TieredBackend([l1])
        original = {'execution_time': 1.0}
        original_keys_before = set(original.keys())
        backend.set('key5', {'data': 'value'}, original)
        # Only 'storage' should be added to the original
        new_keys = set(original.keys()) - original_keys_before
        assert new_keys == {'storage'}


class TestComputedStorageDisplay:
    """Storage cell rendering for COMPUTED rows, end-to-end through the BadgeView pipeline."""

    def test_uncacheable_reasons_shows_no_cache(self):
        html = _storage_html_for({'uncacheable_reasons': ['Side effect: print() (io)']})
        assert '🚫 No Cache' in html
        assert 'Side effect: print()' in html

    def test_skipped_reason_shows_not_cached(self):
        html = _storage_html_for({'skipped_reason': 'Object too large (500MB)'})
        assert '⚠️ Not Cached' in html
        assert 'Object too large' in html

    def test_storage_ram_shows_arrow(self):
        html = _storage_html_for({'storage': ['RAM']})
        assert '→ RAM' in html

    def test_storage_ram_disk_shows_both(self):
        html = _storage_html_for({'storage': ['RAM', 'DISK']})
        assert '→ RAM+DISK' in html

    def test_priority_uncacheable_over_storage(self):
        """uncacheable_reasons should take priority over storage values."""
        html = _storage_html_for({'uncacheable_reasons': ['mutation'], 'storage': ['RAM']})
        assert '🚫 No Cache' in html
        assert '→ RAM' not in html

    def test_empty_storage_no_outputs_shows_reason(self):
        """Statements with no variable outputs should show 'no outputs'."""
        html = _storage_html_for({'evaluated_vars': [], 'total_time': 0.05})
        assert 'no outputs' in html
        assert 'title=' in html

    def test_empty_storage_trivial_execution_shows_reason(self):
        """Fast statements with outputs should show 'trivial'."""
        html = _storage_html_for({'evaluated_vars': ['x'], 'total_time': 0.005})
        assert 'trivial' in html
        assert 'title=' in html

    def test_empty_storage_unknown_shows_dash(self):
        """Fallback: has outputs and non-trivial time but no storage."""
        html = _storage_html_for({'evaluated_vars': ['x'], 'total_time': 1.0})
        assert "style='cursor:help; color: #999;'>-</span>" in html
