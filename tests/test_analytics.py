"""Tests for the AnalyticsManager (analytics.py)."""
import os
from cash.analytics import AnalyticsManager


class TestAnalyticsManager:
    """Test AnalyticsManager SQLite-based analytics."""

    def test_init_creates_db(self, tmp_path):
        """AnalyticsManager creates SQLite database."""
        db_path = str(tmp_path / "analytics.db")
        AnalyticsManager(db_path=db_path)
        assert os.path.exists(db_path)

    def test_init_default_path(self):
        """AnalyticsManager uses default path when none given."""
        am = AnalyticsManager()
        assert am.db_path.endswith("analytics.db")

    def test_session_id_generated(self, tmp_path):
        """Each instance gets a unique session ID."""
        db_path = str(tmp_path / "analytics.db")
        am1 = AnalyticsManager(db_path=db_path)
        am2 = AnalyticsManager(db_path=db_path)
        assert am1.session_id != am2.session_id

    def test_record_event(self, tmp_path):
        """Record a single event."""
        db_path = str(tmp_path / "analytics.db")
        am = AnalyticsManager(db_path=db_path)
        am.record_event("HIT", 0.001, saved_time=1.5, code_hash="abc123")
        stats = am.get_session_stats()
        assert stats['total_events'] == 1
        assert stats['hits'] == 1
        assert stats['total_saved_time'] == 1.5

    def test_record_multiple_events(self, tmp_path):
        """Record multiple events and check counts."""
        db_path = str(tmp_path / "analytics.db")
        am = AnalyticsManager(db_path=db_path)
        am.record_event("HIT", 0.001, saved_time=1.0)
        am.record_event("HIT", 0.001, saved_time=2.0)
        am.record_event("MISS", 1.0)
        am.record_event("EXECUTION", 0.5)

        stats = am.get_session_stats()
        assert stats['total_events'] == 4
        assert stats['hits'] == 2
        assert stats['misses'] == 2
        assert stats['total_saved_time'] == 3.0

    def test_session_isolation(self, tmp_path):
        """Different sessions have separate stats."""
        db_path = str(tmp_path / "analytics.db")
        am1 = AnalyticsManager(db_path=db_path)
        am1.record_event("HIT", 0.001, saved_time=1.0)

        am2 = AnalyticsManager(db_path=db_path)
        am2.record_event("MISS", 0.5)

        stats1 = am1.get_session_stats()
        stats2 = am2.get_session_stats()
        assert stats1['hits'] == 1
        assert stats2['hits'] == 0

    def test_get_global_stats(self, tmp_path):
        """Global stats aggregate across sessions."""
        db_path = str(tmp_path / "analytics.db")
        am1 = AnalyticsManager(db_path=db_path)
        am1.record_event("HIT", 0.001, saved_time=1.0)
        am1.flush()  # Ensure am1's events are persisted before am2 queries

        am2 = AnalyticsManager(db_path=db_path)
        am2.record_event("MISS", 0.5)

        global_stats = am2.get_global_stats()
        assert global_stats['total_sessions'] == 2
        assert global_stats['total_events'] == 2

    def test_get_daily_savings(self, tmp_path):
        """Daily savings aggregation returns data."""
        db_path = str(tmp_path / "analytics.db")
        am = AnalyticsManager(db_path=db_path)
        am.record_event("HIT", 0.001, saved_time=5.0)
        am.record_event("HIT", 0.001, saved_time=3.0)

        daily = am.get_daily_savings(limit=7)
        assert len(daily) >= 1
        # First entry should have today's date
        total_saved = daily[0][1]
        assert total_saved == 8.0

    def test_empty_session_stats(self, tmp_path):
        """Empty session returns empty or zero stats."""
        db_path = str(tmp_path / "analytics.db")
        am = AnalyticsManager(db_path=db_path)
        stats = am.get_session_stats()
        # Could be empty dict or zero values
        assert stats.get('total_events', 0) == 0

    def test_get_stats_for_nonexistent_session(self, tmp_path):
        """Stats for unknown session return empty/zero."""
        db_path = str(tmp_path / "analytics.db")
        am = AnalyticsManager(db_path=db_path)
        stats = am.get_stats_for_session("nonexistent")
        assert stats.get('total_events', 0) == 0

    def test_db_path_creates_parent_directory(self, tmp_path):
        """If db_path directory doesn't exist, it's created."""
        db_path = str(tmp_path / "subdir" / "deep" / "analytics.db")
        AnalyticsManager(db_path=db_path)
        assert os.path.exists(db_path)
