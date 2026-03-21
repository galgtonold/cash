"""Tests for audit logging module."""
import json
import time
from cash.notebook.audit import AuditLogger, AuditEntry


class TestAuditEntry:
    """Test AuditEntry dataclass."""

    def test_entry_creation(self):
        entry = AuditEntry(
            timestamp=time.time(),
            operation='cache_hit',
            variable='df',
            code='df = pd.read_csv("data.csv")',
            status='success',
            duration_ms=12.5,
        )
        assert entry.operation == 'cache_hit'
        assert entry.variable == 'df'
        assert entry.duration_ms == 12.5

    def test_entry_to_dict(self):
        entry = AuditEntry(
            timestamp=1700000000.0,
            operation='cache_miss',
            variable='x',
        )
        d = entry.to_dict()
        assert d['operation'] == 'cache_miss'
        assert d['variable'] == 'x'
        assert 'timestamp_str' in d

    def test_entry_to_json(self):
        entry = AuditEntry(
            timestamp=1700000000.0,
            operation='cache_hit',
            variable='y',
            code='y = x + 1',
        )
        j = entry.to_json()
        data = json.loads(j)
        assert data['operation'] == 'cache_hit'
        assert data['variable'] == 'y'

    def test_timestamp_str_format(self):
        entry = AuditEntry(timestamp=1700000000.0, operation='test', variable='v')
        ts = entry.timestamp_str
        assert len(ts) > 10  # Should be a formatted datetime string


class TestAuditLogger:
    """Test AuditLogger class."""

    def test_disabled_by_default(self):
        logger = AuditLogger()
        assert not logger.enabled

    def test_enable_disable(self):
        logger = AuditLogger()
        logger.enable()
        assert logger.enabled
        logger.disable()
        assert not logger.enabled

    def test_log_when_disabled_does_nothing(self):
        logger = AuditLogger()
        logger.log('cache_hit', 'x')
        assert len(logger.get_entries()) == 0

    def test_log_when_enabled(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x', code='x = 1', duration_ms=5.0)
        entries = logger.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == 'cache_hit'
        assert entries[0].variable == 'x'

    def test_log_multiple_entries(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x')
        logger.log('cache_miss', 'y')
        logger.log('cache_skip', 'z')
        entries = logger.get_entries()
        assert len(entries) == 3

    def test_filter_by_operation(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x')
        logger.log('cache_miss', 'y')
        logger.log('cache_hit', 'z')
        hits = logger.get_entries(operation='cache_hit')
        assert len(hits) == 2

    def test_filter_by_variable(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x')
        logger.log('cache_miss', 'x')
        logger.log('cache_hit', 'y')
        x_entries = logger.get_entries(variable='x')
        assert len(x_entries) == 2

    def test_limit_entries(self):
        logger = AuditLogger()
        logger.enable()
        for i in range(100):
            logger.log('cache_hit', f'var_{i}')
        entries = logger.get_entries(limit=10)
        assert len(entries) == 10

    def test_max_entries_eviction(self):
        logger = AuditLogger(max_entries=10)
        logger.enable()
        for i in range(20):
            logger.log('cache_hit', f'var_{i}')
        # Should only keep the last 10
        entries = logger.get_entries(limit=100)
        assert len(entries) == 10
        assert entries[0].variable == 'var_10'

    def test_clear(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x')
        logger.clear()
        assert len(logger.get_entries()) == 0

    def test_get_summary_empty(self):
        logger = AuditLogger()
        summary = logger.get_summary()
        assert summary['total'] == 0

    def test_get_summary(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x')
        logger.log('cache_miss', 'y')
        logger.log('cache_hit', 'z')
        summary = logger.get_summary()
        assert summary['total'] == 3
        assert summary['operations']['cache_hit'] == 2
        assert summary['operations']['cache_miss'] == 1
        assert summary['unique_variables'] == 3
        assert 'time_range' in summary

    def test_format_entries_empty(self):
        logger = AuditLogger()
        result = logger.format_entries([])
        assert result == "No audit entries."

    def test_format_entries_text(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'df', code='df = pd.read_csv("data.csv")', duration_ms=45.2)
        entries = logger.get_entries()
        text = logger.format_entries(entries)
        assert 'cache_hit' in text
        assert 'df' in text
        assert '45.2ms' in text

    def test_format_entries_json(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x')
        entries = logger.get_entries()
        j = logger.format_entries(entries, as_json=True)
        data = json.loads(j)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['operation'] == 'cache_hit'

    def test_file_logging(self, tmp_path):
        log_file = tmp_path / "audit.log"
        logger = AuditLogger()
        logger.enable(str(log_file))
        logger.log('cache_hit', 'x', code='x = 1')
        logger.log('cache_miss', 'y', code='y = 2')
        logger.disable()

        # Verify file was written
        content = log_file.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data['operation'] == 'cache_hit'

    def test_shutdown(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('test', 'x')
        logger.shutdown()
        assert not logger.enabled

    def test_code_truncation(self):
        logger = AuditLogger()
        logger.enable()
        long_code = "x = " + "a" * 300
        logger.log('cache_hit', 'x', code=long_code)
        entries = logger.get_entries()
        assert len(entries[0].code) <= 200

    def test_extra_details(self):
        logger = AuditLogger()
        logger.enable()
        logger.log('cache_hit', 'x', cache_key='abc123', backend='FileBackend')
        entries = logger.get_entries()
        assert entries[0].details['cache_key'] == 'abc123'
        assert entries[0].details['backend'] == 'FileBackend'
