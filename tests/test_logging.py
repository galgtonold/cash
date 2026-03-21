"""
Tests for Cash structured logging module.
"""

import json
import logging
import os

from cash.logging import JsonFormatter, CashLogHandler, setup_logging


class TestJsonFormatter:
    """Tests for JSON log formatting."""

    def test_basic_format(self):
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="cash.test", level=logging.INFO,
            pathname="", lineno=0, msg="test message",
            args=(), exc_info=None
        )
        result = fmt.format(record)
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["message"] == "test message"
        assert data["logger"] == "cash.test"
        assert "timestamp" in data

    def test_extra_fields(self):
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="cash", level=logging.DEBUG,
            pathname="", lineno=0, msg="cache hit",
            args=(), exc_info=None
        )
        record.event = "cache_hit"
        record.duration_ms = 1.5
        record.cache_key = "stmt:abc123"
        result = fmt.format(record)
        data = json.loads(result)
        assert data["event"] == "cache_hit"
        assert data["duration_ms"] == 1.5
        assert data["cache_key"] == "stmt:abc123"


class TestCashLogHandler:
    """Tests for in-memory log handler."""

    def test_emit_and_retrieve(self):
        handler = CashLogHandler()
        record = logging.LogRecord(
            name="cash", level=logging.INFO,
            pathname="", lineno=0, msg="test",
            args=(), exc_info=None
        )
        handler.emit(record)
        events = handler.get_events()
        assert len(events) == 1
        assert events[0]["msg"] == "test"

    def test_max_entries(self):
        handler = CashLogHandler()
        handler.MAX_ENTRIES = 10
        for i in range(20):
            record = logging.LogRecord(
                name="cash", level=logging.INFO,
                pathname="", lineno=0, msg=f"msg-{i}",
                args=(), exc_info=None
            )
            handler.emit(record)
        events = handler.get_events(limit=100)
        assert len(events) == 10
        # Should have the last 10 messages
        assert events[0]["msg"] == "msg-10"
        assert events[-1]["msg"] == "msg-19"

    def test_filter_by_event(self):
        handler = CashLogHandler()
        for event_type in ["cache_hit", "cache_miss", "cache_hit"]:
            record = logging.LogRecord(
                name="cash", level=logging.INFO,
                pathname="", lineno=0, msg=event_type,
                args=(), exc_info=None
            )
            record.event = event_type
            handler.emit(record)
        hits = handler.get_events(event_type="cache_hit")
        assert len(hits) == 2

    def test_clear(self):
        handler = CashLogHandler()
        record = logging.LogRecord(
            name="cash", level=logging.INFO,
            pathname="", lineno=0, msg="test",
            args=(), exc_info=None
        )
        handler.emit(record)
        assert len(handler.get_events()) == 1
        handler.clear()
        assert len(handler.get_events()) == 0

    def test_limit(self):
        handler = CashLogHandler()
        for i in range(10):
            record = logging.LogRecord(
                name="cash", level=logging.INFO,
                pathname="", lineno=0, msg=f"msg-{i}",
                args=(), exc_info=None
            )
            handler.emit(record)
        events = handler.get_events(limit=3)
        assert len(events) == 3
        assert events[0]["msg"] == "msg-7"


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_basic_setup(self):
        handler = setup_logging(level=logging.DEBUG)
        assert isinstance(handler, CashLogHandler)
        cash_logger = logging.getLogger("cash")
        assert cash_logger.level == logging.DEBUG

    def test_json_output(self):
        setup_logging(level=logging.INFO, json_output=True)
        cash_logger = logging.getLogger("cash")
        # Should have a handler with JsonFormatter
        json_handlers = [
            h for h in cash_logger.handlers
            if isinstance(h.formatter, JsonFormatter)
        ]
        assert len(json_handlers) > 0

    def test_file_logging(self, tmp_path):
        log_file = str(tmp_path / "cash.log")
        setup_logging(level=logging.DEBUG, log_file=log_file)
        cash_logger = logging.getLogger("cash")
        cash_logger.info("test file logging")
        # Flush handlers
        for h in cash_logger.handlers:
            h.flush()
        assert os.path.isfile(log_file)
        with open(log_file) as f:
            content = f.read()
        assert "test file logging" in content
        # Should be valid JSON per line
        data = json.loads(content.strip())
        assert data["message"] == "test file logging"

    def test_no_duplicate_handlers(self):
        """Calling setup_logging twice should not create duplicate handlers."""
        setup_logging(level=logging.INFO)
        handler_count_1 = len(logging.getLogger("cash").handlers)
        setup_logging(level=logging.INFO)
        handler_count_2 = len(logging.getLogger("cash").handlers)
        assert handler_count_2 == handler_count_1
