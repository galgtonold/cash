"""
Tests for Cash structured logging module.
"""

import json
import logging
import os

from cash._log import JsonFormatter, setup_logging


class TestJsonFormatter:
    """Tests for JSON log formatting."""

    def test_basic_format(self):
        fmt = JsonFormatter()
        record = logging.LogRecord(
            name="cash.test", level=logging.INFO, pathname="", lineno=0, msg="test message", args=(), exc_info=None
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
            name="cash", level=logging.DEBUG, pathname="", lineno=0, msg="cache hit", args=(), exc_info=None
        )
        record.event = "cache_hit"
        record.duration_ms = 1.5
        record.cache_key = "stmt:abc123"
        result = fmt.format(record)
        data = json.loads(result)
        assert data["event"] == "cache_hit"
        assert data["duration_ms"] == 1.5
        assert data["cache_key"] == "stmt:abc123"


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_basic_setup(self):
        setup_logging(level=logging.DEBUG)
        cash_logger = logging.getLogger("cash")
        assert cash_logger.level == logging.DEBUG

    def test_json_output(self):
        setup_logging(level=logging.INFO, json_output=True)
        cash_logger = logging.getLogger("cash")
        # Should have a handler with JsonFormatter
        json_handlers = [h for h in cash_logger.handlers if isinstance(h.formatter, JsonFormatter)]
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
