"""
Tests for Cash structured logging module.
"""

import json
import logging
import os
import sys

import pytest

from cash import _log
from cash._log import JsonFormatter, setup_logging


@pytest.fixture(autouse=True)
def _restore_cash_logger():
    """Leave the ``cash`` logger and cash's record of its handlers as found."""
    cash_logger = logging.getLogger("cash")
    handlers, level = list(cash_logger.handlers), cash_logger.level
    own, level_set = list(_log._OWN_HANDLERS), _log._LEVEL_SET
    yield
    cash_logger.handlers[:] = handlers
    cash_logger.setLevel(level)
    _log._OWN_HANDLERS[:] = own
    _log._LEVEL_SET = level_set


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


class TestOneConfiguration:
    """`enable` and `setup_logging` share one record of cash's own handlers."""

    def _clean(self):
        cash_logger = logging.getLogger("cash")
        for handler in list(_log._OWN_HANDLERS):
            cash_logger.removeHandler(handler)
        _log._OWN_HANDLERS.clear()
        cash_logger.setLevel(logging.NOTSET)
        cash_logger.propagate = True
        return cash_logger

    def test_enable_after_setup_logging_adds_no_second_console(self, monkeypatch):
        cash_logger = self._clean()
        monkeypatch.setattr(_log, "application_handlers", lambda logger: [])
        _log.enable(logging.DEBUG)
        assert [h for h in cash_logger.handlers if getattr(h, "stream", None) is sys.stderr]
        setup_logging(level=logging.DEBUG)
        _log.enable(logging.DEBUG)
        own = [h for h in cash_logger.handlers if h in _log._OWN_HANDLERS]
        assert len(own) == 1, "setup_logging's console handler, and nothing added after it"

    def test_setup_logging_keeps_the_applications_handler(self):
        cash_logger = self._clean()
        app_handler = logging.NullHandler()
        cash_logger.addHandler(app_handler)
        setup_logging(level=logging.INFO)
        setup_logging(level=logging.INFO)
        assert app_handler in cash_logger.handlers
        assert len([h for h in cash_logger.handlers if h in _log._OWN_HANDLERS]) == 1
