"""
Tests for CashMagics methods that need additional coverage.

Targets: cash_badge, cash_status, _capture_cell_id.

Object-hashing helpers (`compute_hash`, `calculate_memory_size`,
`_recursive_getsizeof`) moved to `cash.object_hashing`;
their tests live in `test_object_hashing.py`.
"""

from unittest.mock import MagicMock


class TestSaveHintLiveReaderAware:
    """The 'save your upstream edits' hint is for the FILE reader only.

    Wherever a live reader supplies the cells the advice is not redundant, it is
    false, so it is suppressed: Colab reads them from the frontend
    (``get_ipynb``), and on JupyterLab cash's own extension pushes them over a
    comm. Both gates are monkeypatched in every case below -- the extension
    probe looks at the real filesystem, so leaving it unset would make these
    tests pass or fail depending on whether the developer happens to have
    JupyterLab in the same environment.
    """

    @staticmethod
    def _gates(monkeypatch, *, colab: bool, labext: bool) -> None:
        monkeypatch.setattr("cash.notebook.ipython.magics.in_colab", lambda: colab)
        monkeypatch.setattr("cash.notebook.ipython.magics.labextension_installed", lambda: labext)

    def test_hint_shown_when_no_live_reader(self, cash_magics, capsys, monkeypatch):
        self._gates(monkeypatch, colab=False, labext=False)
        cash_magics._save_hint_shown = False
        cash_magics.cash_on("")
        out = capsys.readouterr().out
        assert "saved notebook file" in out
        assert "Save (Ctrl+S)" in out

    def test_hint_suppressed_in_colab(self, cash_magics, capsys, monkeypatch):
        self._gates(monkeypatch, colab=True, labext=False)
        cash_magics._save_hint_shown = False
        cash_magics.cash_on("")
        out = capsys.readouterr().out
        assert "Cash enabled" in out  # cash_on still ran normally
        assert "Save (Ctrl+S)" not in out  # but the save hint is suppressed
        assert "saved notebook file" not in out

    def test_hint_suppressed_when_the_labextension_is_installed(self, cash_magics, capsys, monkeypatch):
        """The extension makes the save advice false."""
        self._gates(monkeypatch, colab=False, labext=True)
        cash_magics._save_hint_shown = False
        cash_magics.cash_on("")
        out = capsys.readouterr().out
        assert "Cash enabled" in out
        assert "Save (Ctrl+S)" not in out
        assert "saved notebook file" not in out

    def test_a_broken_probe_keeps_the_hint_rather_than_withdrawing_it(self, cash_magics, capsys, monkeypatch):
        """The probe must never raise, and its failure must not silence advice
        that is correct for everyone without the extension."""

        def _boom():
            raise RuntimeError("no filesystem for you")

        monkeypatch.setattr("cash.notebook.ipython.magics.in_colab", lambda: False)
        monkeypatch.setattr("os.path.isdir", lambda *_a, **_k: _boom())
        cash_magics._save_hint_shown = False
        cash_magics.cash_on("")  # must not raise
        out = capsys.readouterr().out
        assert "Save (Ctrl+S)" in out


# ============================================================================
# cash_badge magic
# ============================================================================


class TestCashBadge:
    """Test %cash_badge magic command."""

    def test_set_badge_html(self, cash_magics, capsys):
        cash_magics.cash_badge("html")
        assert cash_magics.badges.mode == "html"
        captured = capsys.readouterr()
        assert "Badge mode set to: html" in captured.out

    def test_set_badge_print(self, cash_magics, capsys):
        cash_magics.cash_badge("print")
        assert cash_magics.badges.mode == "print"
        captured = capsys.readouterr()
        assert "Badge mode set to: print" in captured.out

    def test_set_badge_off(self, cash_magics, capsys):
        cash_magics.cash_badge("off")
        assert cash_magics.badges.mode == "off"
        captured = capsys.readouterr()
        assert "Badge mode set to: off" in captured.out

    def test_badge_invalid_shows_current(self, cash_magics, capsys):
        cash_magics.badges.mode = "print"
        cash_magics.cash_badge("invalid_mode")
        captured = capsys.readouterr()
        assert "Current badge mode: print" in captured.out
        assert "Usage:" in captured.out

    def test_badge_empty_shows_current(self, cash_magics, capsys):
        cash_magics.badges.mode = "html"
        cash_magics.cash_badge("")
        captured = capsys.readouterr()
        assert "Current badge mode: html" in captured.out


# ============================================================================
# cash_status magic
# ============================================================================


class TestCashStatus:
    """Test %cash_status magic command."""

    def test_status_print_mode(self, cash_magics, capsys):
        result = cash_magics.cash_status("")
        assert isinstance(result, dict)
        assert "lineage" in result
        assert "auto_cache_enabled" in result
        assert "debug_enabled" in result
        captured = capsys.readouterr()
        assert captured.out.strip()  # Should print something

    def test_status_dict_mode(self, cash_magics):
        result = cash_magics.cash_status("dict")
        assert isinstance(result, dict)
        assert "last_cell" in result
        assert "lineage" in result
        assert "cache_stats" in result

    def test_status_counts_entries_without_listing_them(self, cash_magics, clean_backend, monkeypatch):
        """The count comes from ``entry_count``: ``list_entries`` reads every
        entry's metadata, seconds on a large file cache."""
        clean_backend.set("k1", 1, {})
        clean_backend.set("k2", 2, {})

        def no_listing():
            raise AssertionError("%cash_status listed the backend to count it")

        monkeypatch.setattr(clean_backend, "list_entries", no_listing)
        assert cash_magics.cash_status("dict")["cache_stats"] == {"keys": 2}

    def test_status_json_mode(self, cash_magics):
        result = cash_magics.cash_status("json")
        assert isinstance(result, str)
        import json

        parsed = json.loads(result)
        assert "lineage" in parsed

    def test_status_reflects_execution(self, cash_magics):
        """After executing a statement, status should reflect it."""
        processor = cash_magics._statement_processor
        processor.process_statement("x = 42")
        result = cash_magics.cash_status("dict")
        assert "x" in result["executed_codes"]

    def test_last_cell_metrics_empty_status_is_none(self, cash_magics):
        """A cell that produced no statement metrics yields overall status None (magics.py 778)."""
        cash_magics._update_last_cell_metrics([], 0.0)
        last_cell = cash_magics.cash_status("dict")["last_cell"]
        assert last_cell["status"] is None
        assert last_cell["statements"] == []


# ============================================================================
# _capture_cell_id
# ============================================================================


class TestCaptureCellId:
    """Test _capture_cell_id method."""

    def test_capture_from_info_cell_id(self, cash_magics):
        info = MagicMock()
        info.cell_id = "test-cell-123"
        cash_magics._capture_cell_id(info)
        assert cash_magics.current_cell_id == "test-cell-123"

    def test_capture_from_vscode_metadata(self, cash_magics, mock_shell):
        info = MagicMock(spec=[])  # No cell_id attribute
        # Simulate VS Code parent header
        mock_shell.get_parent = MagicMock(return_value={"metadata": {"vscode": {"cellId": "vscode-cell-456"}}})
        cash_magics._capture_cell_id(info)
        assert cash_magics.current_cell_id == "vscode-cell-456"

    def test_capture_from_parent_metadata_cellId(self, cash_magics, mock_shell):
        info = MagicMock(spec=[])
        mock_shell.get_parent = MagicMock(return_value={"metadata": {"cellId": "parent-cell-789"}})
        cash_magics._capture_cell_id(info)
        assert cash_magics.current_cell_id == "parent-cell-789"

    def test_capture_no_cell_id_available(self, cash_magics):
        info = MagicMock(spec=[])
        cash_magics._capture_cell_id(info)
        assert cash_magics.current_cell_id is None

    def test_capture_exception_handled(self, cash_magics, mock_shell):
        """Exceptions in capture_cell_id should not propagate."""
        # Create an info object where accessing cell_id raises
        info = MagicMock(spec=[])  # No cell_id
        # Make shell.get_parent raise an exception
        mock_shell.get_parent = MagicMock(side_effect=RuntimeError("test error"))
        # This should not raise
        cash_magics._capture_cell_id(info)
        assert cash_magics.current_cell_id is None

    def test_capture_debug_output(self, cash_magics, caplog):
        """cell_id capture emits a DEBUG log record, not a raw stdout print."""
        import logging

        cash_magics._debug = True
        info = MagicMock()
        info.cell_id = "debug-cell"
        with caplog.at_level(logging.DEBUG, logger="cash.notebook.ipython.magics"):
            cash_magics._capture_cell_id(info)
        messages = [r.getMessage() for r in caplog.records]
        assert any("[CELL_ID]" in m and "debug-cell" in m for m in messages), messages

    def test_capture_no_cell_id_does_not_print(self, cash_magics, capsys):
        """The no-cell_id path must not write to stdout (was noisy every cell)."""
        cash_magics._debug = True
        info = MagicMock(spec=[])  # no cell_id attribute
        cash_magics._capture_cell_id(info)
        assert "[CELL_ID]" not in capsys.readouterr().out


# ============================================================================
# cash_debug console handler
#
# `%cash_debug on` must install a console handler on the `cash` logger so
# logger.debug() markers ([UPSTREAM_DEBUG], [CACHE_HIT_DEBUG], ...) actually
# reach the captured cell output. Raising the logger level alone is not enough
# on recent Python/ipykernel where ambient root propagation no longer routes
# DEBUG records to the cell.
# ============================================================================

import logging


def _cash_debug_handlers():
    return [h for h in logging.getLogger("cash").handlers if getattr(h, "_cash_debug_console", False)]


class TestCashDebugConsoleHandler:
    """`%cash_debug on/off` installs and quiets a DEBUG console handler."""

    def teardown_method(self):
        # Don't leak the process-global handler into other tests.
        cash_logger = logging.getLogger("cash")
        for h in _cash_debug_handlers():
            cash_logger.removeHandler(h)

    def test_on_installs_handler_routing_debug_to_stdout(self, cash_magics, capsys):
        cash_magics.cash_debug("on")

        handlers = _cash_debug_handlers()
        assert len(handlers) == 1
        assert handlers[0].level == logging.DEBUG

        # A logger.debug() on a cash.* child logger must reach stdout.
        logging.getLogger("cash.notebook.unittest").debug("[UPSTREAM_DEBUG] hello")
        out = capsys.readouterr().out
        assert "[UPSTREAM_DEBUG] hello" in out
        assert "[cash.notebook.unittest]" in out

    def test_on_is_idempotent(self, cash_magics):
        cash_magics.cash_debug("on")
        cash_magics.cash_debug("on")
        # Never add the console handler twice.
        assert len(_cash_debug_handlers()) == 1

    def test_off_quiets_handler(self, cash_magics, capsys):
        cash_magics.cash_debug("on")
        capsys.readouterr()  # drop the "enabled" message

        cash_magics.cash_debug("off")
        # Handler is left attached but raised above DEBUG so nothing emits.
        handlers = _cash_debug_handlers()
        assert len(handlers) == 1
        assert handlers[0].level > logging.DEBUG

        logging.getLogger("cash.notebook.unittest").debug("[UPSTREAM_DEBUG] silent")
        out = capsys.readouterr().out
        assert "[UPSTREAM_DEBUG] silent" not in out

    def test_handler_follows_current_stdout(self, cash_magics):
        """The handler resolves sys.stdout lazily (per cell), not at install."""
        import io
        import sys

        cash_magics.cash_debug("on")
        handler = _cash_debug_handlers()[0]

        new_stream = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stream
        try:
            logging.getLogger("cash.notebook.unittest").debug("[UPSTREAM_DEBUG] routed")
        finally:
            sys.stdout = old_stdout
        # Record landed in the stdout active at emit time, not install time.
        assert "[UPSTREAM_DEBUG] routed" in new_stream.getvalue()
        assert handler.stream is old_stdout
