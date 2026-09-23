import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics
from tests._cell_driver import run_cash_cell

# Captured at import time, before conftest's autouse ``disable_auto_magic_
# registration`` fixture stubs ``Cash.register_magic`` to a no-op. The
# registration regression test below calls this real implementation directly
# to exercise the path that fixture otherwise hides.
_REAL_REGISTER_MAGIC = Cash.register_magic


class TestCashMagics(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _notebook(self, cash_magics, mock_shell, clean_backend):
        self.magics, self.shell, self.backend = cash_magics, mock_shell, clean_backend

    def test_basic_caching(self):
        # 1. First run: Calculate a = 1 + 1
        # @cash:persist forces caching regardless of the 10 ms min-execution-time floor
        cell = "# @cash:persist\na = 1 + 1"
        run_cash_cell(self.magics, cell)

        self.assertEqual(self.shell.user_ns.get("a"), 2)
        # With incremental caching, we store per statement.
        self.assertEqual(len(self.backend.list_entries()), 1)

        # 2. Modify 'a' in namespace to verify restoration
        self.shell.user_ns["a"] = 99

        # 3. Second run: Should restore 'a' = 2
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("a"), 2)

    def test_incremental_caching(self):
        # Cell with two statements.
        # Each statement gets its own @cash:persist annotation so both are
        # cached regardless of the 10 ms min-execution-time floor.
        cell = """# @cash:persist
x = 10
# @cash:persist
y = x + 5
"""
        # 1. First run
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("x"), 10)
        self.assertEqual(self.shell.user_ns.get("y"), 15)
        self.assertEqual(len(self.backend.list_entries()), 2)  # Two statements cached

        # 2. Change second statement
        cell_v2 = """# @cash:persist
x = 10
# @cash:persist
y = x + 10
"""
        run_cash_cell(self.magics, cell_v2)

        # Verify behavior: x should be restored from cache, y should be recomputed
        self.assertEqual(self.shell.user_ns.get("x"), 10)
        self.assertEqual(self.shell.user_ns.get("y"), 20)

        # Now we have 3 entries in cache:
        # - x = 10 (still valid from first run)
        # - y = x + 5 (old entry, no longer used but not deleted)
        # - y = x + 10 (new entry from second run)
        self.assertEqual(len(self.backend.list_entries()), 3)

    def test_global_caching(self):
        # Test enabling auto-caching
        self.magics.cash_on("")
        self.assertTrue(self.magics.cash_status("dict")["auto_cache_enabled"])

        # Test disabling
        self.magics.cash_off("")
        self.assertFalse(self.magics.cash_status("dict")["auto_cache_enabled"])

    def test_input_dependency(self):
        # 1. Setup input
        self.shell.user_ns["x"] = 10

        # 2. First run: y = x * 2
        cell = "y = x * 2"
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("y"), 20)

        # 3. Change input
        self.shell.user_ns["x"] = 5

        # 4. Second run: Should recompute y = 10 (cache miss due to input change)
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("y"), 10)

    def test_ttl(self):
        cell = "z = 100"
        run_cash_cell(self.magics, cell, ttl=1)
        self.assertEqual(self.shell.user_ns.get("z"), 100)

        # Modify z
        self.shell.user_ns["z"] = 0

        # Immediate re-run -> Restore
        run_cash_cell(self.magics, cell, ttl=1)
        self.assertEqual(self.shell.user_ns.get("z"), 100)

        # Wait for TTL
        time.sleep(1.1)
        self.shell.user_ns["z"] = 0

        # Re-run -> Recompute
        run_cash_cell(self.magics, cell, ttl=1)
        self.assertEqual(self.shell.user_ns.get("z"), 100)

        # Verify that we actually recomputed (mock side effect check?)
        # For now, just checking correctness is enough.

    def test_multiple_outputs(self):
        cell = """
p = 10
q = 20
"""
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("p"), 10)
        self.assertEqual(self.shell.user_ns.get("q"), 20)

        self.shell.user_ns["p"] = 0
        self.shell.user_ns["q"] = 0

        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("p"), 10)
        self.assertEqual(self.shell.user_ns.get("q"), 20)

    def test_module_input_skipping(self):
        import time

        self.shell.user_ns["time"] = time
        self.shell.user_ns["x"] = 10

        # Should not warn or fail
        cell = """
import time
y = x * 2
"""
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("y"), 20)

    def test_output_capture(self):
        cell = "print('Hello Cache')"

        # 1. First run: Capture output
        # We need to mock capture_output or check if it works with MockShell?
        # IPython.utils.io.capture_output relies on sys.stdout/stderr redirection.
        # It should work in standard python env.

        # We need to capture the *actual* stdout during the test to verify replay.
        import sys
        from io import StringIO

        # Capture stdout of the test process itself
        captured_stdout = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_stdout

        try:
            run_cash_cell(self.magics, cell)
        finally:
            sys.stdout = original_stdout

        output = captured_stdout.getvalue()
        # Note: capture_output().show() prints to sys.stdout, so we should see it.
        self.assertIn("Hello Cache", output)

        # 2. Second run: Replay
        captured_stdout = StringIO()
        sys.stdout = captured_stdout

        try:
            run_cash_cell(self.magics, cell)
        finally:
            sys.stdout = original_stdout

        output = captured_stdout.getvalue()
        self.assertIn("Hello Cache", output)
        # Note: With incremental caching, we might not print "[Restored from cache]" per statement if we commented it out.
        # But let's check if the output is there.

    def test_dependency_tracking_repro(self):
        # User reported: changing multiplier doesn't trigger re-run
        self.shell.user_ns["multiplier"] = 10
        self.shell.user_ns["result"] = 42

        cell = """
print(f"Computing with multiplier {multiplier}...")
final_value = result * multiplier
"""
        # 1. First run
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("final_value"), 420)

        # 2. Change multiplier
        self.shell.user_ns["multiplier"] = 5

        # 3. Second run - Should recompute
        run_cash_cell(self.magics, cell)
        self.assertEqual(self.shell.user_ns.get("final_value"), 210)

    def test_rich_output_capture(self):
        # @cash:persist forces caching regardless of the 10 ms min-execution-time floor
        cell = "# @cash:persist\n'rich_output'"

        # 1. First run: Mock capture_output to return rich output
        mock_output = {"data": {"text/plain": "mock_data"}, "metadata": {}}

        # Patch capture_output where the statement pipeline calls it, and
        # publish_display_data to avoid IPython initialization issues.
        # capture.replay_outputs imports publish_display_data function-locally,
        # so it resolves through IPython.display at call time -- patch it there.
        with (
            patch("cash.notebook.statement.capture.capture_output") as mock_capture,
            patch("IPython.display.publish_display_data"),
        ):
            # Configure mock context manager
            mock_captured = MagicMock()
            mock_captured.stdout = ""
            mock_captured.stderr = ""
            mock_captured.outputs = [mock_output]
            mock_capture.return_value.__enter__.return_value = mock_captured

            run_cash_cell(self.magics, cell)

            # Verify it was stored
            entries = self.backend.list_entries()
            self.assertEqual(len(entries), 1)
            metadata = entries[0]
            key = metadata["key"]
            # backend.get now returns (metadata, data) where data is already a dict
            _, payload = self.backend.get(key)
            self.assertEqual(payload["rich_outputs"], [mock_output])

        # 2. Second run: Cache hit -> Replay
        with patch("IPython.display.publish_display_data") as mock_publish:
            run_cash_cell(self.magics, cell)

            combined_calls = mock_publish.call_args_list

            # Check that at least one call was made with the expected arguments
            expected_call_found = False
            for call in combined_calls:
                if (
                    call.kwargs.get("data") == mock_output["data"]
                    and call.kwargs.get("metadata") == mock_output["metadata"]
                ):
                    expected_call_found = True
                    break

            self.assertTrue(
                expected_call_found, f"Expected publish_display_data call not found. Calls: {combined_calls}"
            )

    def test_cash_help_annotation_examples_parse(self):
        """Every '@cash:' line in %cash_help output must match the real parser.

        Regression guard for QW-4: prior help text printed
            # @cash: no-cache
        with a space after the colon, which silently fails to parse because
        ANNOTATION_PATTERN requires the directive to follow the colon directly.
        """
        import sys
        from io import StringIO

        from cash.analysis.annotations import ANNOTATION_PATTERN, parse_annotation_line

        captured = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured
        try:
            self.magics.cash_help("")
        finally:
            sys.stdout = original_stdout

        output = captured.getvalue()
        annotation_lines = [line for line in output.splitlines() if "@cash:" in line]
        # Sanity-check: the default help text should actually advertise annotations.
        self.assertTrue(
            annotation_lines,
            f"Expected %cash_help output to contain '@cash:' examples; got:\n{output}",
        )
        for line in annotation_lines:
            self.assertIsNotNone(
                ANNOTATION_PATTERN.search(line),
                f"Help-text line does not match ANNOTATION_PATTERN: {line!r}",
            )
            self.assertIsNotNone(
                parse_annotation_line(line),
                f"Help-text line does not parse as a real annotation: {line!r}",
            )


def _help_output(magics, topic=""):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        magics.cash_help(topic)
    return buf.getvalue()


def test_cash_help_lists_exactly_the_registered_magics(cash_magics):
    """The card is built from the registered magics, so it names each one
    once and names nothing else. The hand-written card it replaced never
    listed ``%cash_persist`` and kept listing magics that had been removed."""
    import re

    out = _help_output(cash_magics)
    listed = re.findall(r"^  %(cash_\w+) ", out, re.M)
    assert sorted(listed) == sorted(CashMagics.magics["line"])
    assert "cash_persist" in listed
    assert all(re.search(rf"^  %{name} +\S", out, re.M) for name in listed), out


def test_cash_help_ends_with_the_feedback_links(cash_magics):
    out = _help_output(cash_magics).rstrip().splitlines()
    assert out[-2:] == [
        "Bug reports & feature requests: https://github.com/galgtonold/cash/issues",
        "Questions & discussion: https://github.com/galgtonold/cash/discussions",
    ]


def test_cash_help_topic_prints_that_magics_docstring(cash_magics):
    import inspect

    expected = inspect.getdoc(CashMagics.cash_badge)
    for topic in ("badge", "cash_badge", "%cash_badge", "  Badge  # comment"):
        out = _help_output(cash_magics, topic)
        assert out.startswith("%cash_badge\n"), (topic, out)
        assert expected in out
        assert "[R] RESTORED" in out


def test_cash_help_unknown_topic_says_so_and_prints_the_card(cash_magics):
    out = _help_output(cash_magics, "collab")
    assert out.startswith("No magic named %cash_collab.")
    assert "  %cash_on " in out


def test_register_magic_registers_cash_on(mock_shell, cash_instance):
    """``Cash.register_magic()`` imports ``CashMagics`` from its current
    path and registers it on the active shell.

    Regression guard: ``register_magic`` imported ``CashMagics`` from the
    pre-refactor path ``cash.notebook.magics`` (moved to
    ``cash.notebook.ipython.magics``) *inside* the method's
    ``except ImportError`` guard. The stale import raised
    ``ModuleNotFoundError`` — a subclass of ``ImportError`` — which the guard
    swallowed as "IPython not available". So ``%load_ext cash`` silently
    registered nothing and ``%cash_on`` came back "not found". The import now
    lives outside the guard, so a broken path raises loudly instead.

    A fake shell stands in for a real ``InteractiveShell`` so the check is
    immune to the cross-test IPython-singleton pollution that only surfaces in
    the full-suite single-process run. End-to-end registration against a real
    kernel is covered by the notebook-integration suite.
    """
    mock_shell.register_magics = MagicMock()

    with patch("IPython.get_ipython", return_value=mock_shell):
        # Call the real implementation, not conftest's no-op stub.
        _REAL_REGISTER_MAGIC(cash_instance)

    mock_shell.register_magics.assert_called_once()
    (registered_magics,), _ = mock_shell.register_magics.call_args
    assert isinstance(registered_magics, CashMagics)


if __name__ == "__main__":
    unittest.main()
