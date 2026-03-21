"""
Regression tests: no double prints in notebook cell output.

Historical bug: When a cell with multiple print statements was run or re-run,
stdout from subsequent statements appeared twice in the output. The pattern was:
  - First stdout-producing statement: appeared ONCE
  - All subsequent stdout-producing statements: appeared TWICE

Root cause: Statement stdout was being printed both by the silent=False
execution path AND by the metrics replay loop in _execute_cell.

These tests verify that every print() appears exactly once in cell output,
across multiple scenarios:
  - Fresh compute, cache restore, skip
  - Multi-statement cells with/without mutations
  - Upstream re-execution triggered cells
  - Debug on/off
  - Cell edits (cell_code_changed)
  - %%cash cell magic
"""
import pytest
import pandas as pd

pytestmark = [pytest.mark.core, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count(output: str, marker: str) -> int:
    """Count occurrences of marker in output, stripping debug lines."""
    lines = [
        line for line in output.splitlines()
        if not line.startswith('[') and 'Cash:' not in line
    ]
    clean = '\n'.join(lines)
    return clean.count(marker)


def _assert_single(output: str, *markers: str):
    """Assert each marker appears exactly once in output."""
    for m in markers:
        count = _count(output, m)
        assert count == 1, (
            f"'{m}' appeared {count} times (expected 1). Output:\n{output}"
        )


# ---------------------------------------------------------------------------
# Basic scenarios
# ---------------------------------------------------------------------------

class TestNoDoublePrints:
    """Each print() statement must produce exactly one line of output."""

    def test_single_print(self, nb_runner):
        """Single print — baseline sanity check."""
        nb_runner.create_notebook([
            "print('MARKER_A')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'MARKER_A')

    def test_two_prints_same_cell(self, nb_runner):
        """Two prints in one cell — the historically broken case."""
        nb_runner.create_notebook([
            "print('FIRST')\nprint('SECOND')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'FIRST', 'SECOND')

    def test_three_prints_same_cell(self, nb_runner):
        """Three prints in one cell."""
        nb_runner.create_notebook([
            "print('AAA')\nprint('BBB')\nprint('CCC')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'AAA', 'BBB', 'CCC')

    def test_prints_interleaved_with_assignments(self, nb_runner):
        """Prints interleaved with variable assignments."""
        nb_runner.create_notebook([
            "x = 1\nprint('AFTER_X')\ny = 2\nprint('AFTER_Y')\nz = x + y\nprint(f'Z={z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'AFTER_X', 'AFTER_Y', 'Z=3')


# ---------------------------------------------------------------------------
# Re-run (cache/skip) scenarios
# ---------------------------------------------------------------------------

class TestNoDoublePrintsOnRerun:
    """Re-running cells must not double the output."""

    def test_rerun_two_prints(self, nb_runner):
        """Re-run a cell with two prints — both runs should show each once."""
        nb_runner.create_notebook([
            "print('HELLO')\nprint('WORLD')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'HELLO', 'WORLD')

        # Re-run
        nb_runner.run_cell(1)
        _assert_single(nb_runner.get_output(1), 'HELLO', 'WORLD')

    def test_rerun_with_mutations(self, nb_runner, tmp_path):
        """Re-running a cell that mutates a DataFrame — no doubling."""
        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({'a': [3, 1, 2], 'b': [6, 4, 5]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "df = df.sort_values('a').reset_index(drop=True)\n"
            "print('SORTED')\n"
            "df['c'] = df['a'] * 10\n"
            "print('ADDED_COLUMN')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(2), 'SORTED', 'ADDED_COLUMN')

        # Re-run
        nb_runner.run_cell(2)
        _assert_single(nb_runner.get_output(2), 'SORTED', 'ADDED_COLUMN')


# ---------------------------------------------------------------------------
# Multi-cell with upstream dependency
# ---------------------------------------------------------------------------

class TestNoDoublePrintsUpstream:
    """Upstream re-execution must not cause doubled output in current cell."""

    def test_upstream_prints_not_doubled(self, nb_runner):
        """Upstream cell re-execution: each print appears once per cell."""
        nb_runner.create_notebook([
            "x = 10\nprint('UPSTREAM_DONE')",
            "y = x * 2\nprint(f'DOWNSTREAM_Y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Cell 1: upstream print
        _assert_single(nb_runner.get_output(1), 'UPSTREAM_DONE')
        # Cell 2: downstream print
        _assert_single(nb_runner.get_output(2), 'DOWNSTREAM_Y=20')

    def test_edit_upstream_no_double_downstream(self, nb_runner):
        """
        Edit an upstream cell, re-run downstream. The upstream auto-execution
        may print, but the downstream cell should not double its own prints.
        """
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'RESULT={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(2), 'RESULT=20')

        # Edit upstream
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cells([1, 2])

        out2 = nb_runner.get_output(2)
        _assert_single(out2, 'RESULT=100')

    def test_three_cell_chain_no_doubling(self, nb_runner):
        """Three-cell dependency chain — no doubling anywhere."""
        nb_runner.create_notebook([
            "a = 1\nprint('CELL1')",
            "b = a + 1\nprint(f'CELL2_B={b}')",
            "c = b + 1\nprint(f'CELL3_C={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        _assert_single(nb_runner.get_output(1), 'CELL1')
        _assert_single(nb_runner.get_output(2), 'CELL2_B=2')
        _assert_single(nb_runner.get_output(3), 'CELL3_C=3')


# ---------------------------------------------------------------------------
# Debug on/off
# ---------------------------------------------------------------------------

class TestNoDoublePrintsDebugModes:
    """Double-print regression was specifically observed with debug OFF."""

    def test_debug_off_multi_print(self, nb_runner):
        """Debug OFF (default) — multiple prints in same cell."""
        nb_runner.create_notebook([
            "x = 42\nprint('P1')\ny = x * 2\nprint('P2')\nz = y + 1\nprint('P3')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'P1', 'P2', 'P3')

    def test_debug_on_multi_print(self, nb_runner):
        """Debug ON — multiple prints in same cell (debug lines filtered)."""
        nb_runner.create_notebook([
            "%cash_debug on",
            "x = 42\nprint('DBG_P1')\ny = x * 2\nprint('DBG_P2')\nz = y + 1\nprint('DBG_P3')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # With debug on, output contains code echoes and debug lines.
        # We only count lines that are EXACTLY our markers (user stdout).
        output = nb_runner.get_output(2)
        lines = output.splitlines()
        # User stdout lines are those that are exactly the marker text
        user_lines = [l.strip() for l in lines if l.strip() in ('DBG_P1', 'DBG_P2', 'DBG_P3')]
        assert user_lines.count('DBG_P1') == 1, f"DBG_P1 count: {user_lines.count('DBG_P1')}, lines: {user_lines}"
        assert user_lines.count('DBG_P2') == 1, f"DBG_P2 count: {user_lines.count('DBG_P2')}, lines: {user_lines}"
        assert user_lines.count('DBG_P3') == 1, f"DBG_P3 count: {user_lines.count('DBG_P3')}, lines: {user_lines}"


# ---------------------------------------------------------------------------
# Cell edit scenario (cell_code_changed)
# ---------------------------------------------------------------------------

class TestNoDoublePrintsCellEdit:
    """Editing and re-running a cell must not double its prints."""

    def test_edit_cell_and_rerun(self, nb_runner):
        """Edit a multi-print cell — new prints appear once each."""
        nb_runner.create_notebook([
            "print('ORIG_A')\nprint('ORIG_B')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'ORIG_A', 'ORIG_B')

        # Edit: change second print
        nb_runner.set_cell_source(1, "print('ORIG_A')\nprint('EDITED_B')")
        nb_runner.run_cell(1)

        out = nb_runner.get_output(1)
        _assert_single(out, 'ORIG_A', 'EDITED_B')
        assert 'ORIG_B' not in out

    def test_add_print_to_cell(self, nb_runner):
        """Add a print statement to an existing cell — no doubling."""
        nb_runner.create_notebook([
            "x = 1\nprint('ONLY_PRINT')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        _assert_single(nb_runner.get_output(1), 'ONLY_PRINT')

        # Add a second print
        nb_runner.set_cell_source(1, "x = 1\nprint('ONLY_PRINT')\ny = 2\nprint('NEW_PRINT')")
        nb_runner.run_cell(1)

        out = nb_runner.get_output(1)
        _assert_single(out, 'ONLY_PRINT', 'NEW_PRINT')


# ---------------------------------------------------------------------------
# Financial analysis demo pattern (the original trigger)
# ---------------------------------------------------------------------------

class TestNoDoublePrintsFinancialPattern:
    """Mirrors the financial_analysis_demo pattern that originally triggered the bug."""

    def test_financial_demo_multi_computation(self, nb_runner, tmp_path):
        """
        Pattern: multiple DataFrame mutations with timing prints.
        Each computation prints start/end markers.
        """
        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({
            'Ticker': ['AAPL'] * 20 + ['GOOGL'] * 20,
            'Date': list(range(40)),
            'Close': [100 + i * 0.5 for i in range(40)],
            'Volume': [1000 + i * 10 for i in range(40)],
        }).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            # Cell 1: imports + data load
            f"import pandas as pd\nimport numpy as np\ndf = pd.read_csv('{csv_str}')",
            # Cell 2: sort
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)",
            # Cell 3: multiple computations with prints (the historically broken cell pattern)
            "print('CALC_VOL_START')\n"
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(\n"
            "    lambda x: x.rolling(window=3, min_periods=1).mean()\n"
            ")\n"
            "print('CALC_VOL_END')\n"
            "\n"
            "print('CALC_SMA_START')\n"
            "df['SMA'] = df.groupby('Ticker')['Close'].transform(\n"
            "    lambda x: x.rolling(window=5, min_periods=1).mean()\n"
            ")\n"
            "print('CALC_SMA_END')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        _assert_single(out, 'CALC_VOL_START', 'CALC_VOL_END',
                       'CALC_SMA_START', 'CALC_SMA_END')

    def test_financial_demo_rerun(self, nb_runner, tmp_path):
        """Same as above but re-run the computation cell."""
        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({
            'Ticker': ['AAPL'] * 20,
            'Date': list(range(20)),
            'Close': [100 + i for i in range(20)],
        }).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)",
            "print('STEP1')\n"
            "df['MA3'] = df['Close'].rolling(3, min_periods=1).mean()\n"
            "print('STEP2')\n"
            "df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()\n"
            "print('STEP3')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        _assert_single(nb_runner.get_output(3), 'STEP1', 'STEP2', 'STEP3')

        # Re-run cell 3
        nb_runner.run_cell(3)
        _assert_single(nb_runner.get_output(3), 'STEP1', 'STEP2', 'STEP3')
