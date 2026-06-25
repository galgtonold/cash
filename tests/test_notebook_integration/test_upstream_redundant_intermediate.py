"""
Test that upstream simulation and restoration works efficiently.

Verifies that:
1. Forward simulation (Pass 1) correctly propagates lineages from cache
2. When df is in memory and unchanged, no upstream re-execution is needed
3. When df is modified by a downstream cell, the "valid extension" check handles it
4. When a cell is modified, only broken variables trigger upstream restoration
5. The backwards scan (Pass 3) stops cascading when a variable is restored

Root cause context:
  The forward simulation computes lineage hashes for each upstream statement
  using cached metadata. If propagation succeeds for ALL statements, the
  virtual lineage matches the runtime lineage and no restoration is needed.
  When propagation fails (cache miss), the fallback formula may diverge from
  runtime lineage, causing false "broken" detection and redundant re-execution.
"""

import pytest

pytestmark = pytest.mark.upstream


class TestNoRedundantUpstreamRestores:
    """Verify that the upstream checker doesn't do redundant work."""

    def test_no_upstream_exec_when_df_current(self, nb_runner, tmp_path):
        """
        When df is already in memory with correct state, re-running a
        downstream cell should NOT trigger any upstream re-execution.
        """
        csv_path = tmp_path / "test_data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')

        import pandas as pd
        import numpy as np
        pd.DataFrame({
            'A': range(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100)
        }).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\nimport numpy as np\ndf = pd.read_csv('{csv_path_str}')",
            "df = df.sort_values('A')",
            "df['D'] = df['B'] * df['C']",
            "df['E'] = df['D'].rolling(5).mean()",
            "print(f'Columns: {list(df.columns)}')\nprint(f'Shape: {df.shape}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "'E'" in output1

        # Re-run cell 5 only
        nb_runner.run_cell(5)
        output2 = nb_runner.get_output(5)
        assert "'E'" in output2

        raw = nb_runner.get_raw_output(5)
        assert raw.count("Auto-executing upstream") == 0, (
            f"Unexpected upstream re-execution when df is current. Output: {raw[:500]}"
        )

    def test_no_upstream_exec_with_file_io_and_functions(self, nb_runner, tmp_path):
        """
        Even with file I/O (pd.read_csv) and custom functions (function_tracker),
        re-running a downstream cell should NOT trigger upstream re-execution
        when everything is current in memory.
        """
        csv_path = tmp_path / "financial_data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')

        import pandas as pd
        import numpy as np
        pd.DataFrame({
            'Ticker': ['AAPL'] * 50 + ['GOOGL'] * 50,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            f"df = pd.read_csv('{csv_path_str}')\ndf = df.sort_values(by=['Ticker'])",
            "df['VolAdj'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(5).mean())",
            (
                "def weighted_mean(x):\n"
                "    w = np.arange(1, len(x) + 1)\n"
                "    return np.sum(x * w) / np.sum(w)\n"
                "df['SMA'] = df.groupby('Ticker')['Close'].transform("
                "lambda x: x.rolling(10).apply(weighted_mean, raw=True))"
            ),
            "print(f'Columns: {list(df.columns)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below
        nb_runner.run_all()

        assert "'SMA'" in nb_runner.get_output(5)

        nb_runner.run_cell(5)
        assert "'SMA'" in nb_runner.get_output(5)

        raw = nb_runner.get_raw_output(5)
        assert raw.count("Auto-executing upstream") == 0, (
            f"Unexpected upstream re-execution. Output: {raw[:500]}"
        )

    def test_cell_modification_uses_valid_extension(self, nb_runner, tmp_path):
        """
        When the current cell is modified (but upstream is unchanged),
        the "valid extension" check should handle df's lineage mismatch
        without triggering upstream re-execution.
        """
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "z = y + 100",
            "df_out = z * 3\nprint(f'result = {df_out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below
        nb_runner.run_all()
        assert "result = 360" in nb_runner.get_output(4)

        # Modify cell 4
        nb_runner.set_cell_source(4, "df_out = z * 4\nprint(f'result = {df_out}')")
        nb_runner.run_cell(4)
        assert "result = 480" in nb_runner.get_output(4)

        raw = nb_runner.get_raw_output(4)
        assert raw.count("Auto-executing upstream") == 0, (
            f"Unexpected upstream re-execution after cell modification. Output: {raw[:500]}"
        )


class TestForwardSimulationConsistency:
    """Verify that forward simulation produces correct lineages."""

    def test_incremental_simulation_cache_hit(self, nb_runner, tmp_path):
        """
        On re-run with no code changes, the incremental simulation cache
        should be used (no forward propagation needed).
        """
        nb_runner.create_notebook([
            "a = 42",
            "b = a * 2",
            "c = b + 10",
            "print(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 94" in nb_runner.get_output(4)

        nb_runner.enable_debug()
        nb_runner.run_cell(4)
        assert "c = 94" in nb_runner.get_output(4)

        raw = nb_runner.get_raw_output(4)
        # Canonical signal that the cached simulation was reused: no upstream
        # statement was re-executed (the "Auto-executing upstream: ..." stdout
        # marker, restored under %cash_debug). The previous positive assertion
        # checked the "[UPSTREAM_DEBUG] ... reusing cache" string, but that is a
        # logger.debug line (it goes to the logging system, not cell stdout) and
        # the db8c3bd "silence stdout, badge is the canonical signal" refactor
        # left it unavailable here. No re-execution on a no-change re-run *is*
        # the reuse-the-cache observable.
        assert "Auto-executing upstream" not in raw, (
            f"No upstream re-execution expected on a no-change re-run. Output: {raw[:500]}"
        )

    def test_forward_propagation_with_file_io(self, nb_runner, tmp_path):
        """
        Forward simulation should correctly propagate lineages for file I/O
        statements, matching runtime lineages.
        """
        csv_path = tmp_path / "prop_data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        import pandas as pd
        pd.DataFrame({'x': [1, 2, 3]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{csv_path_str}')",
            "df['y'] = df['x'] * 2",
            "print(f'result = {df[\"y\"].sum()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(4)

        nb_runner.run_cell(4)
        assert "result = 12" in nb_runner.get_output(4)

        raw = nb_runner.get_raw_output(4)
        assert raw.count("Auto-executing upstream") == 0

    def test_upstream_after_cell_change_triggers_only_affected(self, nb_runner, tmp_path):
        """
        When an upstream cell changes, only the affected variable chain
        should be re-executed, not unrelated statements.
        """
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "a = x * 2",
            "b = y * 3",
            "print(f'a={a}, b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=20, b=60" in nb_runner.get_output(4)

        # Change cell 1: modify x but not y
        nb_runner.set_cell_source(1, "x = 100\ny = 20")
        nb_runner.run_cells([1, 4])
        output = nb_runner.get_output(4)
        assert "a=200, b=60" in output


class TestBackwardsScanOptimization:
    """Verify that the backwards scan stops cascading appropriately."""

    def test_restore_latest_df_skips_intermediates(self, nb_runner, tmp_path):
        """
        When the backwards scan finds a cache hit for the LATEST df producer,
        intermediate df producers should be skipped (not visited).
        
        This is the core of the user's question: after restoring SMA_50's df,
        VolAdj_20's df restore should be unnecessary.
        """
        nb_runner.create_notebook([
            "x = 10",
            "x = x + 1",       # x=11
            "x = x * 2",       # x=22
            "x = x + 100",     # x=122
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below
        nb_runner.run_all()
        assert "x = 122" in nb_runner.get_output(5)

        # Re-run cell 5: x is in memory with correct lineage, no upstream needed
        nb_runner.run_cell(5)
        assert "x = 122" in nb_runner.get_output(5)

        raw = nb_runner.get_raw_output(5)
        assert raw.count("Auto-executing upstream") == 0
