"""A value derived from a file-loaded value inherits its file dependencies only
when it can carry the file's data.

A DataFrame or list built from a CSV depends on that CSV; a count, mean or
flag computed from it does not, so editing the file does not invalidate them.
"""

from tests._cell_driver import run_cash_cell


class TestNarrowFileDependencyPropagation:
    """Tests for narrow file dependency propagation (only to data-bearing types)."""

    def test_scalar_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Scalar output (int) should NOT inherit file deps from DataFrame input."""
        magics = cash_magics
        sp = magics._statement_processor

        # Simulate file-loaded DataFrame
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        # Cell 1: Load CSV
        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        # Manually set file deps as if FileAccessTracker tracked it
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Compute scalar from DataFrame
        run_cash_cell(magics, "n_rows = len(df)")

        # n_rows (int) should NOT have file deps
        assert (
            "n_rows" not in sp.tracking_state.executed_file_deps
            or len(sp.tracking_state.executed_file_deps.get("n_rows", set())) == 0
        )

    def test_dataframe_does_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """DataFrame output should inherit file deps from DataFrame input."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Transform DataFrame
        run_cash_cell(magics, "df2 = df[df['a'] > 1]")

        # df2 (DataFrame) SHOULD have file deps
        assert "df2" in sp.tracking_state.executed_file_deps
        assert csv_path_str in sp.tracking_state.executed_file_deps["df2"]

    def test_list_does_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """List output should inherit file deps (it could hold data)."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Convert to list (data-bearing type)
        run_cash_cell(magics, "values = df['a'].tolist()")

        # values (list) SHOULD have file deps
        assert "values" in sp.tracking_state.executed_file_deps
        assert csv_path_str in sp.tracking_state.executed_file_deps["values"]

    def test_float_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Float output should NOT inherit file deps."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Compute float from DataFrame
        run_cash_cell(magics, "mean_val = df['a'].mean()")

        # mean_val (float) should NOT have file deps
        assert (
            "mean_val" not in sp.tracking_state.executed_file_deps
            or len(sp.tracking_state.executed_file_deps.get("mean_val", set())) == 0
        )

    def test_bool_does_not_inherit_file_deps(self, cash_magics, mock_shell, tmp_path):
        """Bool output should NOT inherit file deps."""
        magics = cash_magics
        sp = magics._statement_processor

        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_path, index=False)

        run_cash_cell(magics, f"import pandas as pd; df = pd.read_csv('{csv_path_str}')")
        sp.tracking_state.executed_file_deps["df"] = {csv_path_str}

        # Cell 2: Compute bool from DataFrame
        run_cash_cell(magics, "has_data = len(df) > 0")

        # has_data (bool) should NOT have file deps
        assert (
            "has_data" not in sp.tracking_state.executed_file_deps
            or len(sp.tracking_state.executed_file_deps.get("has_data", set())) == 0
        )
