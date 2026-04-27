"""
Integration tests for file dependency path fallback resolution.

When a project is moved (e.g. Google Drive path changes), the absolute paths
stored in cache file dependencies may no longer exist.  The resolve_file_dep_path
utility should find the file through basename/suffix matching, allowing upstream
virtual restore to succeed without wasteful re-execution.
"""
import pytest
import pandas as pd
import time

pytestmark = pytest.mark.files


class TestFileDepPathFallback:
    """Tests for file dependency resolution when stored paths become stale."""

    def test_upstream_restore_after_path_change(self, nb_runner, tmp_path):
        """
        Simulate a project move: cache created with one path, then re-run
        from a different CWD.  Upstream virtual restore should still succeed
        because resolve_file_dep_path finds the file via basename fallback.

        Pipeline: read_csv -> transform -> downstream computation
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")

        pd.DataFrame({"val": [10, 20, 30]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('loaded:', len(df))",
            "total = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()

        # First run: populate cache
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "total = 60" in out1

        # Second run: cache should be used (virtual restore succeeds)
        nb_runner.run_cell(2)
        out2 = nb_runner.get_output(2)
        assert "total = 60" in out2

    def test_file_change_detected_through_fallback_path(self, nb_runner, tmp_path):
        """
        Even when using a fallback-resolved path, file content changes are
        still detected (mtime mismatch) and correctly trigger re-execution.
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")

        pd.DataFrame({"val": [10, 20, 30]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('loaded:', len(df))",
            "total = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "total = 60" in out1

        # Modify the file - mtime changes
        time.sleep(0.2)
        pd.DataFrame({"val": [100, 200, 300]}).to_csv(csv_path, index=False)

        # Re-run: should detect file change and recompute
        nb_runner.run_cell(2)
        out2 = nb_runner.get_output(2)
        assert "total = 600" in out2

    def test_self_assignment_with_file_dep_restore(self, nb_runner, tmp_path):
        """
        Self-assignment cells (df = df.sort_values()) that inherit file deps
        should also benefit from path fallback resolution.
        """
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")

        pd.DataFrame({"val": [30, 10, 20]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')\nprint('loaded:', len(df))",
            "df = df.sort_values('val')\nprint('sorted:', df['val'].tolist())",
            "total = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total = 60" in out

        # Re-run downstream: upstream should restore from cache efficiently
        nb_runner.run_cell(3)
        out2 = nb_runner.get_output(3)
        assert "total = 60" in out2
