"""Batch 149 – Pandas DataFrame interaction tests.

Tests with pandas DataFrames across cells — loading, filtering,
transforming, and editing operations while verifying caching.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestDataFrameEdits:
    """DataFrame manipulation with cell edits."""

    def test_edit_filter_condition(self, nb_runner, tmp_path):
        """Edit DataFrame filter condition."""
        csv = tmp_path / "data.csv"
        csv.write_text("name,score\nAlice,85\nBob,92\nCharlie,78\nDiana,95\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "high = df[df['score'] > 90]\nprint(f'count = {len(high)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 2" in nb_runner.get_output(2)

        # Lower threshold
        nb_runner.set_cell_source(
            2, "high = df[df['score'] > 80]\nprint(f'count = {len(high)}')"
        )
        nb_runner.run_all()
        assert "count = 3" in nb_runner.get_output(2)

    def test_edit_aggregation(self, nb_runner, tmp_path):
        """Edit aggregation function."""
        csv = tmp_path / "sales.csv"
        csv.write_text("product,amount\nA,100\nB,200\nA,150\nB,300\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "result = df['amount'].sum()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 750" in nb_runner.get_output(2)

        # Change to mean
        nb_runner.set_cell_source(
            2, "result = df['amount'].mean()\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 187.5" in nb_runner.get_output(2)

    def test_edit_column_selection(self, nb_runner, tmp_path):
        """Edit which columns are selected."""
        csv = tmp_path / "multi.csv"
        csv.write_text("a,b,c\n1,10,100\n2,20,200\n3,30,300\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "total = df['a'].sum()\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Switch to column c
        nb_runner.set_cell_source(
            2, "total = df['c'].sum()\nprint(f'total = {total}')"
        )
        nb_runner.run_all()
        assert "total = 600" in nb_runner.get_output(2)


class TestDataFrameChainEdits:
    """Multi-step DataFrame pipelines with edits."""

    def test_filter_then_aggregate_edit_filter(self, nb_runner, tmp_path):
        """Filter → aggregate, edit filter."""
        csv = tmp_path / "records.csv"
        csv.write_text("cat,val\nX,10\nY,20\nX,30\nY,40\nX,50\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "subset = df[df['cat'] == 'X']",
            "total = subset['val'].sum()\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 90" in nb_runner.get_output(3)

        # Change filter to Y
        nb_runner.set_cell_source(2, "subset = df[df['cat'] == 'Y']")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(3)
