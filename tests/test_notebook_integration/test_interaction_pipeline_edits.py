"""Batch 158 – Complex real-world simulation: data science pipeline.

Full end-to-end data science workflow: load data, clean,
feature engineer, model (simple), evaluate — with edits
at each stage.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(60)]


class TestDataSciencePipeline:
    """Full data science pipeline simulation."""

    def test_full_pipeline_edit_source(self, nb_runner, tmp_path):
        """Full pipeline, edit source data."""
        csv = tmp_path / "dataset.csv"
        csv.write_text("feature,target\n1,10\n2,20\n3,30\n4,40\n5,50\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "# Clean\ndf_clean = df.dropna()",
            "# Feature engineering\ndf_clean = df_clean.copy()\ndf_clean['feature_sq'] = df_clean['feature'] ** 2",
            "# Simple model: linear average\nmean_target = df_clean['target'].mean()\nprint(f'mean = {mean_target}')",
            "# Evaluate\nresiduals = [(t - mean_target) for t in df_clean['target']]\nrmse = (sum(r**2 for r in residuals) / len(residuals)) ** 0.5\nprint(f'rmse = {round(rmse, 2)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 30.0" in nb_runner.get_output(4)
        assert "rmse = " in nb_runner.get_output(5)

        # Edit source data
        csv.write_text("feature,target\n10,100\n20,200\n30,300\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 200.0" in nb_runner.get_output(4)

    def test_full_pipeline_edit_feature_engineering(self, nb_runner, tmp_path):
        """Edit feature engineering step."""
        csv = tmp_path / "data2.csv"
        csv.write_text("x,y\n1,2\n2,4\n3,6\n4,8\n")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "# Feature\nfeature_sum = df['x'].sum()\nprint(f'feature_sum = {feature_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "feature_sum = 10" in nb_runner.get_output(2)

        # Change to use y column
        nb_runner.set_cell_source(
            2,
            "# Feature v2\nfeature_sum = df['y'].sum()\nprint(f'feature_sum = {feature_sum}')",
        )
        nb_runner.run_all()
        assert "feature_sum = 20" in nb_runner.get_output(2)


class TestETLPipeline:
    """Extract-Transform-Load pipeline simulation."""

    def test_etl_edit_transform(self, nb_runner):
        """ETL pipeline, edit transform step."""
        nb_runner.create_notebook([
            "# Extract\nraw = [{'name': 'Alice', 'score': 85}, {'name': 'Bob', 'score': 92}]",
            "# Transform\ntransformed = [{'name': r['name'], 'grade': 'A' if r['score'] >= 90 else 'B'} for r in raw]",
            "# Load (print)\nfor t in transformed:\n    print(f\"{t['name']}: {t['grade']}\")",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Alice: B" in output
        assert "Bob: A" in output

        # Edit transform — lower A threshold
        nb_runner.set_cell_source(
            2,
            "# Transform v2\ntransformed = [{'name': r['name'], 'grade': 'A' if r['score'] >= 80 else 'B'} for r in raw]",
        )
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Alice: A" in output
        assert "Bob: A" in output

    def test_etl_edit_extract(self, nb_runner):
        """ETL pipeline, edit extract step."""
        nb_runner.create_notebook([
            "# Extract\ndata = [10, 20, 30]",
            "# Transform\nscaled = [x * 2 for x in data]",
            "# Load\nresult = sum(scaled)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(3)

        # Edit extract
        nb_runner.set_cell_source(1, "# Extract v2\ndata = [100, 200, 300, 400]")
        nb_runner.run_all()
        assert "result = 2000" in nb_runner.get_output(3)
