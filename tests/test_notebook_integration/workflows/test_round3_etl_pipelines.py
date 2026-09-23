"""
Multi-step data transformation pipelines — realistic ETL-like workflows
with many intermediate variables and complex data flow.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestETLPipeline:
    """Test caching with ETL-like transformation pipelines."""

    def test_extract_transform_load(self, nb_runner, tmp_path):
        """Full ETL pipeline: extract from CSV, transform, write output."""
        input_csv = tmp_path / "raw.csv"
        output_csv = tmp_path / "clean.csv"
        input_csv.write_text(
            "id,name,value,category\n1,Alice,100,A\n2,Bob,-5,B\n3,Charlie,200,A\n4,Diana,150,B\n5,Eve,-10,A\n"
        )
        in_str = str(input_csv).replace("\\", "/")
        out_str = str(output_csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd\nimport numpy as np",
                # Extract
                f"raw = pd.read_csv('{in_str}')",
                # Validate
                textwrap.dedent("""\
                valid = raw[raw['value'] > 0].copy()
                print(f"valid rows: {len(valid)}")
            """),
                # Transform (explicit reassignment so cash tracks lineage)
                textwrap.dedent("""\
                valid = valid.assign(
                    normalized=(valid['value'] - valid['value'].mean()) / valid['value'].std()
                )
            """),
                # Aggregate
                textwrap.dedent("""\
                summary = valid.groupby('category').agg(
                    count=('id', 'count'),
                    avg_value=('value', 'mean')
                ).reset_index()
                print(summary.to_string(index=False))
            """),
                # Load
                textwrap.dedent(f"""\
                valid.to_csv('{out_str}', index=False)
                print(f"saved {{len(valid)}} rows")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid rows: 3" in nb_runner.get_output(3)
        assert "saved 3 rows" in nb_runner.get_output(6)

    def test_etl_modify_filter_and_rerun(self, nb_runner, tmp_path):
        """Modify filter criteria in ETL and re-run."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                "filtered = df[df['x'] > 2]",
                textwrap.dedent("""\
                result = filtered['y'].sum()
                print(f"sum={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x>2: y=30+40+50=120
        assert "sum=120" in nb_runner.get_output(4)

        # Change filter
        nb_runner.set_cell_source(3, "filtered = df[df['x'] > 3]")
        nb_runner.run_all()
        # x>3: y=40+50=90
        assert "sum=90" in nb_runner.get_output(4)
