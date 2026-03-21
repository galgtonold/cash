"""
Batch 31: Multi-step data transformation pipelines — realistic ETL-like workflows
with many intermediate variables and complex data flow.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestETLPipeline:
    """Test caching with ETL-like transformation pipelines."""

    def test_extract_transform_load(self, nb_runner, tmp_path):
        """Full ETL pipeline: extract from CSV, transform, write output."""
        input_csv = tmp_path / "raw.csv"
        output_csv = tmp_path / "clean.csv"
        input_csv.write_text(
            "id,name,value,category\n"
            "1,Alice,100,A\n"
            "2,Bob,-5,B\n"
            "3,Charlie,200,A\n"
            "4,Diana,150,B\n"
            "5,Eve,-10,A\n"
        )
        in_str = str(input_csv).replace('\\', '/')
        out_str = str(output_csv).replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid rows: 3" in nb_runner.get_output(3)
        assert "saved 3 rows" in nb_runner.get_output(6)

    def test_etl_modify_filter_and_rerun(self, nb_runner, tmp_path):
        """Modify filter criteria in ETL and re-run."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n"
        )
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{path_str}')",
            "filtered = df[df['x'] > 2]",
            textwrap.dedent("""\
                result = filtered['y'].sum()
                print(f"sum={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x>2: y=30+40+50=120
        assert "sum=120" in nb_runner.get_output(4)

        # Change filter
        nb_runner.set_cell_source(3, "filtered = df[df['x'] > 3]")
        nb_runner.run_all()
        # x>3: y=40+50=90
        assert "sum=90" in nb_runner.get_output(4)


class TestMultiStepTransformations:
    """Test complex multi-step data transformations."""

    def test_chained_transformations(self, nb_runner):
        """Chain of transformations on a DataFrame."""
        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                df = pd.DataFrame({
                    'name': ['A', 'B', 'C', 'D', 'E'],
                    'score': [85, 92, 78, 95, 88],
                    'weight': [1.0, 1.5, 1.0, 2.0, 1.5]
                })
            """),
            # Step 1: weighted score
            "df['weighted'] = df['score'] * df['weight']",
            # Step 2: rank
            "df['rank'] = df['weighted'].rank(ascending=False).astype(int)",
            # Step 3: grade
            textwrap.dedent("""\
                def assign_grade(score):
                    if score >= 90: return 'A'
                    if score >= 80: return 'B'
                    return 'C'
                df['grade'] = df['score'].apply(assign_grade)
            """),
            # Step 4: Summary
            textwrap.dedent("""\
                print(f"top={df.loc[df['rank']==1, 'name'].values[0]}")
                print(f"grades={df['grade'].value_counts().to_dict()}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(6)
        assert "top=D" in output  # D has 95*2.0=190
        assert "grades=" in output

    def test_transformation_branch_and_merge(self, nb_runner):
        """Two transformation branches that merge."""
        nb_runner.create_notebook([
            "import pandas as pd",
            textwrap.dedent("""\
                df = pd.DataFrame({
                    'id': [1, 2, 3, 4],
                    'revenue': [100, 200, 150, 300],
                    'cost': [80, 120, 100, 180]
                })
            """),
            # Branch A: revenue metrics
            textwrap.dedent("""\
                rev_stats = pd.DataFrame({
                    'id': df['id'],
                    'rev_pct': df['revenue'] / df['revenue'].sum() * 100
                })
            """),
            # Branch B: cost metrics
            textwrap.dedent("""\
                cost_stats = pd.DataFrame({
                    'id': df['id'],
                    'cost_pct': df['cost'] / df['cost'].sum() * 100
                })
            """),
            # Merge
            textwrap.dedent("""\
                merged = rev_stats.merge(cost_stats, on='id')
                merged['efficiency'] = merged['rev_pct'] / merged['cost_pct']
                best = merged.loc[merged['efficiency'].idxmax(), 'id']
                print(f"most_efficient_id={best}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(5)
        assert "most_efficient_id=" in output


class TestPivotAndReshape:
    """Test pivot/melt/reshape operations across cells."""

    def test_pivot_table_across_cells(self, nb_runner):
        """Pivot table created in one cell, used in another."""
        nb_runner.create_notebook([
            "import pandas as pd",
            textwrap.dedent("""\
                sales = pd.DataFrame({
                    'region': ['East', 'East', 'West', 'West'],
                    'product': ['A', 'B', 'A', 'B'],
                    'amount': [100, 150, 200, 120]
                })
            """),
            textwrap.dedent("""\
                pivot = sales.pivot_table(values='amount', index='region',
                                          columns='product', aggfunc='sum')
            """),
            textwrap.dedent("""\
                print(f"East_A={pivot.loc['East', 'A']} West_B={pivot.loc['West', 'B']}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "East_A=100 West_B=120" in nb_runner.get_output(4)


class TestAggregationWorkflows:
    """Test aggregation-heavy workflows."""

    def test_rolling_window(self, nb_runner):
        """Rolling window calculations across cells."""
        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                ts = pd.Series(np.random.randn(20).cumsum(), name='value')
            """),
            textwrap.dedent("""\
                rolling_mean = ts.rolling(5).mean()
                rolling_std = ts.rolling(5).std()
            """),
            textwrap.dedent("""\
                # Count how many values are above rolling mean
                above = (ts[4:] > rolling_mean[4:]).sum()
                print(f"above_mean={above}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "above_mean=" in output

    def test_cumulative_operations(self, nb_runner):
        """Cumulative sum/product across cells."""
        nb_runner.create_notebook([
            "import pandas as pd",
            textwrap.dedent("""\
                df = pd.DataFrame({
                    'day': range(1, 8),
                    'sales': [10, 15, 8, 22, 18, 25, 12]
                })
            """),
            textwrap.dedent("""\
                df['cumsum'] = df['sales'].cumsum()
                df['cummax'] = df['sales'].cummax()
            """),
            textwrap.dedent("""\
                final_cumsum = df['cumsum'].iloc[-1]
                final_cummax = df['cummax'].iloc[-1]
                print(f"cumsum={final_cumsum} cummax={final_cummax}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10+15+8+22+18+25+12=110, max=25
        assert "cumsum=110 cummax=25" in nb_runner.get_output(4)
