"""
Batch 16: Pandas integration patterns - DataFrame creation, transformation,
aggregation, merge/join, and file I/O caching behavior.

Tests real-world pandas workflows that are the primary use case for notebook
caching: data loading, cleaning, transformation pipelines, and analysis.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.files]


# ============================================================
# Test Group 1: DataFrame Creation & Basic Operations
# ============================================================

class TestDataFrameBasicOps:
    """Test basic pandas DataFrame operations across cells."""

    def test_dataframe_creation_and_query(self, nb_runner):
        """Create DataFrame in one cell, query in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                df = pd.DataFrame({
                    'name': ['Alice', 'Bob', 'Charlie'],
                    'age': [30, 25, 35],
                    'salary': [70000, 60000, 80000]
                })
            """),
            textwrap.dedent("""\
                avg_age = df['age'].mean()
                print(f"{avg_age:.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30.0" in nb_runner.get_output(2)

    def test_dataframe_column_operations(self, nb_runner):
        """Add/modify columns across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
            """),
            "df['y'] = df['x'] ** 2",
            "df['z'] = df['x'] + df['y']",
            textwrap.dedent("""\
                total = df['z'].sum()
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # z = x + x^2: 2+6+12+20+30 = 70
        assert "70" in nb_runner.get_output(4)

    def test_dataframe_filtering(self, nb_runner):
        """Filter DataFrame across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                df = pd.DataFrame({
                    'product': ['A', 'B', 'C', 'D', 'E'],
                    'price': [10, 25, 15, 30, 5],
                    'quantity': [100, 50, 75, 25, 200]
                })
            """),
            textwrap.dedent("""\
                expensive = df[df['price'] > 15]
                print(len(expensive))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "2" in nb_runner.get_output(2)  # B(25) and D(30)

    def test_dataframe_groupby(self, nb_runner):
        """GroupBy aggregation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                sales = pd.DataFrame({
                    'region': ['East', 'West', 'East', 'West', 'East'],
                    'product': ['A', 'A', 'B', 'B', 'A'],
                    'revenue': [100, 200, 150, 250, 300]
                })
            """),
            textwrap.dedent("""\
                by_region = sales.groupby('region')['revenue'].sum()
                print(by_region.to_dict())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "East" in output
        assert "550" in output  # 100+150+300
        assert "450" in output  # 200+250


# ============================================================
# Test Group 2: Pandas File I/O Caching
# ============================================================

class TestPandasFileIO:
    """Test pandas read/write operations and file dependency tracking."""

    def test_csv_read_cached(self, nb_runner, tmp_path):
        """CSV read should be tracked and cached."""
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{path_str}')",
            textwrap.dedent("""\
                total = df['a'].sum()
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "12" in nb_runner.get_output(3)  # 1+4+7

    def test_csv_change_invalidates_cache(self, nb_runner, tmp_path):
        """Modifying CSV file should invalidate cached reads."""
        csv_path = tmp_path / "changing_data.csv"
        csv_path.write_text("x,y\n1,10\n2,20\n")
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{path_str}')",
            textwrap.dedent("""\
                result = df['y'].sum()
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(3)

        # Modify CSV
        csv_path.write_text("x,y\n1,100\n2,200\n")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3)

    def test_csv_write_and_read_roundtrip(self, nb_runner, tmp_path):
        """Write CSV then read it back."""
        csv_path = tmp_path / "roundtrip.csv"
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            textwrap.dedent(f"""\
                df_out = pd.DataFrame({{'a': [10, 20], 'b': [30, 40]}})
                df_out.to_csv('{path_str}', index=False)
            """),
            textwrap.dedent(f"""\
                df_in = pd.read_csv('{path_str}')
                print(df_in['a'].sum(), df_in['b'].sum())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30 70" in nb_runner.get_output(3)


# ============================================================
# Test Group 3: DataFrame Merge/Join
# ============================================================

class TestDataFrameMergeJoin:
    """Test DataFrame merge and join operations across cells."""

    def test_inner_merge(self, nb_runner):
        """Inner merge of two DataFrames."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                users = pd.DataFrame({
                    'user_id': [1, 2, 3],
                    'name': ['Alice', 'Bob', 'Charlie']
                })
            """),
            textwrap.dedent("""\
                orders = pd.DataFrame({
                    'order_id': [101, 102, 103],
                    'user_id': [1, 2, 1],
                    'amount': [50, 75, 30]
                })
            """),
            textwrap.dedent("""\
                merged = pd.merge(users, orders, on='user_id')
                print(len(merged), merged['amount'].sum())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3 155" in nb_runner.get_output(3)

    def test_left_join(self, nb_runner):
        """Left join preserving all left rows."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                employees = pd.DataFrame({
                    'emp_id': [1, 2, 3, 4],
                    'name': ['A', 'B', 'C', 'D']
                })
                departments = pd.DataFrame({
                    'emp_id': [1, 3],
                    'dept': ['Sales', 'Engineering']
                })
            """),
            textwrap.dedent("""\
                result = employees.merge(departments, on='emp_id', how='left')
                null_count = result['dept'].isna().sum()
                print(len(result), null_count)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "4 2" in nb_runner.get_output(2)

    def test_concat_dataframes(self, nb_runner):
        """Concatenating DataFrames from different cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                df1 = pd.DataFrame({'x': [1, 2]})
                df2 = pd.DataFrame({'x': [3, 4]})
                df3 = pd.DataFrame({'x': [5, 6]})
            """),
            textwrap.dedent("""\
                combined = pd.concat([df1, df2, df3], ignore_index=True)
                print(len(combined), combined['x'].sum())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6 21" in nb_runner.get_output(2)


# ============================================================
# Test Group 4: DataFrame Transformation Pipelines
# ============================================================

class TestTransformationPipelines:
    """Test multi-step data transformation workflows."""

    def test_etl_pipeline(self, nb_runner, tmp_path):
        """Extract-Transform-Load pattern across cells."""
        csv_path = tmp_path / "raw_data.csv"
        csv_path.write_text("name,score,category\nalice,85,A\nbob,92,B\ncharlie,78,A\ndavid,95,B\neve,88,A\n")
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            f"import pandas as pd\nraw = pd.read_csv('{path_str}')",
            textwrap.dedent("""\
                # Transform: add computed columns
                raw['grade'] = raw['score'].apply(lambda s: 'Pass' if s >= 80 else 'Fail')
                transformed = raw.copy()
            """),
            textwrap.dedent("""\
                # Aggregate
                summary = transformed.groupby('category').agg(
                    avg_score=('score', 'mean'),
                    pass_count=('grade', lambda x: (x == 'Pass').sum())
                ).reset_index()
                print(summary.to_string(index=False))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "category" in output

    def test_pipeline_change_middle_step(self, nb_runner):
        """Change middle of a transformation pipeline."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                data = pd.DataFrame({'val': [10, 20, 30, 40, 50]})
            """),
            "data['doubled'] = data['val'] * 2",
            textwrap.dedent("""\
                result = data['doubled'].sum()
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3)

        # Change transformation
        nb_runner.set_cell_source(2, "data['doubled'] = data['val'] * 3")
        nb_runner.run_all()
        assert "450" in nb_runner.get_output(3)

    def test_chained_pandas_methods(self, nb_runner):
        """Chained method calls on DataFrame."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                df = pd.DataFrame({
                    'name': ['Alice', 'Bob', 'Charlie', 'David'],
                    'dept': ['Sales', 'Eng', 'Sales', 'Eng'],
                    'salary': [70000, 80000, 65000, 90000]
                })
            """),
            textwrap.dedent("""\
                result = (df
                    .query("salary > 67000")
                    .groupby('dept')['salary']
                    .mean()
                    .sort_values(ascending=False)
                )
                print(result.iloc[0])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # Eng: (80000+90000)/2=85000, Sales: 70000 only
        assert "85000" in nb_runner.get_output(2)

    def test_pivot_table(self, nb_runner):
        """Pivot table across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                sales = pd.DataFrame({
                    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
                    'product': ['A', 'B', 'A', 'B'],
                    'revenue': [100, 200, 150, 250]
                })
            """),
            textwrap.dedent("""\
                pivot = sales.pivot_table(
                    values='revenue',
                    index='date',
                    columns='product',
                    aggfunc='sum'
                )
                print(pivot.to_string())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "100" in output
        assert "250" in output
