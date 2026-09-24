"""pandas frames edited and re-run across cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.integration
@pytest.mark.files
class TestPandasFileIO:
    """Test pandas read/write operations and file dependency tracking."""

    def test_csv_read_cached(self, nb_runner, tmp_path):
        """CSV read should be tracked and cached."""
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                textwrap.dedent("""\
                total = df['a'].sum()
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "12" in nb_runner.get_output(3)  # 1+4+7

    def test_csv_change_invalidates_cache(self, nb_runner, tmp_path):
        """Modifying CSV file should invalidate cached reads."""
        csv_path = tmp_path / "changing_data.csv"
        csv_path.write_text("x,y\n1,10\n2,20\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                textwrap.dedent("""\
                result = df['y'].sum()
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(3)

        # Modify CSV
        csv_path.write_text("x,y\n1,100\n2,200\n", encoding="utf-8")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3)

    def test_csv_write_and_read_roundtrip(self, nb_runner, tmp_path):
        """Write CSV then read it back."""
        csv_path = tmp_path / "roundtrip.csv"
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                textwrap.dedent(f"""\
                df_out = pd.DataFrame({{'a': [10, 20], 'b': [30, 40]}})
                df_out.to_csv('{path_str}', index=False)
            """),
                textwrap.dedent(f"""\
                df_in = pd.read_csv('{path_str}')
                print(df_in['a'].sum(), df_in['b'].sum())
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30 70" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.files
class TestTransformationPipelines:
    """Test multi-step data transformation workflows."""

    def test_etl_pipeline(self, nb_runner, tmp_path):
        """Extract-Transform-Load pattern across cells."""
        csv_path = tmp_path / "raw_data.csv"
        csv_path.write_text(
            "name,score,category\nalice,85,A\nbob,92,B\ncharlie,78,A\ndavid,95,B\neve,88,A\n", encoding="utf-8"
        )
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "category" in output

    def test_pipeline_change_middle_step(self, nb_runner):
        """Change middle of a transformation pipeline."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import pandas as pd
                data = pd.DataFrame({'val': [10, 20, 30, 40, 50]})
            """),
                "data['doubled'] = data['val'] * 2",
                textwrap.dedent("""\
                result = data['doubled'].sum()
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3)

        # Change transformation
        nb_runner.set_cell_source(2, "data['doubled'] = data['val'] * 3")
        nb_runner.run_all()
        assert "450" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestDataFrameEdits:
    """DataFrame manipulation with cell edits."""

    def test_edit_filter_condition(self, nb_runner, tmp_path):
        """Edit DataFrame filter condition."""
        csv = tmp_path / "data.csv"
        csv.write_text("name,score\nAlice,85\nBob,92\nCharlie,78\nDiana,95\n", encoding="utf-8")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "high = df[df['score'] > 90]\nprint(f'count = {len(high)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 2" in nb_runner.get_output(2)

        # Lower threshold
        nb_runner.set_cell_source(2, "high = df[df['score'] > 80]\nprint(f'count = {len(high)}')")
        nb_runner.run_all()
        assert "count = 3" in nb_runner.get_output(2)

    def test_edit_aggregation(self, nb_runner, tmp_path):
        """Edit aggregation function."""
        csv = tmp_path / "sales.csv"
        csv.write_text("product,amount\nA,100\nB,200\nA,150\nB,300\n", encoding="utf-8")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "result = df['amount'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 750" in nb_runner.get_output(2)

        # Change to mean
        nb_runner.set_cell_source(2, "result = df['amount'].mean()\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 187.5" in nb_runner.get_output(2)

    def test_edit_column_selection(self, nb_runner, tmp_path):
        """Edit which columns are selected."""
        csv = tmp_path / "multi.csv"
        csv.write_text("a,b,c\n1,10,100\n2,20,200\n3,30,300\n", encoding="utf-8")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "total = df['a'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Switch to column c
        nb_runner.set_cell_source(2, "total = df['c'].sum()\nprint(f'total = {total}')")
        nb_runner.run_all()
        assert "total = 600" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestDataFrameChainEdits:
    """Multi-step DataFrame pipelines with edits."""

    def test_filter_then_aggregate_edit_filter(self, nb_runner, tmp_path):
        """Filter → aggregate, edit filter."""
        csv = tmp_path / "records.csv"
        csv.write_text("cat,val\nX,10\nY,20\nX,30\nY,40\nX,50\n", encoding="utf-8")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "subset = df[df['cat'] == 'X']",
                "total = subset['val'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 90" in nb_runner.get_output(3)

        # Change filter to Y
        nb_runner.set_cell_source(2, "subset = df[df['cat'] == 'Y']")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(3)
