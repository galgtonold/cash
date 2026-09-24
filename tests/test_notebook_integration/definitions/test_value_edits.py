"""Cells that compute plain values, edited: reassignments, constants, config dicts, pipelines."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.integration
class TestVariableReassignment:
    """Test variable reassignment and shadow patterns."""

    def test_progressive_refinement(self, nb_runner):
        """Progressively refine a variable across cells."""
        nb_runner.create_notebook(
            [
                "data = list(range(10))",
                "data = [x for x in data if x % 2 == 0]",  # filter evens
                "data = [x ** 2 for x in data]",  # square
                "print(data)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 4, 16, 36, 64]" in nb_runner.get_output(4)

    def test_conditional_variable_assignment(self, nb_runner):
        """Variable assigned conditionally across cells."""
        nb_runner.create_notebook(
            [
                "threshold = 50",
                "value = 75",
                textwrap.dedent("""\
                if value > threshold:
                    status = 'HIGH'
                else:
                    status = 'LOW'
                print(status)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HIGH" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "value = 25")
        nb_runner.run_all()
        assert "LOW" in nb_runner.get_output(3)


# Variable shadowing and scope interaction tests.
#
# Tests where variables are overwritten in later cells,
# edits change which version of a variable is used, and
# scoping rules interact with caching.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestVariableOverwriting:
    """Variable overwritten in subsequent cells."""

    def test_overwrite_then_edit_first_def(self, nb_runner):
        """Variable defined twice, edit first definition."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "x = x + 5",  # now x = 15
                "result = x * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit first definition
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "result = 210" in nb_runner.get_output(3)

    def test_three_overwrites_edit_middle(self, nb_runner):
        """Variable overwritten 3 times, edit middle one."""
        nb_runner.create_notebook(
            [
                "val = 1",
                "val = val + 10",
                "val = val * 2",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # val = (1+10)*2 = 22
        assert "val = 22" in nb_runner.get_output(4)

        # Edit middle
        nb_runner.set_cell_source(2, "val = val + 100")
        nb_runner.run_all()
        # val = (1+100)*2 = 202
        assert "val = 202" in nb_runner.get_output(4)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestMultipleVariables:
    """Multiple variables with interleaved definitions."""

    def test_introduce_new_variable(self, nb_runner):
        """Introduce a new variable mid-notebook."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "result = x * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Add a multiplier variable
        nb_runner.set_cell_source(1, "x = 5\nmultiplier = 10")
        nb_runner.set_cell_source(2, "result = x * multiplier\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)


# Tricky multi-cell variable shadowing, reassignment, deletion,
# and scope interactions that stress the lineage tracker.
@pytest.mark.integration
class TestConditionalAssignment:
    """Test conditional assignment patterns."""

    def test_ternary_expression(self, nb_runner):
        """Ternary expression across cells."""
        nb_runner.create_notebook(
            [
                "threshold = 50",
                "score = 75",
                textwrap.dedent("""\
                status = 'pass' if score >= threshold else 'fail'
                print(status)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pass" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "score = 30")
        nb_runner.run_all()
        assert "fail" in nb_runner.get_output(3)

    def test_or_default_pattern(self, nb_runner):
        """x = val or default pattern."""
        nb_runner.create_notebook(
            [
                "user_input = ''",
                "name = user_input or 'Anonymous'",
                "print(name)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Anonymous" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "user_input = 'Alice'")
        nb_runner.run_all()
        assert "Alice" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestConstantEdits:
    """Edit shared constants used by many cells."""

    def test_edit_constant_three_consumers(self, nb_runner):
        """One constant used by three cells."""
        nb_runner.create_notebook(
            [
                "PI = 3.14",
                "circumference = 2 * PI * 5\nprint(f'circ = {circumference}')",
                "area = PI * 5 ** 2\nprint(f'area = {area}')",
                "volume = (4/3) * PI * 5 ** 3\nprint(f'vol = {volume}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "circ = " in nb_runner.get_output(2)
        assert "area = " in nb_runner.get_output(3)
        assert "vol = " in nb_runner.get_output(4)

        # Use more precise PI
        nb_runner.set_cell_source(1, "PI = 3.14159")
        nb_runner.run_all()
        # All downstream should update
        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        out4 = nb_runner.get_output(4)
        assert "3.14159" in out2 or "31.4159" in out2
        assert "area = " in out3
        assert "vol = " in out4

    def test_edit_constant_with_restart(self, nb_runner):
        """Edit constant, restart, verify restored correctly."""
        nb_runner.create_notebook(
            [
                "MULTIPLIER = 5",
                "result = MULTIPLIER * 20\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Edit and run
        nb_runner.set_cell_source(1, "MULTIPLIER = 50")
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)

        # Restart - should restore
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)


# Global constant and config pattern interaction tests.
#
# Tests where shared constants/config are defined in an early cell
# and used by many downstream cells. Edit the config and verify
# all downstream cells update correctly.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestConfigDrivenWorkflow:
    """Config dict driving multiple downstream cells."""

    def test_edit_config_value(self, nb_runner):
        """Edit config value, verify all downstream updates."""
        nb_runner.create_notebook(
            [
                "config = {'scale': 2, 'offset': 10}",
                "a = 5 * config['scale']\nprint(f'a = {a}')",
                "b = 100 + config['offset']\nprint(f'b = {b}')",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10" in nb_runner.get_output(2)
        assert "b = 110" in nb_runner.get_output(3)
        assert "c = 120" in nb_runner.get_output(4)

        # Edit config
        nb_runner.set_cell_source(1, "config = {'scale': 10, 'offset': 0}")
        nb_runner.run_all()
        assert "a = 50" in nb_runner.get_output(2)
        assert "b = 100" in nb_runner.get_output(3)
        assert "c = 150" in nb_runner.get_output(4)

    def test_add_config_key(self, nb_runner):
        """Add a new key to config, use it downstream."""
        nb_runner.create_notebook(
            [
                "params = {'lr': 0.01}",
                "effective_lr = params['lr'] * 10\nprint(f'lr = {effective_lr}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lr = 0.1" in nb_runner.get_output(2)

        # Add key and use it
        nb_runner.set_cell_source(1, "params = {'lr': 0.01, 'decay': 0.5}")
        nb_runner.set_cell_source(
            2,
            "effective_lr = params['lr'] * params['decay']\nprint(f'lr = {effective_lr}')",
        )
        nb_runner.run_all()
        assert "lr = 0.005" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestPrintOutputEdits:
    """Print output + cell edits."""

    def test_edit_print_format(self, nb_runner):
        """Edit the format of a print statement."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "print(f'The answer is {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "The answer is 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "print(f'x = {x}')")
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_multiple_prints_edit(self, nb_runner):
        """Cell with multiple prints, edit one value."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = 2\nc = 3",
                "print(f'a={a}')\nprint(f'b={b}')\nprint(f'c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "a=1" in output
        assert "b=2" in output
        assert "c=3" in output

        nb_runner.set_cell_source(1, "a = 10\nb = 20\nc = 30")
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert "a=10" in output2
        assert "b=20" in output2
        assert "c=30" in output2


@pytest.mark.integration
class TestDatetimePatterns:
    """Test datetime handling across cells."""

    def test_datetime_creation_and_formatting(self, nb_runner):
        """Create and format datetime objects across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime, timedelta",
                textwrap.dedent("""\
                start = datetime(2024, 1, 15, 10, 30)
                end = start + timedelta(days=7, hours=3)
            """),
                textwrap.dedent("""\
                diff = end - start
                formatted = end.strftime('%Y-%m-%d %H:%M')
                print(formatted, diff.days)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "2024-01-22 13:30" in output
        assert "7" in output

    def test_timedelta_arithmetic(self, nb_runner):
        """Timedelta arithmetic across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import timedelta",
                textwrap.dedent("""\
                work_day = timedelta(hours=8)
                work_week = work_day * 5
                work_month = work_week * 4
            """),
                textwrap.dedent("""\
                total_hours = work_month.total_seconds() / 3600
                print(int(total_hours))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "160" in nb_runner.get_output(3)


# Complex real-world simulation: data science pipeline.
#
# Full end-to-end data science workflow: load data, clean,
# feature engineer, model (simple), evaluate — with edits
# at each stage.
@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestDataSciencePipeline:
    """Full data science pipeline simulation."""

    def test_full_pipeline_edit_source(self, nb_runner, tmp_path):
        """Full pipeline, edit source data."""
        csv = tmp_path / "dataset.csv"
        csv.write_text("feature,target\n1,10\n2,20\n3,30\n4,40\n5,50\n", encoding="utf-8")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "# Clean\ndf_clean = df.dropna()",
                "# Feature engineering\ndf_clean = df_clean.copy()\ndf_clean['feature_sq'] = df_clean['feature'] ** 2",
                "# Simple model: linear average\nmean_target = df_clean['target'].mean()\nprint(f'mean = {mean_target}')",
                "# Evaluate\nresiduals = [(t - mean_target) for t in df_clean['target']]\nrmse = (sum(r**2 for r in residuals) / len(residuals)) ** 0.5\nprint(f'rmse = {round(rmse, 2)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 30.0" in nb_runner.get_output(4)
        assert "rmse = " in nb_runner.get_output(5)

        # Edit source data
        csv.write_text("feature,target\n10,100\n20,200\n30,300\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 200.0" in nb_runner.get_output(4)

    def test_full_pipeline_edit_feature_engineering(self, nb_runner, tmp_path):
        """Edit feature engineering step."""
        csv = tmp_path / "data2.csv"
        csv.write_text("x,y\n1,2\n2,4\n3,6\n4,8\n", encoding="utf-8")
        csv_str = str(csv).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "# Feature\nfeature_sum = df['x'].sum()\nprint(f'feature_sum = {feature_sum}')",
            ]
        )
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


@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestETLPipeline:
    """Extract-Transform-Load pipeline simulation."""

    def test_etl_edit_transform(self, nb_runner):
        """ETL pipeline, edit transform step."""
        nb_runner.create_notebook(
            [
                "# Extract\nraw = [{'name': 'Alice', 'score': 85}, {'name': 'Bob', 'score': 92}]",
                "# Transform\ntransformed = [{'name': r['name'], 'grade': 'A' if r['score'] >= 90 else 'B'} for r in raw]",
                "# Load (print)\nfor t in transformed:\n    print(f\"{t['name']}: {t['grade']}\")",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "# Extract\ndata = [10, 20, 30]",
                "# Transform\nscaled = [x * 2 for x in data]",
                "# Load\nresult = sum(scaled)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(3)

        # Edit extract
        nb_runner.set_cell_source(1, "# Extract v2\ndata = [100, 200, 300, 400]")
        nb_runner.run_all()
        assert "result = 2000" in nb_runner.get_output(3)


@pytest.mark.integration
class TestSelectiveCellReexecution:
    """Test running subsets of cells (common notebook interaction pattern)."""

    def test_rerun_subset_of_cells(self, nb_runner):
        """Re-run only cells 2 and 3 out of 4."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a + 10",
                "c = b * 2",
                "print(c)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(4)

        # Re-run cells 2-4
        nb_runner.run_cells([2, 3, 4])
        assert "30" in nb_runner.get_output(4)

    def test_modify_and_rerun_downstream(self, nb_runner):
        """Modify a cell and only re-run it and downstream cells."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1",
                "z = y + 1",
                "print(z)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3" in nb_runner.get_output(4)

        # Modify cell 2 and run cells 2-4
        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cells([2, 3, 4])
        assert "102" in nb_runner.get_output(4)
