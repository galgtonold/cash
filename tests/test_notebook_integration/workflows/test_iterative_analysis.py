"""Analysis refined over many rounds: exploring, tuning, retraining, debugging, editing back and forth."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestRealWorldDataScience:
    """Tests mimicking real-world data science notebook workflows."""

    @pytest.mark.libraries
    def test_sklearn_like_pipeline(self, nb_runner):
        """Simulate a typical sklearn train/predict pipeline."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                "np.random.seed(42)\nX = np.random.randn(100, 3)\ny = (X[:, 0] + X[:, 1] * 2 > 0).astype(int)",
                "# Simple manual logistic regression-like scoring\nweights = np.array([1.0, 2.0, 0.0])\nscores = X @ weights\npredictions = (scores > 0).astype(int)",
                "accuracy = np.mean(predictions == y)\nprint(f'accuracy={accuracy:.4f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "accuracy=" in output
        acc_val = float(output.split("=")[1])
        assert acc_val > 0.5  # should be reasonably accurate

        # Re-run: should be cached
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert output2.strip() == output.strip()

    @pytest.mark.libraries
    def test_regex_text_processing(self, nb_runner):
        """Regex-based text processing common in NLP notebooks."""
        nb_runner.create_notebook(
            [
                "import re",
                "text = 'Hello World! This is test 123. Contact: user@email.com'",
                "emails = re.findall(r'[\\w.]+@[\\w.]+', text)\nnumbers = re.findall(r'\\d+', text)\nwords = re.findall(r'\\b[A-Z][a-z]+\\b', text)",
                "print(f'emails={emails} numbers={numbers} words={words}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "user@email.com" in output
        assert "123" in output

        # Change text
        nb_runner.set_cell_source(2, "text = 'New text with number 456 and admin@site.org'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "admin@site.org" in output2
        assert "456" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestNotebookIterativeWorkflow:
    """Tests for realistic iterative development patterns."""

    @pytest.mark.core
    def test_iterative_refinement(self, nb_runner):
        """Simulate iterative development: define, test, refine."""
        nb_runner.create_notebook(
            [
                "def process(data):\n    return [x * 2 for x in data]",
                "test_data = [1, 2, 3, 4, 5]",
                "result = process(test_data)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "2, 4, 6, 8, 10" in output

        # Refine function
        nb_runner.set_cell_source(1, "def process(data):\n    return [x ** 2 for x in data]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "1, 4, 9, 16, 25" in output2

    @pytest.mark.core
    def test_parameter_sweep_pattern(self, nb_runner):
        """Simulate changing parameters and re-running analysis."""
        nb_runner.create_notebook(
            [
                "threshold = 0.5",
                "data = [0.1, 0.3, 0.5, 0.7, 0.9]",
                "above = [x for x in data if x > threshold]\nbelow = [x for x in data if x <= threshold]",
                "print(f'above={len(above)} below={len(below)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "above=2" in output
        assert "below=3" in output

        # Change threshold
        nb_runner.set_cell_source(1, "threshold = 0.3")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "above=3" in output2
        assert "below=2" in output2


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestDataExplorationWorkflow:
    """Simulate a user exploring data iteratively."""

    def test_explore_then_refine(self, nb_runner):
        """User explores, then refines analysis."""
        nb_runner.create_notebook(
            [
                "data = list(range(1, 21))",
                "mean_val = sum(data) / len(data)\nprint(f'mean = {mean_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 10.5" in nb_runner.get_output(2)

        # User decides to filter first
        nb_runner.set_cell_source(
            2,
            "filtered = [x for x in data if x > 10]\nmean_val = sum(filtered) / len(filtered)\nprint(f'mean = {mean_val}')",
        )
        nb_runner.run_all()
        assert "mean = 15.5" in nb_runner.get_output(2)

        # User changes the data source
        nb_runner.set_cell_source(1, "data = list(range(1, 101))")
        nb_runner.run_all()
        assert "mean = 55.5" in nb_runner.get_output(2)

    def test_iterative_parameter_tuning(self, nb_runner):
        """User tunes parameters across multiple iterations."""
        nb_runner.create_notebook(
            [
                "threshold = 50\nscale = 2",
                "data = list(range(100))",
                "filtered = [x for x in data if x > threshold]",
                "result = sum(x * scale for x in filtered)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        assert "result = " in output1

        # Tune threshold
        nb_runner.set_cell_source(1, "threshold = 75\nscale = 2")
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert "result = " in output2

        # Tune scale
        nb_runner.set_cell_source(1, "threshold = 75\nscale = 10")
        nb_runner.run_all()
        output3 = nb_runner.get_output(4)
        assert "result = " in output3

        # Values should be different
        assert output1 != output2 or output2 != output3


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestExploratoryAnalysis:
    """Simulate exploratory data analysis with back-and-forth edits."""

    def test_explore_and_refine(self, nb_runner):
        """Explore data, refine analysis, go back and change approach."""
        nb_runner.create_notebook(
            [
                "data = [3, 7, 2, 9, 1, 5, 8, 4, 6, 10]",
                "# Approach 1: simple sort\nsorted_data = sorted(data)",
                "top3 = sorted_data[-3:]\nprint(f'top3 = {top3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3 = [8, 9, 10]" in nb_runner.get_output(3)

        # Refine: use different sort approach
        nb_runner.set_cell_source(2, "# Approach 2: reverse sort\nsorted_data = sorted(data, reverse=True)")
        nb_runner.set_cell_source(3, "top3 = sorted_data[:3]\nprint(f'top3 = {top3}')")
        nb_runner.run_all()
        assert "top3 = [10, 9, 8]" in nb_runner.get_output(3)

    def test_multi_round_exploration(self, nb_runner):
        """Multiple rounds of exploration on the same data."""
        nb_runner.create_notebook(
            [
                "nums = [1, 4, 9, 16, 25]",
                "# Analysis\nresult = sum(nums)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 55" in nb_runner.get_output(2)

        # Edit 2: average
        nb_runner.set_cell_source(2, "# Analysis\nresult = sum(nums) / len(nums)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 11.0" in nb_runner.get_output(2)

        # Edit 3: max - min range
        nb_runner.set_cell_source(2, "# Analysis\nresult = max(nums) - min(nums)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestDataAnalysisWorkflow:
    """Simulate an iterative data analysis workflow."""

    def test_parameter_tuning_workflow(self, nb_runner):
        """Simulate parameter tuning — change params and re-evaluate."""
        nb_runner.create_notebook(
            [
                "data = list(range(1, 101))",
                "threshold = 50",
                "filtered = [x for x in data if x > threshold]",
                "count = len(filtered)\navg = sum(filtered) / count if count else 0",
                "print(f'count={count}, avg={avg:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # Above 50: 51..100 = 50 items, avg = 75.5
        assert "count=50, avg=75.5" in nb_runner.get_output(5)

        # Tune threshold down
        nb_runner.set_cell_source(2, "threshold = 80")
        nb_runner.run_all()
        # Above 80: 81..100 = 20 items, avg = 90.5
        assert "count=20, avg=90.5" in nb_runner.get_output(5)

        # Tune threshold up
        nb_runner.set_cell_source(2, "threshold = 95")
        nb_runner.run_all()
        # Above 95: 96..100 = 5 items, avg = 98.0
        assert "count=5, avg=98.0" in nb_runner.get_output(5)

    def test_feature_engineering_workflow(self, nb_runner):
        """Simulate feature engineering with iterative changes."""
        nb_runner.create_notebook(
            [
                "raw = [10, 20, 30, 40, 50]",
                "def transform(data):\n    return [x / max(data) for x in data]",
                "features = transform(raw)",
                "score = sum(features)\nprint(f'score = {score:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0.2 + 0.4 + 0.6 + 0.8 + 1.0 = 3.0
        assert "score = 3.00" in nb_runner.get_output(4)

        # Change transform
        nb_runner.set_cell_source(
            2, "def transform(data):\n    mean = sum(data) / len(data)\n    return [(x - mean) / mean for x in data]"
        )
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "score = " in output  # should be 0.0 (mean-centered)

    def test_etl_pipeline_edit(self, nb_runner):
        """ETL pipeline: extract → transform → load. Edit transform."""
        nb_runner.create_notebook(
            [
                "# Extract\nraw_records = [{'name': 'A', 'val': 10}, {'name': 'B', 'val': 20}, {'name': 'C', 'val': 30}]",
                "# Transform\ntransformed = {r['name']: r['val'] * 2 for r in raw_records}",
                "# Load\ntotal = sum(transformed.values())\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (10+20+30)*2 = 120
        assert "total = 120" in nb_runner.get_output(3)

        # Change transform logic
        nb_runner.set_cell_source(2, "# Transform\ntransformed = {r['name']: r['val'] ** 2 for r in raw_records}")
        nb_runner.run_all()
        # 100+400+900 = 1400
        assert "total = 1400" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestDataScienceWorkflow:
    """Simulate a complete data science exploration workflow."""

    def test_eda_workflow(self, nb_runner, tmp_path):
        """Full EDA workflow: load data, clean, analyze, summarize."""
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text(
            "date,product,quantity,price\n"
            "2024-01-01,Widget,10,5.99\n"
            "2024-01-02,Gadget,5,12.99\n"
            "2024-01-03,Widget,8,5.99\n"
            "2024-01-04,Gadget,12,12.99\n"
            "2024-01-05,Widget,15,5.99\n",
            encoding="utf-8",
        )
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd\nimport numpy as np",
                f"df = pd.read_csv('{path_str}')",
                "df['revenue'] = df['quantity'] * df['price']",
                textwrap.dedent("""\
                summary = df.groupby('product').agg(
                    total_qty=('quantity', 'sum'),
                    total_rev=('revenue', 'sum'),
                    avg_price=('price', 'mean')
                ).reset_index()
                print(summary.to_string(index=False))
            """),
                textwrap.dedent("""\
                total_revenue = df['revenue'].sum()
                print(f"Total: {total_revenue:.2f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output4 = nb_runner.get_output(4)
        assert "Widget" in output4
        assert "Gadget" in output4

        output5 = nb_runner.get_output(5)
        assert "Total:" in output5

    def test_eda_iterate_analysis(self, nb_runner, tmp_path):
        """Iterate on analysis: change aggregation, see updated results."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text(
            "user,action,duration\n"
            "alice,click,1.2\n"
            "bob,click,0.8\n"
            "alice,scroll,2.5\n"
            "bob,scroll,3.1\n"
            "charlie,click,0.5\n",
            encoding="utf-8",
        )
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                textwrap.dedent("""\
                result = df.groupby('action')['duration'].mean()
                print(result.to_dict())
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "click" in output
        assert "scroll" in output

        # Change aggregation
        nb_runner.set_cell_source(
            3,
            textwrap.dedent("""\
            result = df.groupby('user')['duration'].sum()
            print(result.to_dict())
        """),
        )
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "alice" in output
        assert "bob" in output


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestModelRetraining:
    """Simulate a model training workflow with re-iterations."""

    def test_change_hyperparameters(self, nb_runner):
        """Change hyperparameters and retrain."""
        nb_runner.create_notebook(
            [
                "# Hyperparams\nlr = 0.01\nepochs = 10",
                "# Training sim\nimport random\nrandom.seed(42)\nloss = 1.0\nfor e in range(epochs):\n    loss *= (1 - lr)\nfinal_loss = round(loss, 4)",
                "print(f'loss = {final_loss}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(3)
        assert "loss = " in output1

        # Increase learning rate
        nb_runner.set_cell_source(1, "# Hyperparams\nlr = 0.1\nepochs = 10")
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert "loss = " in output2

        # Higher lr → lower loss
        loss1 = float(output1.split("loss = ")[1].strip())
        loss2 = float(output2.split("loss = ")[1].strip())
        assert loss2 < loss1


@pytest.mark.integration
@pytest.mark.stress
class TestMultiPhaseAnalysis:
    """Test multi-phase analysis patterns."""

    def test_phase_1_to_3_pipeline(self, nb_runner):
        """Three-phase analysis: prep, analyze, conclude."""
        nb_runner.create_notebook(
            [
                # Phase 1: Data Preparation
                textwrap.dedent("""\
                import pandas as pd
                raw_data = {
                    'name': ['Product_A', 'Product_B', 'Product_C'],
                    'q1': [100, 200, 150],
                    'q2': [120, 180, 160],
                    'q3': [110, 220, 170],
                    'q4': [130, 250, 180]
                }
                df = pd.DataFrame(raw_data)
            """),
                # Phase 2: Analysis
                textwrap.dedent("""\
                df['annual'] = df[['q1', 'q2', 'q3', 'q4']].sum(axis=1)
                df['avg_quarterly'] = df['annual'] / 4
            """),
                # Phase 3: Conclusion
                textwrap.dedent("""\
                top_product = df.loc[df['annual'].idxmax(), 'name']
                total_market = df['annual'].sum()
                print(f"Top: {top_product}, Market: {total_market}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Top: Product_B" in output
        assert "Market: 1970" in output

    def test_iterative_model_improvement(self, nb_runner):
        """Iteratively improve a simple model by changing parameters."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                data = np.random.randn(50)
            """),
                textwrap.dedent("""\
                # Simple moving average with window=3
                window = 3
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                variance = np.var(smoothed)
                print(f"window={window} var={variance:.4f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(3)
        assert "window=3" in output1

        # Try larger window
        nb_runner.set_cell_source(
            3,
            textwrap.dedent("""\
            # Simple moving average with window=7
            window = 7
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            variance = np.var(smoothed)
            print(f"window={window} var={variance:.4f}")
        """),
        )
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert "window=7" in output2

    def test_ab_testing_workflow(self, nb_runner):
        """A/B testing analysis workflow."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                control = np.random.binomial(1, 0.10, 1000)  # 10% conversion
                treatment = np.random.binomial(1, 0.12, 1000)  # 12% conversion
            """),
                textwrap.dedent("""\
                control_rate = control.mean()
                treatment_rate = treatment.mean()
                lift = (treatment_rate - control_rate) / control_rate * 100
            """),
                textwrap.dedent("""\
                print(f"Control: {control_rate:.3f}")
                print(f"Treatment: {treatment_rate:.3f}")
                print(f"Lift: {lift:.1f}%")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "Control:" in output
        assert "Treatment:" in output
        assert "Lift:" in output


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestConfigDrivenWorkflow:
    """Configuration-driven workflows with config edits."""

    def test_config_dict_workflow(self, nb_runner):
        """Change config dict, verify pipeline adjusts."""
        nb_runner.create_notebook(
            [
                "config = {'scale': 2, 'offset': 10}",
                "data = [1, 2, 3, 4, 5]",
                "processed = [x * config['scale'] + config['offset'] for x in data]",
                "result = sum(processed)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (1*2+10) + (2*2+10) + ... + (5*2+10) = 12+14+16+18+20 = 80
        assert "result = 80" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "config = {'scale': 10, 'offset': 0}")
        nb_runner.run_all()
        # 10+20+30+40+50 = 150
        assert "result = 150" in nb_runner.get_output(4)

    def test_multi_config_changes(self, nb_runner):
        """Change config multiple times in sequence."""
        nb_runner.create_notebook(
            [
                "multiplier = 1",
                "result = 42 * multiplier\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

        for m in [2, 5, 10, 100]:
            nb_runner.set_cell_source(1, f"multiplier = {m}")
            nb_runner.run_all()
            assert f"result = {42 * m}" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestDebuggingWorkflow:
    """Simulate debugging workflow — add prints, fix, remove prints."""

    def test_add_debug_fix_remove(self, nb_runner):
        """Add debug output, fix bug, remove debug."""
        nb_runner.create_notebook(
            [
                "numbers = [1, 2, 3, 4, 5]",
                "# Bug: using wrong formula\nresult = sum(numbers) / (len(numbers) + 1)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # Bug: divides by 6 instead of 5
        output = nb_runner.get_output(2)
        assert "result = 2.5" in output

        # Fix the bug
        nb_runner.set_cell_source(
            2,
            "# Fixed formula\nresult = sum(numbers) / len(numbers)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 3.0" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestBackAndForthEditing:
    """User goes back and forth between cells."""

    def test_edit_cell1_then_cell3_then_cell1_again(self, nb_runner):
        """Edit cell 1, then cell 3, then cell 1 again."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "mid = base * 2",
                "final = mid + 5\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final = 25" in nb_runner.get_output(3)

        # Edit cell 1
        nb_runner.set_cell_source(1, "base = 20")
        nb_runner.run_all()
        assert "final = 45" in nb_runner.get_output(3)

        # Edit cell 3
        nb_runner.set_cell_source(3, "final = mid + 100\nprint(f'final = {final}')")
        nb_runner.run_all()
        assert "final = 140" in nb_runner.get_output(3)

        # Edit cell 1 again
        nb_runner.set_cell_source(1, "base = 1")
        nb_runner.run_all()
        assert "final = 102" in nb_runner.get_output(3)

    def test_oscillate_between_two_values(self, nb_runner):
        """Toggle a value back and forth — each cell gets unique code."""
        nb_runner.create_notebook(
            [
                "mode = 'A'",
                "result = 100 if mode == 'A' else 200\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Switch to B
        nb_runner.set_cell_source(1, "mode = 'B'")
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)

        # Switch back to A with different code
        nb_runner.set_cell_source(1, "mode = 'A'  # restored")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    def test_sequential_distinct_changes(self, nb_runner):
        """Make many distinct changes to same cell — each unique."""
        nb_runner.create_notebook(
            [
                "val = 1",
                "out = val * 10\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 10" in nb_runner.get_output(2)

        for v in [2, 3, 5, 7]:
            nb_runner.set_cell_source(1, f"val = {v}")
            nb_runner.run_all()
            assert f"out = {v * 10}" in nb_runner.get_output(2)
