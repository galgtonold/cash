"""
Real-world workflow simulations — data science, ML preprocessing,
report generation, and multi-phase analysis patterns.

Tests complete realistic notebook workflows that combine multiple features:
data loading, cleaning, transformation, analysis, and visualization prep.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


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
            "2024-01-05,Widget,15,5.99\n"
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
            "charlie,click,0.5\n"
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


class TestReportGenerationWorkflow:
    """Simulate report generation workflows."""

    def test_summary_statistics_report(self, nb_runner, tmp_path):
        """Generate a text summary report from data."""
        csv_path = tmp_path / "quarterly.csv"
        csv_path.write_text(
            "quarter,revenue,costs\nQ1,150000,120000\nQ2,175000,125000\nQ3,160000,130000\nQ4,200000,140000\n"
        )
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                "df['profit'] = df['revenue'] - df['costs']",
                textwrap.dedent("""\
                total_profit = df['profit'].sum()
                best_q = df.loc[df['profit'].idxmax(), 'quarter']
                print(f"Total Profit: {total_profit}")
                print(f"Best Quarter: {best_q}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "Total Profit: 170000" in output
        assert "Best Quarter: Q4" in output

    def test_report_with_data_update(self, nb_runner, tmp_path):
        """Update data file and regenerate report."""
        csv_path = tmp_path / "report_data.csv"
        csv_path.write_text("item,count\nA,10\nB,20\n")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                textwrap.dedent("""\
                total = df['count'].sum()
                print(f"Total: {total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Total: 30" in nb_runner.get_output(3)

        # Update data
        csv_path.write_text("item,count\nA,100\nB,200\nC,300\n")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "Total: 600" in nb_runner.get_output(3)


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
