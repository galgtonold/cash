"""Notebooks that build a report, edited in their format or their data."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestReportBuilding:
    """Report generation workflow."""

    def test_build_report_string(self, nb_runner):
        """Build a report string from data, edit the data."""
        nb_runner.create_notebook(
            [
                "title = 'Sales Report'",
                "items = {'Widget A': 100, 'Widget B': 200}",
                "lines = [title, '=' * len(title)]\nfor name, count in items.items():\n    lines.append(f'{name}: {count}')\nreport = '\\n'.join(lines)\nprint(report)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Sales Report" in out
        assert "Widget A: 100" in out

        # Edit title
        nb_runner.set_cell_source(1, "title = 'Q4 Sales'")
        nb_runner.run_all()
        assert "Q4 Sales" in nb_runner.get_output(3)

    def test_summary_metrics(self, nb_runner):
        """Compute summary metrics, edit the input."""
        nb_runner.create_notebook(
            [
                "values = [10, 20, 30, 40, 50]  # input values",
                "metrics = {\n    'count': len(values),\n    'sum': sum(values),\n    'min': min(values),\n    'max': max(values),\n}",
                "for k, v in metrics.items():\n    print(f'{k}: {v}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count: 5" in out
        assert "sum: 150" in out

        nb_runner.set_cell_source(1, "values = [1, 2, 3]  # input values shorter")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count: 3" in out
        assert "sum: 6" in out


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestReportGeneration:
    """Simulate report generation with formatting changes."""

    def test_change_format_then_data(self, nb_runner):
        """Change formatting, then change data."""
        nb_runner.create_notebook(
            [
                "sales = [100, 200, 300, 400, 500]",
                "total = sum(sales)\navg = total / len(sales)",
                "report = f'Total: {total}, Avg: {avg}'\nprint(report)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Total: 1500, Avg: 300.0" in nb_runner.get_output(3)

        # Change format
        nb_runner.set_cell_source(
            3,
            "report = f'Sales Report: total=${total}, average=${avg}'\nprint(report)",
        )
        nb_runner.run_all()
        assert "Sales Report: total=$1500, average=$300.0" in nb_runner.get_output(3)

        # Change data
        nb_runner.set_cell_source(1, "sales = [1000, 2000, 3000]")
        nb_runner.run_all()
        assert "total=$6000, average=$2000.0" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
class TestReportGenerationWorkflow:
    """Simulate report generation workflows."""

    def test_summary_statistics_report(self, nb_runner, tmp_path):
        """Generate a text summary report from data."""
        csv_path = tmp_path / "quarterly.csv"
        csv_path.write_text(
            "quarter,revenue,costs\nQ1,150000,120000\nQ2,175000,125000\nQ3,160000,130000\nQ4,200000,140000\n",
            encoding="utf-8",
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
        csv_path.write_text("item,count\nA,10\nB,20\n", encoding="utf-8")
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
        csv_path.write_text("item,count\nA,100\nB,200\nC,300\n", encoding="utf-8")
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "Total: 600" in nb_runner.get_output(3)
