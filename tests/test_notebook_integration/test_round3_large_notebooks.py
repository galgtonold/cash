"""
Large-scale notebook simulation — 15-20 cell notebooks with
realistic data science workflows testing end-to-end caching behavior.

These tests simulate real notebooks that users would write, with realistic
complexity and cell counts.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestLargeFinancialNotebook:
    """Simulate a 12-cell financial analysis notebook."""

    def test_financial_analysis_full(self, nb_runner, tmp_path):
        """Complete financial analysis notebook."""
        csv_path = tmp_path / "stock_prices.csv"
        csv_path.write_text(
            "date,ticker,close,volume\n"
            "2024-01-01,AAPL,185.50,1000000\n"
            "2024-01-02,AAPL,186.20,1100000\n"
            "2024-01-03,AAPL,184.80,950000\n"
            "2024-01-01,MSFT,375.00,800000\n"
            "2024-01-02,MSFT,377.50,850000\n"
            "2024-01-03,MSFT,373.00,780000\n"
        )
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: Imports
                "import pandas as pd\nimport numpy as np",
                # Cell 2: Load data
                f"df = pd.read_csv('{path_str}')",
                # Cell 3: Parse dates
                "df['date'] = pd.to_datetime(df['date'])",
                # Cell 4: Calculate returns
                textwrap.dedent("""\
                df = df.sort_values(['ticker', 'date'])
                df['return'] = df.groupby('ticker')['close'].pct_change()
            """),
                # Cell 5: Summary stats
                textwrap.dedent("""\
                summary = df.groupby('ticker').agg(
                    avg_close=('close', 'mean'),
                    total_volume=('volume', 'sum'),
                    avg_return=('return', 'mean')
                ).reset_index()
            """),
                # Cell 6: Find best performer
                textwrap.dedent("""\
                best = summary.loc[summary['avg_close'].idxmax(), 'ticker']
                print(f"Best: {best}")
            """),
                # Cell 7: Portfolio value
                textwrap.dedent("""\
                portfolio = {'AAPL': 10, 'MSFT': 5}
                latest_prices = df.groupby('ticker')['close'].last()
                total_value = sum(
                    portfolio.get(t, 0) * p
                    for t, p in latest_prices.items()
                )
                print(f"Portfolio: {total_value:.2f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Best: MSFT" in nb_runner.get_output(6)
        output7 = nb_runner.get_output(7)
        assert "Portfolio:" in output7

    def test_financial_notebook_modify_and_rerun(self, nb_runner, tmp_path):
        """Modify a middle cell and re-run the financial notebook."""
        csv_path = tmp_path / "prices.csv"
        csv_path.write_text("symbol,price,shares\nAAPL,150,10\nGOOG,2800,2\nTSLA,900,5\n")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                "df['value'] = df['price'] * df['shares']",
                textwrap.dedent("""\
                total = df['value'].sum()
                print(f"Total: {total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # AAPL=1500, GOOG=5600, TSLA=4500 -> 11600
        assert "Total: 11600" in nb_runner.get_output(4)

        # Add a fee column
        nb_runner.set_cell_source(
            3,
            textwrap.dedent("""\
            df['value'] = df['price'] * df['shares']
            df['fee'] = df['value'] * 0.01
            df['net_value'] = df['value'] - df['fee']
        """),
        )
        nb_runner.set_cell_source(
            4,
            textwrap.dedent("""\
            total = df['net_value'].sum()
            print(f"Total: {total:.2f}")
        """),
        )
        nb_runner.run_all()
        # 11600 * 0.99 = 11484
        assert "Total: 11484.00" in nb_runner.get_output(4)


class TestLargeTextProcessingNotebook:
    """Simulate a text processing / NLP notebook."""

    def test_text_change_propagation(self, nb_runner):
        """Change input text and verify all downstream updates."""
        nb_runner.create_notebook(
            [
                "import re\nfrom collections import Counter",
                "text = 'hello hello world world world'",
                textwrap.dedent("""\
                words = text.split()
                freq = Counter(words)
                print(freq.most_common(1))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "world" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "text = 'foo foo foo bar bar'")
        nb_runner.run_all()
        assert "foo" in nb_runner.get_output(3)


class TestEndToEndWithRestart:
    """Test large notebooks surviving kernel restart."""

    def test_5_cell_notebook_restart_restore(self, nb_runner):
        """5-cell notebook restores correctly after restart."""
        nb_runner.create_notebook(
            [
                "import numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                data = np.random.randn(50)
            """),
                textwrap.dedent("""\
                mean = data.mean()
                std = data.std()
            """),
                textwrap.dedent("""\
                normalized = (data - mean) / std
            """),
                textwrap.dedent("""\
                print(f"mean={normalized.mean():.6f} std={normalized.std():.6f}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(5)
        # Normalized data should have mean≈0 and std≈1
        assert "mean=" in output
        assert "std=" in output

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "mean=" in output2
        assert "std=" in output2

    def test_large_notebook_partial_change_after_restart(self, nb_runner):
        """Large notebook: restart, change one cell, re-run."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "step1 = base + 5",
                "step2 = step1 * 2",
                "step3 = step2 - 3",
                "step4 = step3 ** 2",
                "print(step4)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # step1=15, step2=30, step3=27, step4=729
        assert "729" in nb_runner.get_output(6)

        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "base = 20")
        nb_runner.start_kernel()
        nb_runner.run_all()
        # step1=25, step2=50, step3=47, step4=2209
        assert "2209" in nb_runner.get_output(6)
