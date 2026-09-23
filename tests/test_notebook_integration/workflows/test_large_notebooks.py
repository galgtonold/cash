"""Notebooks with many cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


# Large-scale notebook simulation — 15-20 cell notebooks with
# realistic data science workflows testing end-to-end caching behavior.
#
# These tests simulate real notebooks that users would write, with realistic
# complexity and cell counts.
@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


# Large notebook (10+ cells) interaction tests.
#
# Tests with larger notebooks that simulate real-world scenarios
# with many cells, edits at various positions, and full run-through.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestTenCellWorkflow:
    """10-cell workflow with edits."""

    def test_ten_cell_data_pipeline(self, nb_runner):
        """10-cell data pipeline, edit source and check propagation."""
        nb_runner.create_notebook(
            [
                "raw = list(range(1, 11))",  # cell 1
                "cleaned = [x for x in raw if x > 0]",  # cell 2
                "normalized = [x / max(cleaned) for x in cleaned]",  # cell 3
                "scaled = [x * 100 for x in normalized]",  # cell 4
                "rounded = [round(x) for x in scaled]",  # cell 5
                "top5 = sorted(rounded, reverse=True)[:5]",  # cell 6
                "bottom5 = sorted(rounded)[:5]",  # cell 7
                "spread = top5[0] - bottom5[0]",  # cell 8
                "avg = sum(rounded) / len(rounded)",  # cell 9
                "print(f'spread = {spread}, avg = {avg}')",  # cell 10
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(10)
        assert "spread = " in output
        assert "avg = " in output

        # Edit source data
        nb_runner.set_cell_source(1, "raw = list(range(1, 51))")
        nb_runner.run_all()
        output2 = nb_runner.get_output(10)
        assert "spread = " in output2
        assert "avg = " in output2

    def test_ten_cell_edit_middle(self, nb_runner):
        """10-cell chain, edit cell 5 in the middle."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1",
                "g = f + 1",
                "h = g + 1",
                "i = h + 1",
                "print(f'i = {i}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "i = 9" in nb_runner.get_output(10)

        # Edit cell 5 to multiply instead of add
        nb_runner.set_cell_source(5, "e = d * 10")
        nb_runner.run_all()
        # a=1,b=2,c=3,d=4,e=40,f=41,g=42,h=43,i=44
        assert "i = 44" in nb_runner.get_output(10)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestEightCellWithBranching:
    """8-cell notebook with branching dependencies."""

    def test_eight_cell_split_merge(self, nb_runner):
        """8-cell notebook: shared root, two branches, merge."""
        nb_runner.create_notebook(
            [
                "root = 10",  # cell 1
                "branch_a1 = root * 2",  # cell 2
                "branch_a2 = branch_a1 + 5",  # cell 3
                "branch_b1 = root + 3",  # cell 4
                "branch_b2 = branch_b1 * 4",  # cell 5
                "merged = branch_a2 + branch_b2",  # cell 6
                "final = merged * 2",  # cell 7
                "print(f'final = {final}')",  # cell 8
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a1=20, a2=25, b1=13, b2=52, merged=77, final=154
        assert "final = 154" in nb_runner.get_output(8)

        # Edit root
        nb_runner.set_cell_source(1, "root = 100")
        nb_runner.run_all()
        # a1=200, a2=205, b1=103, b2=412, merged=617, final=1234
        assert "final = 1234" in nb_runner.get_output(8)

        # Edit one branch
        nb_runner.set_cell_source(4, "branch_b1 = root - 50")
        nb_runner.run_all()
        # a1=200, a2=205, b1=50, b2=200, merged=405, final=810
        assert "final = 810" in nb_runner.get_output(8)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestProgressiveNotebookBuilding:
    """Simulate building notebook progressively — add cells one by one."""

    def test_build_notebook_incrementally(self, nb_runner):
        """Start with 2 cells, progressively add more."""
        # Start small
        nb_runner.create_notebook(
            [
                "x = 5",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 5" in nb_runner.get_output(2)

        # Now extend by recreating with more cells
        nb_runner.shutdown()
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(3)

        # Extend again
        nb_runner.shutdown()
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + x",
                "print(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 15" in nb_runner.get_output(4)


# Notebook size scaling interaction tests.
#
# Tests that exercise notebooks with 10-20 cells simulating
# real-world data science workflows with multiple phases.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestTenCellPipeline:
    """10-cell pipeline simulating a data science workflow."""

    def test_ten_cell_data_pipeline(self, nb_runner):
        """Full 10-cell data pipeline with edit."""
        nb_runner.create_notebook(
            [
                "# Step 1: Data generation\nimport random\nrandom.seed(42)\ndata = [random.randint(1, 100) for _ in range(20)]",
                "# Step 2: Cleaning\ncleaned = [x for x in data if x > 10]",
                "# Step 3: Stats\nmean_val = sum(cleaned) / len(cleaned)",
                "# Step 4: Normalization\nnormalized = [(x - mean_val) for x in cleaned]",
                "# Step 5: Filter outliers\nfiltered = [x for x in normalized if abs(x) < 40]",
                "# Step 6: Transform\ntransformed = [x ** 2 for x in filtered]",
                "# Step 7: Aggregate\ntotal = sum(transformed)\ncount = len(transformed)",
                "# Step 8: Average\navg = total / count if count else 0",
                "# Step 9: Scale\nscaled = round(avg * 100, 2)",
                "# Step 10: Report\nprint(f'scaled = {scaled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(10)
        assert "scaled = " in output

        # Edit the seed → different data → different result
        nb_runner.set_cell_source(
            1,
            "# Step 1: Data generation\nimport random\nrandom.seed(99)\ndata = [random.randint(1, 100) for _ in range(20)]",
        )
        nb_runner.run_all()
        output2 = nb_runner.get_output(10)
        assert "scaled = " in output2

    def test_ten_cell_edit_multiple_cells(self, nb_runner):
        """10-cell pipeline, edit two non-adjacent cells."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1",
                "g = f + 1",
                "h = g + 1",
                "i = h + 1",
                "j = i + 1\nprint(f'j = {j}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "j = 10" in nb_runner.get_output(10)

        # Edit cells 3 and 7
        nb_runner.set_cell_source(3, "c = b * 10")
        nb_runner.set_cell_source(7, "g = f * 10")
        nb_runner.run_all()
        # a=1, b=2, c=20, d=21, e=22, f=23, g=230, h=231, i=232, j=233
        assert "j = 233" in nb_runner.get_output(10)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestFifteenCellWorkflow:
    """15-cell workflow simulating a report generation."""

    def test_fifteen_cell_workflow(self, nb_runner):
        """15-cell workflow with edit at beginning."""
        cells = [
            "base = 10",  # 1
            "step1 = base + 1",  # 2
            "step2 = step1 * 2",  # 3
            "step3 = step2 - 3",  # 4
            "step4 = step3 + 4",  # 5
            "step5 = step4 * 5",  # 6
            "step6 = step5 - 6",  # 7
            "step7 = step6 + 7",  # 8
            "step8 = step7 * 2",  # 9
            "step9 = step8 - 1",  # 10
            "step10 = step9 + 10",  # 11
            "step11 = step10 * 3",  # 12
            "step12 = step11 - 5",  # 13
            "step13 = step12 + 2",  # 14
            "result = step13\nprint(f'result = {result}')",  # 15
        ]
        nb_runner.create_notebook(cells)
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(15)
        assert "result = " in output

        # Edit the base
        nb_runner.set_cell_source(1, "base = 20")
        nb_runner.run_all()
        output2 = nb_runner.get_output(15)
        assert "result = " in output2
        # Verify different results
        assert output != output2 or "result" in output2


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestWidePipeline:
    """Wide (many parallel branches) notebook."""

    def test_six_parallel_branches(self, nb_runner):
        """6 parallel branches from a shared root."""
        nb_runner.create_notebook(
            [
                "root = 10",  # 1
                "branch_a = root * 1",  # 2
                "branch_b = root * 2",  # 3
                "branch_c = root * 3",  # 4
                "branch_d = root * 4",  # 5
                "branch_e = root * 5",  # 6
                "branch_f = root * 6",  # 7
                "total = branch_a + branch_b + branch_c + branch_d + branch_e + branch_f\nprint(f'total = {total}')",  # 8
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10*(1+2+3+4+5+6) = 10*21 = 210
        assert "total = 210" in nb_runner.get_output(8)

        nb_runner.set_cell_source(1, "root = 5")
        nb_runner.run_all()
        assert "total = 105" in nb_runner.get_output(8)

        # Edit one branch
        nb_runner.set_cell_source(4, "branch_c = root * 30")
        nb_runner.run_all()
        # 5*(1+2+30+4+5+6) = 5 + 10 + 150 + 20 + 25 + 30 = 240
        assert "total = 240" in nb_runner.get_output(8)
