"""
Test the exact scenario: ticker_stats = {} followed by loop, add new item

This mimics the financial_analysis_demo.ipynb structure exactly.
"""
import pytest
from tests.test_notebook_integration.conftest import REFERENCE_NOTEBOOKS_DIR

pytestmark = [pytest.mark.core, pytest.mark.loops]


def test_financial_demo_exact_structure(nb_runner):
    """
    Exact structure of financial_analysis_demo.ipynb:
    - Cell 1: Create mock DataFrame
    - Cell 2: Loop cell with ticker_stats = {}, loop, prints
    - Cell 3: ticker_stats.keys()
    
    Test: Add new ticker, run cell 3 only
    """
    df_setup = """import pandas as pd
df = pd.DataFrame({
    'Ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL', 'TSLA', 'TSLA', 'AMZN', 'AMZN'],
    'Close': [100.0, 101.0, 200.0, 201.0, 300.0, 301.0, 400.0, 401.0, 500.0, 501.0],
    'Volume': [1000, 1100, 2000, 2100, 3000, 3100, 4000, 4100, 5000, 5100]
})"""
    
    loop_cell_4_tickers = """ticker_stats = {}
print(df)

for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA"]:
    ticker_data = df[df["Ticker"] == ticker]
    stats = {
        "mean_close": ticker_data["Close"].mean(),
        "std_close": ticker_data["Close"].std(),
    }
    ticker_stats[ticker] = stats
    print(f"{ticker}: mean={stats['mean_close']:.2f}")

print(ticker_stats.keys())
print("Done processing!")"""

    keys_cell = """print(f"Keys cell: {list(ticker_stats.keys())}")"""
    
    nb_runner.create_notebook([
        df_setup,
        loop_cell_4_tickers,
        keys_cell
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    out1 = nb_runner.get_output(3)
    print(f"Run 1 keys cell: {out1}")
    assert "AAPL" in out1
    assert "TSLA" in out1
    
    # Modify loop to add AMZN
    loop_cell_5_tickers = """ticker_stats = {}
print(df)

for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]:
    ticker_data = df[df["Ticker"] == ticker]
    stats = {
        "mean_close": ticker_data["Close"].mean(),
        "std_close": ticker_data["Close"].std(),
    }
    ticker_stats[ticker] = stats
    print(f"{ticker}: mean={stats['mean_close']:.2f}")

print(ticker_stats.keys())
print("Done processing!")"""
    
    nb_runner.set_cell_source(2, loop_cell_5_tickers)
    nb_runner.run_all()
    
    out2 = nb_runner.get_output(3)
    print(f"Run 2 keys cell: {out2}")
    
    # All 5 should be present
    for ticker in ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]:
        assert ticker in out2, f"Missing {ticker} in: {out2}"


def test_init_before_loop_no_overwrite(nb_runner):
    """
    Test that ticker_stats = {} initialization doesn't overwrite restored data
    when a new iteration is added.
    """
    nb_runner.create_notebook([
        """data = {}
for x in [1, 2, 3]:
    data[x] = x * 10
print(f"After loop: {data}")""",
        """print(f"Downstream: {data}")"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    out1 = nb_runner.get_output(2)
    print(f"Run 1: {out1}")
    assert "3: 30" in out1
    
    # Add 4th item
    nb_runner.set_cell_source(1, """data = {}
for x in [1, 2, 3, 4]:
    data[x] = x * 10
print(f"After loop: {data}")""")
    nb_runner.run_all()
    
    out2 = nb_runner.get_output(2)
    print(f"Run 2: {out2}")
    
    # Verify ALL items present
    for x in [1, 2, 3, 4]:
        assert f"{x}: {x*10}" in out2, f"Missing {x} in: {out2}"


def test_same_session_edit_and_rerun(nb_runner):
    """
    Test the exact user scenario:
    1. Run notebook with 4 items (all cells)
    2. In SAME session, edit loop to have 5 items
    3. Run all cells again
    """
    nb_runner.create_notebook([
        """data = {}
for x in ['A', 'B', 'C', 'D']:
    data[x] = x * 2
print(f"First: {data}")""",
        """print(f"Keys: {list(data.keys())}")"""
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    out1 = nb_runner.get_output(2)
    print(f"Run 1: {out1}")
    assert "'D'" in out1
    
    # Edit and re-run in same session
    nb_runner.set_cell_source(1, """data = {}
for x in ['A', 'B', 'C', 'D', 'E']:
    data[x] = x * 2
print(f"Second: {data}")""")
    nb_runner.run_all()
    
    out2 = nb_runner.get_output(2)
    print(f"Run 2: {out2}")
    
    for x in ['A', 'B', 'C', 'D', 'E']:
        assert f"'{x}'" in out2, f"Missing {x} in: {out2}"


def test_financial_demo_load_reference(nb_runner):
    """
    Test loading the actual financial_demo reference notebook.
    """
    nb_runner.load(REFERENCE_NOTEBOOKS_DIR / "financial_demo.ipynb")
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    # Check output from cell 3 (the keys display cell)
    output = nb_runner.get_output(3)
    print(f"Cell 3 output: {output}")
    
    # Should have 4 tickers
    assert "AAPL" in output
    assert "TSLA" in output


def test_financial_demo_modify_and_rerun(nb_runner):
    """
    Test modifying the financial demo notebook and re-running.
    """
    nb_runner.load(REFERENCE_NOTEBOOKS_DIR / "financial_demo.ipynb")
    nb_runner.start_kernel()
    nb_runner.run_all()
    
    output1 = nb_runner.get_output(3)
    assert "AMZN" not in output1  # AMZN not in original
    
    # Modify cell 2 to add AMZN
    new_cell2 = '''ticker_stats = {}
print(df)

for ticker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']:
    ticker_data = df[df['Ticker'] == ticker]
    stats = {
        'mean_close': ticker_data['Close'].mean(),
        'std_close': ticker_data['Close'].std(),
    }
    ticker_stats[ticker] = stats
    print(f"{ticker}: mean={stats['mean_close']:.2f}")

print(ticker_stats.keys())
print('Done processing!')'''
    
    nb_runner.set_cell_source(2, new_cell2)
    nb_runner.run_cells([2, 3])
    
    output2 = nb_runner.get_output(3)
    assert "AMZN" in output2, f"Expected AMZN in output: {output2}"
