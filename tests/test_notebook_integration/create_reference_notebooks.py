"""Script to create reference notebooks for integration tests."""
import nbformat
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "reference_notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def create_financial_demo():
    """Create the financial_demo.ipynb notebook."""
    nb = nbformat.v4.new_notebook()

    # Cell 1: Create test DataFrame
    cell1 = nbformat.v4.new_code_cell('''# Cell 1: Create test DataFrame
import pandas as pd
df = pd.DataFrame({
    'Ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL', 'TSLA', 'TSLA', 'AMZN', 'AMZN'],
    'Close': [100.0, 101.0, 200.0, 201.0, 300.0, 301.0, 400.0, 401.0, 500.0, 501.0],
    'Volume': [1000, 1100, 2000, 2100, 3000, 3100, 4000, 4100, 5000, 5100]
})''')

    # Cell 2: Process 4 tickers
    cell2 = nbformat.v4.new_code_cell('''# Cell 2: Process 4 tickers
ticker_stats = {}
print(df)

for ticker in ['AAPL', 'MSFT', 'GOOGL', 'TSLA']:
    ticker_data = df[df['Ticker'] == ticker]
    stats = {
        'mean_close': ticker_data['Close'].mean(),
        'std_close': ticker_data['Close'].std(),
    }
    ticker_stats[ticker] = stats
    print(f"{ticker}: mean={stats['mean_close']:.2f}")

print(ticker_stats.keys())
print('Done processing!')''')

    # Cell 3: Display keys
    cell3 = nbformat.v4.new_code_cell('''# Cell 3: Display keys
print(f'Keys cell: {list(ticker_stats.keys())}')''')

    nb.cells = [cell1, cell2, cell3]

    with open(NOTEBOOKS_DIR / 'financial_demo.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print('Created financial_demo.ipynb')


def create_loop_accumulator():
    """Create the loop_accumulator.ipynb notebook."""
    nb = nbformat.v4.new_notebook()

    # Cell 1: Initialize data
    cell1 = nbformat.v4.new_code_cell('''# Cell 1: Initialize data
data = {}
for x in [1, 2, 3]:
    data[x] = x * 10
print(f"After loop: {data}")''')

    # Cell 2: Display data (downstream)
    cell2 = nbformat.v4.new_code_cell('''# Cell 2: Display data (downstream)
print(f"Downstream: {data}")''')

    nb.cells = [cell1, cell2]

    with open(NOTEBOOKS_DIR / 'loop_accumulator.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print('Created loop_accumulator.ipynb')


if __name__ == '__main__':
    create_financial_demo()
    create_loop_accumulator()
    print("All reference notebooks created!")
