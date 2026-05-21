# Installation

## Requirements

- Python 3.10 or higher
- IPython/Jupyter (for notebook caching)

## Basic Installation

```bash
pip install cash-lib
```

## Optional Dependencies

Cash ships with no required runtime dependencies — every integration is opt-in. Install only the extras you need:

```bash
pip install "cash-lib[pandas]"        # pandas + pyarrow (stable DataFrame hashing)
pip install "cash-lib[polars]"        # polars support
pip install "cash-lib[notebook]"      # IPython integration for %cash_on and friends
pip install "cash-lib[memory]"        # psutil for memory-aware cache budgets
pip install "cash-lib[redis]"         # Redis backend (shared/distributed caching)
pip install "cash-lib[s3]"            # S3 backend (cloud storage, via boto3)
pip install "cash-lib[sqlite]"        # SQLite backend (uses stdlib sqlite3; no extra deps)
pip install "cash-lib[cloudpickle]"   # cloudpickle for serializing closures and lambdas
pip install "cash-lib[ipynbname]"     # notebook-name detection helper
pip install "cash-lib[viz]"           # matplotlib + ipywidgets for visualization helpers
pip install "cash-lib[all]"           # everything above
```

## Development Installation

For contributing to Cash:

```bash
git clone https://github.com/galgtonold/cash.git
cd cash
pip install -e ".[dev]"
```

## Verify Installation

```python
import cash
print(cash.__version__)
```

In a Jupyter notebook:

```python
import cash
%cash_on
print("Cash is working!")
```
