# Installation

## Requirements

- Python 3.10 or higher
- IPython/Jupyter (for notebook caching)

## Basic Installation

```bash
pip install cash-lib
```

## Optional Dependencies

Cash supports multiple storage backends. Install the ones you need:

```bash
# Redis backend (for shared/distributed caching)
pip install cash-lib[redis]

# S3 backend (for cloud storage)
pip install cash-lib[s3]

# All optional dependencies
pip install cash-lib[all]
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
