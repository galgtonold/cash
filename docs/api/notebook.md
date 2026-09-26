---
search:
  boost: 0.5
---

# Notebook integration

For the notebook path: the one public class for tooling around notebooks.
The magics are on [Magic commands](../magics.md).

```python
from cash.nbconvert import CashStripPreprocessor
```

`CashStripPreprocessor` removes cash's cell badges and debug lines from a
notebook's outputs, and optionally the `%cash_*` lines from its cells, so
you can share or commit it clean.

```bash
jupyter nbconvert --to html \
  --Exporter.preprocessors='["cash.nbconvert.CashStripPreprocessor"]' \
  notebook.ipynb
```

<!-- test:skip reason="reads a notebook file that does not exist in the test environment" -->
```python
import nbformat
from cash.nbconvert import CashStripPreprocessor

nb = nbformat.read("notebook.ipynb", as_version=4)
nb, _ = CashStripPreprocessor(strip_magics=True).preprocess(nb, {})
nbformat.write(nb, "clean_notebook.ipynb")
```

::: cash.nbconvert.CashStripPreprocessor
    options:
      members:
        - strip_badges
        - strip_debug
        - strip_magics
