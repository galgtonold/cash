# Notebook integration

The Python-side API for working with cash inside notebooks. For the
user-facing magics (`%cash_on`, `%cash_stats`, etc.) see
[Magic Commands](../magics.md); for the architectural deep-dive see
[Notebook caching](../notebook_caching_api.md) and
[How Cash Works](../how-it-works/overview.md).

This page covers programmatic entry points — useful when you're
writing tooling around cash (custom dashboards, CI checks, notebook
exporters, lint integrations) rather than driving the magics
directly.

## Imports

```python
# nbconvert integration:
from cash.nbconvert import CashStripPreprocessor

# Notebook subsystem public types:
from cash.notebook import (
    CacheStatus,           # cell-status enum (COMPUTED / RESTORED / SKIPPED / …)
    ExecutionResult,       # statement-execution result wrapper
    CodeAnalyzer,          # source hashing + called-function detection
    CashMagics,            # the magics class (registered automatically)
)
```

---

## CashStripPreprocessor

Strip cash-specific artifacts from a notebook before sharing or
committing — removes the cell-status badges, the debug output
prefixes, and (optionally) the `%cash_*` magic invocations
themselves.

### CLI usage

```bash
jupyter nbconvert --to html \
    --Exporter.preprocessors='["cash.nbconvert.CashStripPreprocessor"]' \
    notebook.ipynb
```

### Programmatic usage

<!-- test:skip reason="reads notebook.ipynb which doesn't exist in test env" -->
```python
from cash.nbconvert import CashStripPreprocessor
import nbformat

nb = nbformat.read("notebook.ipynb", as_version=4)

preprocessor = CashStripPreprocessor(
    strip_badges=True,    # remove the cell-status badge HTML outputs
    strip_debug=True,     # remove [UPSTREAM_DEBUG] / [LINEAGE_DEBUG] / … lines
    strip_magics=False,   # set True to also strip %cash_on / %cash_off / … from cell source
)
nb, _ = preprocessor.preprocess(nb, {})

nbformat.write(nb, "clean_notebook.ipynb")
```

::: cash.nbconvert.CashStripPreprocessor
    options:
      members:
        - strip_badges
        - strip_debug
        - strip_magics
        - preprocess
        - preprocess_cell

---

## CacheStatus

Status of a processed statement. Returned in execution-result
metadata and surfaced in the cell badge.

::: cash.notebook.CacheStatus
    options:
      members: true

---

## ExecutionResult

Lightweight container for the outcome of executing a single
statement. Exposed in places that drive the statement processor
programmatically.

::: cash.notebook.ExecutionResult
    options:
      members:
        - __init__

---

## CodeAnalyzer

Static helpers for source hashing and called-function detection.
`Cash` uses these internally for the dependency graph; you may want
them directly when building lint tools that need a stable hash of a
callable's body.

::: cash.notebook.analysis.CodeAnalyzer

---

## CashMagics

The IPython magic class. Registered automatically when `cash` is
imported inside an IPython session (or via `%load_ext cash` / the
`cash autoload on` CLI hook). You normally interact with it via the
magic commands themselves — see [Magic Commands](../magics.md).

::: cash.notebook.ipython.magics.CashMagics
    options:
      members: false
