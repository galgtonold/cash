# Purity & annotations

<!-- claim: cash/notebook/purity.py:pure @de701258, cash/notebook/purity.py:stateful @d2b97ef0, cash/notebook/purity.py:analyze_function_purity @7323a225 -->
The decorators and helpers that control what Cash considers safe to
cache. For a walkthrough — when to use each, the auto-detection
heuristic, common footguns — see the [Purity tutorial](../tutorials/feature-guides/purity-decorators.md).

## Imports

```python
from cash import (
    pure, stateful,           # decorators
    mark_pure, mark_stateful, # in-place markers for third-party callables
    is_pure, is_stateful,     # introspection
    analyze_function_purity,  # AST-based heuristic (returns bool)
)

# For richer programmatic analysis:
from cash.purity_analyzer import (
    PurityAnalyzer, PurityReport, PurityIssue,
    get_analyzer,             # process-wide singleton
    ISSUE_IMPURE_CALL, ISSUE_DYNAMIC_PATTERN,
    ISSUE_DISCARDED_CALL, ISSUE_SCOPE_MUTATION,
)
```

## Decorators

The two below are decorators — apply with `@pure` / `@stateful` above
your function. They wrap the function and set `_cash_pure` /
`_cash_stateful` on the returned object so the analyzer trusts your
declaration.

::: cash.pure

::: cash.stateful

---

## Module-level markers

For library callables you can't decorate at the source (C extensions,
classes you can't subclass) — annotate them in your code where you
import them. **Plain function calls, NOT decorators** — they set the
marker in place without wrapping the callable.

```python
import cash, pandas as pd

cash.mark_pure(pd.DataFrame.merge)        # tell the analyzer this is fine
cash.mark_stateful(pd.DataFrame.to_sql)   # tell it this writes
```

::: cash.mark_pure

::: cash.mark_stateful

---

## Introspection

The three helpers below let you query the marker state or run the
auto-detection heuristic directly.

::: cash.is_pure

::: cash.is_stateful

::: cash.analyze_function_purity

---

## Programmatic analysis

The richer analyzer behind `@cash.cache`'s purity warnings. Use it
when `analyze_function_purity` (which returns a bool) isn't enough
— e.g. to surface specific issues in a custom lint tool, to drive a
pre-commit hook, or to walk a function's helper hierarchy yourself.

```python
from cash.purity_analyzer import get_analyzer

analyzer = get_analyzer()
report = analyzer.analyze(my_function)
print(report.format())
# Lists each detected issue (line + kind + description) grouped by
# the function where it appears. Empty when the function is clean.
```

::: cash.purity_analyzer.PurityAnalyzer
    options:
      members:
        - analyze

::: cash.purity_analyzer.get_analyzer

::: cash.purity_analyzer.PurityReport
    options:
      members:
        - issues
        - helper_source_hashes
        - helper_resolution_paths
        - opaque_callees
        - is_clean
        - format

::: cash.purity_analyzer.PurityIssue
    options:
      members:
        - kind
        - description
        - where
        - line

### Issue kinds

The six `kind` values you'll see in `PurityIssue.kind` are
exported as module-level constants. Five of them **warn**;
`ISSUE_UNTRACKABLE_DEP` is the one that **raises** in default mode:

| Constant | Value | Meaning |
|---|---|---|
| `ISSUE_IMPURE_CALL` | `"impure_call"` | Known I/O / side-effecting call (`requests.post`, `os.system`, `to_csv`, pandas `inplace=True`, `print`, `logging.*`, etc.) |
| `ISSUE_DYNAMIC_PATTERN` | `"dynamic_pattern"` | Code chosen at runtime out of a table that cannot reach the cache key: one built inside the body (`t = {...}; t[key]()`), one on a parameter (`router.table[key]()`), or a runtime namespace (`globals()[name]()`, `vars(mod)[name]()`). A **module-level** table (`HANDLERS[key]()`) is not flagged — cash hashes it as a global — and neither is a callable passed as an argument |
| `ISSUE_UNTRACKABLE_DEP` | `"untrackable_dep"` | Explicit dynamism cash refuses to cache silently — `eval`/`exec`/`compile`, `getattr(obj, name)()` with a non-constant name, `getattr(mod, "exec")(...)`, `importlib.import_module`. **Raises `CashImpureFunctionError` on the first call even in default mode**; `assume_safe=True` suppresses it |
| `ISSUE_DISCARDED_CALL` | `"discarded_call"` | Bare-statement call whose return value is thrown away, with a callee not in `KNOWN_PURE_BUILTINS` |
| `ISSUE_SCOPE_MUTATION` | `"scope_mutation"` | `global`/`nonlocal` declaration, attribute assignment, subscript assignment, augmented-assign to same |
| `ISSUE_MUTABLE_GLOBAL` | `"mutable_global"` | Reads a module-level global that is reassigned or mutated somewhere in its module, so the cached result won't reflect changes to it |

```python
from cash.purity_analyzer import (
    ISSUE_IMPURE_CALL,
    ISSUE_DYNAMIC_PATTERN,
    ISSUE_UNTRACKABLE_DEP,
    ISSUE_DISCARDED_CALL,
    ISSUE_SCOPE_MUTATION,
    ISSUE_MUTABLE_GLOBAL,
)

if any(i.kind == ISSUE_IMPURE_CALL for i in report.issues):
    print("function does I/O — definitely not pure")
```
