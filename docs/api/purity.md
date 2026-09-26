---
search:
  boost: 0.5
---

# Purity markers

For both paths, mainly the decorator: the markers that tell cash what to
trust about a function or class. The
[purity guide](../tutorials/feature-guides/purity-decorators.md) shows when
to use each; line-level `# @cash:` comments are on
[Annotations](../annotations.md).

```python
from cash import (
    assume_safe,
    is_pure,
    is_stateful,
    opaque,
    pure,
    stateful,
)
```

| Marker | With `@cash.cache` | In a notebook |
|---|---|---|
| `pure` | A cached function that calls it is not warned about it. | A statement that calls a file-writing helper caches only if the helper is marked. |
| `stateful` | Reported as a side effect; refused under `strict=True`. | A statement that calls it is never cached. |
| `opaque` | The class's code is left out of the key of any call that receives the class or an instance. | No effect on statements. |

To mark a function you do not own, call the marker on it where you import
it. The mark is set on the function itself, so you can ignore the return
value:

```python
import cash

cash.stateful(pd.DataFrame.to_sql)  # pd is pandas
```

<!-- claim: cash/purity.py:pure @26a3db86, cash/purity.py:stateful @afae9270 -->
::: cash.pure

::: cash.stateful

::: cash.opaque

::: cash.is_pure

::: cash.is_stateful

## Waiving findings for a block

`assume_safe` is not a marker: it is a context manager for the lines of a
`@cash.cache` function. It sits between `# @cash:assume-safe` on one line and
`@cash.cache(assume_safe=True)` for the whole function;
[Side effects](../decorator.md#side-effects) shows when to use which. In a
notebook, keep to the comment (see
[Annotations](../annotations.md#cashassume-safe)).

<!-- claim: cash/effect_observer.py:assume_safe @aa4ceeea -->
::: cash.assume_safe
