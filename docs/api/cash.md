# Cash class

The main entry point for decorator-style caching. See the
[decorator guide](../decorator.md) for a narrative walkthrough; this
page is the exhaustive parameter and method reference.

## Imports

```python
from cash import Cash, CacheExplanation, configure, reset_session
```

`cash.cache`, `cash.show_stats`, and `cash.register_hasher` are
also reachable directly on the top-level module — they proxy to a
lazily-created `Cash()` singleton (see *Module-level convenience*
below).

::: cash.Cash
    options:
      members:
        - __init__
        - cache
        - register_hasher
        - register_file_handler
        - cleanup
        - explorer
        - show_stats
        - register_magic
        - shutdown
        - backend
        - drain_decorator_calls

---

When you don't need a custom `Cash` configuration, the top-level
`cash` module proxies to a lazily-created singleton:

```python
import cash

@cash.cache
def expensive(x):
    return x ** 2

cash.show_stats()
cash.register_hasher(MyType, my_hasher)
```

Equivalent to calling these methods on a singleton `Cash()`. For custom
configuration, instantiate `Cash(...)` explicitly. The two
module-level functions below cover runtime reconfiguration and session
reset.

::: cash.configure

::: cash.reset_session

---

Returned by `f.explain(*args, **kwargs)` on any `@cash.cache`-wrapped
function. Inspecting an explanation never calls the function, mutates
stats, or writes to the backend.

::: cash.CacheExplanation
    options:
      members:
        - would_hit
        - reason
        - func_name
        - cache_key
        - details
