# Page with an expected cash warning

The same warning as `page_with_warning.md`, but the fence opts in with
`test:expect-warning`, so the harness should let it pass.

<!-- test:expect-warning -->
```python
import cash
import threading

c = cash.Cash(register_magic=False)

@c.cache
def f(lock):
    return id(lock)

f(threading.Lock())
```
