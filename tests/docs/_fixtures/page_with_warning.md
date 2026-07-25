# Page with a runtime cash warning

This fence emits a `CashCacheIneffectiveWarning` at runtime (an unpicklable
argument means no cache key can be built). It carries no `test:expect-warning`
annotation, so the harness should fail the page.

```python
import cash
import threading

c = cash.Cash(register_magic=False)

@c.cache
def f(lock):
    return id(lock)

f(threading.Lock())
```
