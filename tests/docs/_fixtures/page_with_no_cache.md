# Stateful Function

```python
import cash

@cash.stateful
def get_time():
    import time
    return time.time()

# These should not be cached
t1 = get_time()
t2 = get_time()
```
