# Cached Function

```python
import cash

@cash.cache
def double(x):
    return x * 2

result1 = double(5)   # First call: cache miss
result2 = double(5)   # Second call: cache hit
```
