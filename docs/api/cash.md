# Cash class

For both paths, though mostly the decorator: the `Cash` class, the helpers
every cached function gets, and the module-level functions. The
[decorator guide](../decorator.md) explains when to use each option.

```python
from cash import Cash, CacheExplanation, configure, disabled, reset_session, cleanup
```

`cash.cache`, `cash.register_hasher` and `cash.show_stats` are those methods
of a default `Cash` built on first use; `cash.help()` prints the guide for
coding agents. Create your own `Cash(...)` when you need different settings
for some functions.

## Signatures at a glance

<!-- claim: cash/core.py:Cash.__init__ @9c872b72, cash/core.py:Cash.cache @d8475d94 -->
```text
Cash(backend=None, cache_dir=None, backends=None, compress=None,
     register_magic=None, debug=None, use_locking=False, config_path=None,
     verbose=None, **config_overrides)

Cash.cache(func=None, *, depends_on=None, dynamic_depends_on=None,
           file_depends_on=None, ttl=None, cache_if=None,
           chunk_max_items=1_000_000, chunk_max_bytes=1_000_000_000,
           strict=False, assume_safe=False, allow_random=False, frozen=False)
```

`**config_overrides` takes any [setting](../getting-started/configuration.md)
by name and wins over every config file and environment variable.

::: cash.Cash
    options:
      members:
        - __init__
        - cache
        - reconfigure
        - register_hasher
        - register_file_handler
        - cleanup
        - clear_all
        - explorer
        - show_stats
        - shutdown
        - backend

## Cached function

A function decorated with `@cash.cache` (or `@c.cache`) is called as before
and also has these attributes:

| Attribute | What it does |
|---|---|
| `f.explain(*args, **kwargs)` | Returns a [`CacheExplanation`](cash.md#cash.CacheExplanation): whether a call with these arguments would hit, and why. Runs nothing and changes nothing. |
| `f.cache_info()` | Returns a dict: `hits`, `misses`, `hit_rate`, `total_time_saved` (seconds), `miss_reasons` (count per reason, such as `"new arguments"`) and `warnings` (the last 20, each with `category`, `code`, `message`, `timestamp`). |
| `f.cache_clear()` | Deletes this function's entries and resets its statistics and warning log, so its warnings are shown again. |
| `f.__wrapped__` | The undecorated function; calling it bypasses the cache. |

```python
import cash

@cash.cache
def double(x):
    return x * 2

double(1)
double(1)
print(double.cache_info()["hits"])  # 1
print(double.explain(1).reason)     # "hit"
```

::: cash.CacheExplanation
    options:
      members:
        - would_hit
        - reason
        - func_name
        - cache_key
        - entry_id
        - cache_dir
        - details

## Module-level functions

These act on the default `Cash` that `@cash.cache` uses.

::: cash.configure

::: cash.disabled

::: cash.reset_session

::: cash.cleanup
