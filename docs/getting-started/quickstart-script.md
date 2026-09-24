# Quick start: decorator

!!! info "Applies to: decorator"
    Scripts, services and libraries that use `@cash.cache`.

## 1. Install

```bash
pip install cash-lib
```

## 2. Decorate a slow function

<!-- test:skip reason="illustrative: a 5-second sleep stands in for real work" -->
```python
import time
import cash

@cash.cache
def expensive_function(x, y):
    time.sleep(5)          # stand-in for real work
    return x + y

expensive_function(1, 2)   # first call: runs, about 5 seconds
expensive_function(1, 2)   # second call: cache hit, instant
```

<!-- claim: cash/backends/persistence_policy.py:PersistencePolicy.decide @dfaf7643 -->
cash keys each call on its arguments and on the function's source. Every result
is written to disk, in a `.cash/` folder in your project, so the next run of the
script gets a hit on the first call too.

<!-- claim: cash/decorator/arg_hashing.py:ArgHashingMixin._hash_arg_payload @7bc7e4ca, cash/object_hashing.py:builtin_hash @bd4210c7 -->
Arguments do not need to be hashable: DataFrames and arrays are hashed by
content, so an equal copy hits the same entry.

## 3. Change what it depends on

### Edit a helper: the cache notices

cash also keys on the source of the functions your function calls, however
deep.
`pipeline` below never mentions `clean`, but editing `clean` still
invalidates it:

```python
import cash

def clean(text):
    return text.strip().lower()

def features(text):
    return len(set(clean(text).split()))

@cash.cache
def pipeline(text):
    return features(text)

sample = "  The cat  the HAT  "
print(pipeline(sample))          # 3  -> {'the', 'cat', 'hat'}
print(pipeline(sample))          # 3  -> cache hit; the body never ran
print(pipeline.cache_info())     # {'hits': 1, 'misses': 1, ...}
```

Now edit `clean` so it stops lower-casing:

<!-- test:skip reason="the docs harness infers expected hits/misses from call sites, which cannot model an invalidation between two identical calls; this fence is executed and its output asserted by tests/docs/test_quickstart_helper_invalidation.py instead" -->
```python
def clean(text):
    return text.strip()

print(pipeline(sample))          # 4  -> {'The', 'cat', 'the', 'HAT'}
print(pipeline.cache_info())     # {'hits': 1, 'misses': 2, ...}
```

The answer changed and `misses` went up.

<!-- claim: cash/decorator/code_identity.py:CodeIdentityMixin._is_user_module @998fb383 -->
This also works for helpers imported from other modules of your project. cash
stops at installed code (`site-packages` and the standard library); if a
third-party function matters to a result, name it with `depends_on=`.

### Change a file it reads

<!-- test:skip reason="illustrative: data.csv is not shipped with the docs" -->
```python
import cash
import pandas as pd

@cash.cache
def load():
    return pd.read_csv("data.csv")   # change data.csv and the next call recomputes
```

cash tracks the files a decorated function reads, with nothing to declare.

## What's next

- [`@cash.cache` guide](../decorator.md): every parameter, and what invalidates
  a result.
- [Testing your code](../tutorials/feature-guides/testing-your-code.md): keep
  your test suite's cache apart from your application's.
- [File dependencies](../tutorials/feature-guides/custom-file-sources.md):
  remote files and files cash cannot see being read.
- [Deploying](../tutorials/feature-guides/deploying.md): where the cache lives
  on a server or in a container.
- [Coming from other caches](../migration_guide.md): `lru_cache`, joblib and
  diskcache side by side with `@cash.cache`.
