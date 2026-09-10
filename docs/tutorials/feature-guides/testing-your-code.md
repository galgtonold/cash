# Testing code that uses cash

A test suite is the one place where a cache hit can make a result *look*
right without it being right. This page is how to keep `@cash.cache` from
vouching for code it did not run.

## The trap: a determinism test that the cache passes for you

<!-- test:skip reason="illustrative: simulate and params belong to the reader's project" -->
```python
def test_simulation_is_reproducible():
    assert simulate(params, seed=1) == simulate(params, seed=1)
```

Under caching, the second call is a hit, so the assertion compares one result
with itself. It passes even if `simulate` ignores its seed and draws fresh
random numbers every time — and it passes on a fresh cache too, because the
first call writes the entry the second call reads. Any test that calls a
cached function twice to check that it agrees with itself is testing the
cache.

Here is the whole trap, runnable — a function that ignores its seed, and the
two checks side by side:

```python
import random
import cash

@cash.cache(allow_random=True)
def simulate(n, seed=0):
    return random.random()          # the bug: seed is never used

# Through the cache: the second call is a hit, so this compares a value with itself.
assert simulate(3, seed=1) == simulate(3, seed=1)

# The undecorated function tells the truth.
assert simulate.__wrapped__(3, seed=1) != simulate.__wrapped__(3, seed=1)
```

Two fixes, and CI wants both:

- in a test that is *about* the function's own behaviour, call the undecorated
  function, `simulate.__wrapped__(...)`;
- run the whole suite once with caching off, which catches the tests nobody
  thought to write that way.

## Turning caching off: `CASH_DISABLE=1`

<!-- claim: cash/config.py:CashConfig.disable == False, cash/core.py:Cash._wrap_with_stats @3025ea40 -->
```bash
CASH_DISABLE=1 pytest
```

Every `@cash.cache` function then calls straight through: no key, no lookup,
no store, no analysis, and nothing is read from or written to the cache
directory. `f.explain(...)` reports `disabled`, and with `CASH_SUMMARY=1` the
exit summary says caching was off and how many calls ran uncached. In a
notebook, `%cash_on` declines to switch caching on and says why — so a CI job
that executes notebooks with the variable set runs them uncached too.

The same switch is `Cash(disable=True)`, `cash.configure(disable=True)` (read
on every call, so it takes effect immediately), or `disable = true` under
`[tool.cash]`.

A CI job that runs the suite this way costs one extra run, and it is the only
thing that shows the suite passes on the code's merits.

## Isolating the suite's cache

`pytest` is installed code, so it caches where your project is — in the same
`.cash` your application uses (see
[what paths are relative to](../../getting-started/configuration.md#what-paths-are-relative-to)).
A test run then reads entries that a previous run, or the application, wrote.
To give each session a cache of its own:

<!-- test:skip reason="a conftest.py fixture; exercised by running pytest twice in a scratch project" -->
```python
# conftest.py
import cash
import pytest

@pytest.fixture(scope="session", autouse=True)
def _cash_isolated(tmp_path_factory):
    cash.configure(cache_dir=str(tmp_path_factory.mktemp("cash")))
```

Or from the outside, with no code at all:

```bash
CASH_CACHE_DIR="$(mktemp -d)" pytest
```

`cash.configure(backend="memory")` keeps the cache in RAM only: nothing is
written anywhere, and each test process — every xdist worker is one — starts
empty.

## Which to use

| You want | Use |
|---|---|
| The suite to prove the code, not the cache | a CI job with `CASH_DISABLE=1` |
| Local runs that never see another run's entries | a session `cache_dir` from `tmp_path_factory` |
| A test about the function itself | `f.__wrapped__(...)` |
| A test about the caching | a fresh `Cash(cache_dir=tmp_path)` per test |
