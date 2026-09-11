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

<!-- claim: cash/config.py:CashConfig.disable == False, cash/core.py:Cash._wrap_with_stats @da9cc76d -->
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

A session-wide cache directory keeps a test run from reading the application's
entries, and it also gives up everything the cache saves you between local
runs. To keep that, and switch the cache off only for the tests that must not
see it, use a fixture:

<!-- test:skip reason="a conftest.py fixture; configure(disable=...) is exercised by tests/test_core/test_disable.py" -->
```python
# conftest.py
import cash
import pytest

@pytest.fixture
def no_cache():
    cash.configure(disable=True)      # every @cash.cache call runs its body
    yield
    cash.configure(disable=False)

# test_model.py
def test_training_really_trains(no_cache):
    ...
```

## Mocking and monkeypatching

Patching a helper that a cached function calls works the way the test expects:
the helper is looked up through the name the cached function's module uses, so
the patched call gets its own entry and the real one comes back when the patch
is undone.

<!-- test:skip reason="needs importable modules; exercised by tests/test_core/test_patched_helper_binding.py" -->
```python
# primes.py
from sievelib import sieve as _sieve

@cash.cache
def count(n):
    return len(_sieve(n))

# test_primes.py
def test_count_uses_the_sieve(monkeypatch):
    monkeypatch.setattr(primes, "_sieve", lambda n: [2, 3])   # patch where it is USED
    assert primes.count(10) == 2

def test_real_count():
    assert primes.count(10**6) == 78498                        # not the patched answer
```

Patch the name where it is **used** (`primes._sieve`), as with any mocking.
Patching `sievelib.sieve` after `primes` imported it changes nothing `count`
runs, and so nothing about its key. That holds at any depth: a helper's own
helpers are looked up in the helper's module.

A `unittest.mock` object (`mock.patch(..., return_value=...)`, `MagicMock`,
`pytest-mock`'s `mocker`) has no code for cash to key, and its answer is
whatever the test configured, so a call that reaches one **runs uncached**.
That is usually what a test with a mock wants.

## Which to use

| You want | Use |
|---|---|
| The suite to prove the code, not the cache | a CI job with `CASH_DISABLE=1` |
| Local runs that never see another run's entries | a session `cache_dir` from `tmp_path_factory` |
| A test about the function itself | `f.__wrapped__(...)` |
| A test about the caching | a fresh `Cash(cache_dir=tmp_path)` per test |
