# Testing code that uses cash

!!! info "Applies to: decorator"
    Anyone writing a test suite for code that uses `@cash.cache`.

In a test suite, a cache hit can make a wrong result look right. This page shows
how to keep the cache from vouching for code it did not run.

## The trap: the cache passes your determinism test

```python
import random
import cash

@cash.cache(allow_random=True)
def simulate(n, seed=0):
    return random.random()          # the bug: seed is never used

# Through the cache, the second call is a hit: a value compared with itself.
assert simulate(3, seed=1) == simulate(3, seed=1)

# The undecorated function tells the truth.
assert simulate.__wrapped__(3, seed=1) != simulate.__wrapped__(3, seed=1)
```

Any test that calls a cached function twice to check that it agrees with itself
is testing the cache. It passes on an empty cache too, because the first call
writes what the second reads. Two fixes, and CI wants both:

- In a test **about the function's behaviour**, call the undecorated function,
  `f.__wrapped__(...)`.
- Run the whole suite once with caching off. That catches the tests nobody
  thought to write that way.

## Turning caching off

<!-- claim: cash/config.py:CashConfig.disable == False, cash/core.py:Cash._wrap_with_stats @b50986a7 -->
```bash
CASH_DISABLE=1 pytest
```

Every cached function then calls straight through: no key, no lookup, no store,
and nothing read from or written to the cache folder. `f.explain(...)` reports
`disabled`. The same switch is `Cash(disable=True)`,
`cash.configure(disable=True)` (it takes effect on the next call), or
`disable = true` under `[tool.cash]`.

## Isolating the suite's cache

A test run caches in your project's `.cash` folder by default, the same one the
application uses, so it reads entries that earlier runs wrote. To give each
session a cache of its own:

<!-- test:skip reason="a conftest.py fixture, not a standalone script" -->
```python
# conftest.py
import cash
import pytest

@pytest.fixture(scope="session", autouse=True)
def _cash_isolated(tmp_path_factory):
    cash.configure(cache_dir=str(tmp_path_factory.mktemp("cash")))
```

Or, with no code: `CASH_CACHE_DIR="$(mktemp -d)" pytest`. With
`cash.configure(backend="memory")` nothing is written anywhere, and each test
process (every xdist worker is one) starts empty.

To keep the cache for most tests and switch it off only for those that must
not see it, use a fixture:

<!-- claim: cash/__init__.py:disabled @c23cd1a2 -->
<!-- test:skip reason="a conftest.py fixture, not a standalone script" -->
```python
# conftest.py
import cash
import pytest

@pytest.fixture
def no_cache():
    with cash.disabled():             # every cached call runs its body
        yield

# test_model.py
def test_training_really_trains(no_cache):
    ...
```

`cash.disabled()` restores whatever was in force before the block, so a run
started with `CASH_DISABLE=1` stays uncached. Don't end the fixture with
`cash.configure(disable=False)`: that switches caching **on** for the rest of
the run.

## Mocking and monkeypatching

Patching a helper that a cached function calls works as the test expects. The
patched call gets its own entry, and the real one comes back when the patch is
undone:

<!-- test:skip reason="needs importable modules primes and sievelib" -->
```python
# primes.py
from sievelib import sieve as _sieve

@cash.cache
def count(n):
    return len(_sieve(n))

# test_primes.py
def test_count_uses_the_sieve(monkeypatch):
    monkeypatch.setattr(primes, "_sieve", lambda n: [2, 3])   # patch where it is used
    assert primes.count(10) == 2

def test_real_count():
    assert primes.count(10**6) == 78498                        # not the patched answer
```

Patch the name where it is **used** (`primes._sieve`), as with any mocking.
Patching `sievelib.sieve` after `primes` imported it changes nothing `count`
runs.

<!-- claim: cash/purity_analyzer.py:is_mock @173f99ff -->
**A call that reaches a `unittest.mock` object runs uncached**, and nothing it
returns is stored. That covers `mock.patch(...)`, `MagicMock`, `pytest-mock`'s
`mocker` and `autospec=True` patches, whether the mock replaces your helper, a
library function (`mock.patch("requests.get")`) or a whole module.

<!-- claim: cash/effect_observer.py:_hook_mock_calls @fbaaf89b -->
A mock deeper down, such as `mock.patch("requests.Session.request")`, is not
part of the key, so cash can't tell the call from a real one before it runs.
It can afterwards: a call during which any mock was called is **not stored**,
so a fake answer never reaches a later real run. The other direction is not
covered: if a real run already stored the entry, the mocked call is a hit and
gets the real answer. A test that must see its own mock needs the `no_cache`
fixture.

Libraries that intercept HTTP without replacing the function you call
(`responses`, `requests-mock`, `vcrpy`, `httpretty`) are invisible to cash. A
recorded response can be stored under the real key, so run those tests with the
`no_cache` fixture.

<!-- claim: cash/_clock.py:perf_counter @9808b623, cash/_plain_data.py:fake_clock @bcb59121 -->
**freezegun** and other clock doubles are not mocks in this sense: the call is
cached as usual. Cash measures run time with a clock `freeze_time` does not
reach, and a frozen `date` is keyed as the date it equals. TTLs do follow the
frozen clock, since they use `time.time()`.

## Which to use

| You want | Use |
|---|---|
| The suite to prove the code, not the cache | a CI job with `CASH_DISABLE=1` |
| Local runs that never see another run's entries | a session `cache_dir` from `tmp_path_factory` |
| A test about the function itself | `f.__wrapped__(...)` |
| A test about the caching | a fresh `Cash(cache_dir=tmp_path)` per test |

For caching inside CI pipelines, see [Deploying](deploying.md#ci).
