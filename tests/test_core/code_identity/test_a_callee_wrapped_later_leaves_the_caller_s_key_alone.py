"""A caller's key does not depend on whether a plain function it calls was
also wrapped with ``cache`` before the caller was first keyed.

A library that caches through a ``cache(func)(...)`` helper (joblib's
``Memory.cache``, swapped for cash's) wraps its callees inside its own
bodies, on first use. Which of them were wrapped by the time a caller was
first keyed then depended on the order the program ran in, and on which
earlier calls were hits (a hit runs no body, so wraps nothing): the
caller's key took the callee in as a cached function in one run and left
it out in the next, and every run missed with "it no longer uses cached
function ...". The callee's code is in the caller's key either way, walked
as the plain function the caller calls.
"""

from __future__ import annotations

from cash import Cash, DataSource, FileBackend


FACTOR = 3


def _times(x, k):
    return x * k


def scale(x, k=FACTOR):
    return _times(x, k)


def run_cached(func, x):
    return func(x)


def outer(x):
    return run_cached(scale, x) + 1


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path / "c")))


def test_wrapping_the_callee_first_leaves_the_caller_s_key_alone(tmp_path):
    first = _cash(tmp_path).cache(outer)
    unwrapped = first.explain(2).cache_key

    c = _cash(tmp_path)
    c.cache(scale)
    wrapped = c.cache(outer).explain(2).cache_key

    assert wrapped == unwrapped


def test_a_hit_that_wraps_nothing_still_hits_the_caller(tmp_path):
    c = _cash(tmp_path)
    c.cache(scale)
    assert c.cache(outer)(2) == 7

    again = _cash(tmp_path).cache(outer)
    assert again.explain(2).reason == "hit"


def test_a_callee_wrapped_first_still_keys_its_edit(tmp_path, monkeypatch):
    c = _cash(tmp_path)
    c.cache(scale)
    cached = c.cache(outer)
    before = cached.explain(2).cache_key

    def scale_edited(x, k=FACTOR):
        return _times(x, k + 1)

    monkeypatch.setattr(scale, "__code__", scale_edited.__code__)
    c2 = _cash(tmp_path)
    c2.cache(scale)
    assert c2.cache(outer).explain(2).cache_key != before


def test_a_callee_wrapped_first_still_keys_its_default(tmp_path, monkeypatch):
    c = _cash(tmp_path)
    c.cache(scale)
    cached = c.cache(outer)
    before = cached.explain(2).cache_key
    monkeypatch.setattr(scale, "__defaults__", (4,))
    assert cached.explain(2).cache_key != before


def test_a_source_the_callee_depends_on_still_keys_the_caller(tmp_path):
    class Version(DataSource):
        def __init__(self) -> None:
            self.token = "v1"

        def get_id(self) -> str:
            return "version"

        def state_token(self) -> str:
            return self.token

    version = Version()
    c = _cash(tmp_path)
    c.cache(depends_on=[version])(scale)
    cached = c.cache(outer)
    before = cached.explain(2).cache_key
    version.token = "v2"
    assert cached.explain(2).cache_key != before
