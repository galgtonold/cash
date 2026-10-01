"""The optional-generator idiom draws unseeded when the caller passes none.

``rng = np.random.default_rng() if rng is None else rng`` and ``rng = rng or
np.random.default_rng()`` froze the first draw with no RANDOM-UNSEEDED, while
the statement form ``if rng is None: rng = default_rng()`` warned: the
detector only recognised a constructor assigned directly.
"""

from __future__ import annotations

import warnings

import pytest

from cash import Cash

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


def conditional(n, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return float(rng.random())


def either(n, rng=None):
    rng = rng or np.random.default_rng()
    return float(rng.random())


def reversed_conditional(n, rng=None):
    g = rng if rng is not None else np.random.default_rng()
    return float(g.random())


def seeded_fallback(n, rng=None):
    g = rng if rng is not None else np.random.default_rng(0)
    return float(g.random())


def _unseeded(tmp_path, fn):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(fn)(1)  # the source is judged when it is decorated
    return [w for w in rec if "RANDOM-UNSEEDED" in str(w.message)]


@pytest.mark.parametrize("fn", [conditional, either, reversed_conditional], ids=lambda f: f.__name__)
def test_an_unseeded_fallback_generator_warns(tmp_path, fn):
    assert _unseeded(tmp_path, fn)


def test_a_seeded_fallback_generator_is_silent(tmp_path):
    """Control: every constructor in the expression is seeded."""
    assert not _unseeded(tmp_path, seeded_fallback)
