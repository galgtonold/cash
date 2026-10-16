"""``seed.sample(xs, k)`` on a parameter is a draw on the argument.

networkx's ``betweenness_centrality`` takes ``seed`` through
``@py_random_state``, which hands the body a ``random.Random``, and calls
``seed.sample(...)``. Decorating it warned RANDOM-UNSEEDED, "seed.sample()
without random_state": the scan for library calls read any ``.sample()``
as a DataFrame's draw from numpy's global stream. A generator passed in is
keyed by its state, so its draw is neither unseeded nor frozen.

A DataFrame passed in does draw from numpy's global stream; that is said
when the call runs and the stream moves. A ``.sample()`` on a frame the
body builds still warns when it is decorated.
"""

from __future__ import annotations

import random
import warnings

import pytest

from cash import Cash

pd = pytest.importorskip("pandas")


def _warned(tmp_path, fn, *args) -> tuple[list[str], list[str]]:
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with warnings.catch_warnings(record=True) as at_decoration:
        warnings.simplefilter("always")
        cached = c.cache(fn)
    with warnings.catch_warnings(record=True) as at_call:
        warnings.simplefilter("always")
        cached(*args)
    pick = lambda rec: [str(w.message) for w in rec if "RANDOM-UNSEEDED" in str(w.message)]
    return pick(at_decoration), pick(at_call)


def subset(xs, k, seed):
    return seed.sample(xs, k)


def frame_sample(df):
    return df.sample(3).a.tolist()


def built_frame_sample(n):
    df = pd.DataFrame({"a": range(10)})
    return df.sample(n).a.tolist()


def test_a_generator_passed_in_is_not_reported(tmp_path):
    assert _warned(tmp_path, subset, [1, 2, 3, 4], 2, random.Random(0)) == ([], [])


def test_a_frame_passed_in_is_reported_when_it_draws(tmp_path):
    """Control: the argument's draw from numpy's global stream is still said."""
    at_decoration, at_call = _warned(tmp_path, frame_sample, pd.DataFrame({"a": range(10)}))
    assert not at_decoration
    assert at_call and "numpy's global random stream" in at_call[0], at_call


def test_a_frame_the_body_builds_is_reported_when_decorated(tmp_path):
    """Control: a receiver that is no parameter keeps the decoration warning."""
    at_decoration, _ = _warned(tmp_path, built_frame_sample, 3)
    assert at_decoration and "df.sample()" in at_decoration[0], at_decoration
