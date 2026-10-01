"""Plain data whose deepest level is empty keys by content.

``[np.float64(1.0), []]`` -- a score and an empty tag list -- has an empty
level below it. Keying it raised StopIteration: as a ``*args`` value the
argument silently dropped out of the key (the second call was served the
first one's result), as a named one the call never cached.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash import Cash, FileBackend


@pytest.fixture
def key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    return lambda *args, **kwargs: c._args.hash_payload(args, kwargs)


@pytest.mark.parametrize(
    "make",
    [
        lambda x: [np.float64(x), []],
        lambda x: (np.float64(x), ()),
        lambda x: [[np.int64(x)], [[]]],
    ],
    ids=["list", "tuple", "deeper"],
)
def test_numpy_scalars_beside_an_empty_list_key_by_content(key, make):
    assert key(make(1)) != key(make(2))
    assert key(make(1)) == key(make(1))
    assert key(rows=make(1)) != key(rows=make(2))


def test_a_star_args_value_stays_in_the_key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def total(*parts):
        return sum(float(x) for p in parts for x in p if not isinstance(x, list))

    assert total([np.float64(1.0), []]) == 1.0
    assert total([np.float64(2.0), []]) == 2.0
