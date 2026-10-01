"""A function reaching an annotated class is cached on every Python.

On 3.14 (PEP 649) an annotated class body compiles to an ``__annotate__``
function closing over ``__classdict__``, the class's own namespace. That
cell is the compiler's, not a value the user captured; hashing it raised
KEY-UNHASHABLE-CAPTURE, so every function reaching a dataclass ran uncached.

No ``from __future__ import annotations`` here: it would turn the
annotations into strings and hide the case.
"""

import dataclasses
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


@dataclasses.dataclass
class _Config:
    weight: int = 2

    def apply(self, v):
        return v * self.weight


def test_a_function_reaching_a_dataclass_still_caches(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    runs: list = []

    @c.cache
    def use(v):
        runs.append(1)
        return _Config().apply(v)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert use(1) == 2
        assert use(1) == 2
    assert not [w for w in record if "KEY-UNHASHABLE-CAPTURE" in str(w.message)]
    assert len(runs) == 1, "an unchanged call was not served from the cache"
