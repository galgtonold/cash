"""Whatever a value goes through, its watched digest is a fresh hash's.

Random nested values (dicts, lists, sets, tuples, objects with a ``__dict__``
or slots, nested classes, functions and classes held by name, shared
sub-objects) take random changes: in place, through an alias, through
``__dict__`` and ``__class__``, class attributes and ``__getstate__``,
global rebinding, ``__qualname__``, ``__code__``, copyreg reducers,
``sys.modules``, and lists rebuilt so their addresses come back. After
every step each value is hashed through the watchers twice and compared
with a hash read from the value (`_fuzz`). A longer run:
``python -m tests.test_core.watchers._fuzz --seconds 600``.
"""

from __future__ import annotations

import pytest

from cash import value_watch
from tests.test_core.watchers._fuzz import fuzz


@pytest.mark.parametrize("seed", range(8))
def test_every_watched_digest_is_a_fresh_hash(watching, seed):
    hits = value_watch.stats()["hits"]
    fuzz(seed, 200, watching)
    assert value_watch.stats()["hits"] > hits, "the fast path was never taken"
