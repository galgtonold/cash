"""A store pickle no store will take is stopped where it turns out so.

A digest of a value a store may keep pickles it as the store does, and
keeps those bytes for the store when they are also the keyed ones
(`stored_pickle.object_digest`). A fitted forest's trees are record arrays
with padding, which a key packs: the store's bytes are not the keyed ones,
they were thrown away, and the digest pickled the value again, keyed. A
VotingClassifier over two forests of 84 MB paid both pickles on every
statement that changed it (``voting_clf.voting = "soft"``: +1.75 s).

The store pickle now stops as soon as it meets such an object: the digest
is the keyed pickle's, as it always was.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash import kept_state, stored_pickle

REDUCED: list[int] = []


class Counted:
    """Counted each time a pickle reduces it."""

    def __reduce__(self):
        REDUCED.append(1)
        return (Counted, ())


class Box:
    """A plain object, pickled with its dict."""


def _padded(rows: int = 1 << 15) -> np.ndarray:
    dtype = np.dtype({"names": ["a", "b"], "formats": ["f8", "u1"], "offsets": [0, 16], "itemsize": 24})
    return np.zeros(rows, dtype=dtype)


@pytest.mark.parametrize("walked_first", [False, True], ids=["pickled", "walked-first"])
def test_the_store_pickle_stops_at_an_object_a_key_pickles_otherwise(walked_first, monkeypatch):
    """Walked first (3.10 on Windows, `kept_state.FLAT_FIRST`), the walk
    reduces the user's objects before the pickle writes a byte: it stops
    at the padded array too, before it reduces what follows."""
    if walked_first:
        monkeypatch.setattr(kept_state, "FLAT_FIRST", True)
    value = [_padded(), Counted()]
    outside = stored_pickle.object_digest(value)
    REDUCED.clear()
    with stored_pickle.keeping(lambda _value: True):
        inside = stored_pickle.object_digest(value)
        assert stored_pickle.kept(value) is None
    assert REDUCED == [1], "reduced by the keyed pickle only"
    assert inside == outside


def test_a_value_whose_store_bytes_are_the_keyed_ones_is_pickled_once_and_kept():
    box = Box()
    box.rows = np.arange(1 << 14, dtype=float)
    box.tail = Counted()
    outside = stored_pickle.object_digest(box)
    REDUCED.clear()
    with stored_pickle.keeping(lambda _value: True):
        inside = stored_pickle.object_digest(box)
        assert stored_pickle.kept(box) is not None
    assert REDUCED == [1]
    assert inside == outside


def test_a_change_past_the_padding_still_moves_the_digest():
    left, right = [_padded(), {"k": 1}], [_padded(), {"k": 2}]
    with stored_pickle.keeping(lambda _value: True):
        assert stored_pickle.object_digest(left) != stored_pickle.object_digest(right)
    right[1]["k"] = 1
    right[0].view(np.uint8).reshape(-1, 24)[:, 20] = 7  # padding only
    with stored_pickle.keeping(lambda _value: True):
        assert stored_pickle.object_digest(left) == stored_pickle.object_digest(right)
