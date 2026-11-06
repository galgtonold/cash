"""Equal objects digest alike for a lineage, however they were built.

Four estimators made in one cell share their parameter-name strings (and
``"gini"``, ``"sqrt"``); the same four restored from two cache entries hold
copies. Pickle's memo writes a string met again as a back-reference, so the
list of them pickled to other bytes though nothing in it differed, and a loop
that fitted them gave the list another lineage depending on which statements
had been served: whether the share check of a fast statement finished within
its 0.1 s decided it. `unshared_hash` digests the content alone.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from cash.value_hash import compute_hash, unshared_hash


class Est:
    def __init__(self, criterion="gini", max_features="sqrt"):
        self.criterion = criterion
        self.max_features = max_features
        self.params = ("criterion", "max_features")

    def fit(self, X):
        self.n_seen_ = len(X)
        return self


def _made_together():
    return [Est(), Est(criterion="entropy")]


def _restored_apart():
    return [pickle.loads(pickle.dumps(m)) for m in _made_together()]


def test_a_list_restored_apart_digests_as_one_made_together():
    together, apart = _made_together(), _restored_apart()
    for models in (together, apart):
        for m in models:
            m.fit(range(10))

    # The memo's view: equal content, other bytes. Were this to hold, the
    # value digest would need no second form.
    assert compute_hash(together) != compute_hash(apart)
    assert unshared_hash(together) == unshared_hash(apart)


def test_a_changed_parameter_still_changes_the_digest():
    """Positive control: blind to sharing, not to content."""
    changed = _made_together()
    changed[1].max_features = "log2"

    assert unshared_hash(changed) != unshared_hash(_made_together())


def test_two_names_for_one_list_still_digest_apart_from_two_lists():
    """A changeable object met twice is still marked: ``[a, a]`` and
    ``[a, copy_of_a]`` are not one value to code that changes ``a``."""
    a = [1, 2]

    assert unshared_hash([a, a]) != unshared_hash([a, [1, 2]])


def test_a_frame_an_array_or_a_value_pickle_cannot_read_digests_as_before():
    frame = pd.DataFrame({"a": [1, 2]})
    array = np.arange(4)
    unpicklable = [lambda: 1]

    assert unshared_hash(frame) == compute_hash(frame)
    assert unshared_hash(array) == compute_hash(array)
    assert unshared_hash([frame, 1]) == compute_hash([frame, 1])
    assert unshared_hash(unpicklable) == compute_hash(unpicklable)
