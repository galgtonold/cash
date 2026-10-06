"""``compute_hash`` reads the attributes a class of yours leaves out of its pickle.

The notebook keys a call's arguments and checks a value for change with
``compute_hash``, which pickled the value: a class whose ``__getstate__``
leaves out a runtime setting hashed alike for every setting.
"""

from __future__ import annotations

from cash.value_hash import compute_hash


class Model:
    def __init__(self, weights, precision=2):
        self.weights = weights
        self.precision = precision

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("precision")
        return state


def test_the_left_out_setting_changes_the_hash():
    assert compute_hash(Model([1.0])) != compute_hash(Model([1.0], precision=5))
    assert compute_hash(Model([1.0], precision=5)) == compute_hash(Model([1.0], precision=5))
    assert compute_hash([Model([1.0])]) != compute_hash([Model([1.0], precision=5)])
