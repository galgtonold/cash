"""A class a function makes per call is not kept alive by cash's memos.

The walks remember answers per class (an estimator? from the user's own
module? which slots? which library family? a lookup of its own?). Kept in
plain dicts keyed by the class, every class a factory defined inside it
stayed alive for the session, with its methods and what they held: a memo
the factory made, a model, a frame. The answers are held weakly now.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from cash._type_memo import TypeMemo
from cash._user_attrs import attribute_or
from cash.notebook import borrowed_memory, shared_objects


def _make(kind):
    payload = list(range(1000))

    if kind == "plain":

        class Made:
            def __init__(self):
                self.rows = payload

    elif kind == "slots":

        class Made:
            __slots__ = ("rows",)

            def __init__(self):
                self.rows = payload

    elif kind == "estimator":

        class Made:
            def __init__(self):
                self.rows = payload

            def get_params(self):
                return {}

            def fit(self, x):
                return self

    else:

        class Made:
            def __init__(self):
                self.rows = payload

            def __getattribute__(self, name):
                return object.__getattribute__(self, name)

    return Made()


@pytest.mark.parametrize("kind", ["plain", "slots", "estimator", "own_lookup"])
def test_the_class_goes_when_its_objects_do(kind):
    obj = _make(kind)
    cls = type(obj)
    # Every per-class question the walks ask of it.
    shared_objects.attributes_of(obj)
    shared_objects._is_estimator_class(cls)
    shared_objects._is_local_module_class(cls)
    shared_objects._slot_members(cls)
    borrowed_memory._family_of_type(cls)
    attribute_or(obj, "fit")
    gone = weakref.ref(cls)
    del obj, cls
    gc.collect()
    assert gone() is None


def test_an_answer_is_found_again_and_not_by_a_new_class_at_the_same_address():
    memo: TypeMemo[str] = TypeMemo()

    class A:
        pass

    memo[A] = "a"
    assert memo.get(A) == "a"
    assert A in memo
    assert memo[A] == "a"
    old = id(A)
    del A
    gc.collect()
    assert len(memo) == 0

    class B:
        pass

    # Whatever address B got (often A's), it has no answer of A's.
    assert memo.get(B) is None
    assert B not in memo
    del old


def test_slots_are_still_read():
    obj = _make("slots")
    assert shared_objects.attributes_of(obj) == {"rows": list(range(1000))}
