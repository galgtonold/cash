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

from cash._type_memo import TypeMemo, type_key
from cash._user_attrs import attribute_or, own_dict
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
    own_dict(obj)
    attribute_or(obj, "fit")
    gone = weakref.ref(cls)
    del obj, cls
    gc.collect()
    assert gone() is None


def test_an_answer_is_found_again_and_not_by_a_new_class_at_the_same_address():
    memo: TypeMemo[str] = TypeMemo()

    class A:
        pass

    memo.put(A, "a")
    assert memo.get(type_key(A)) == "a"
    old = id(A)
    old_key = type_key(A)
    del A
    gc.collect()
    assert len(memo) == 0
    # Make classes until one lands at A's address: it has no answer of A's.
    made = []
    for _ in range(20_000):
        made.append(type("B", (), {}))
        if id(made[-1]) == old:
            break
    else:
        pytest.skip("no class was made at the freed address")
    assert memo.get(old_key) is None


def test_a_memo_is_not_written_by_item():
    with pytest.raises(TypeError):
        TypeMemo()[int] = 1


BODIES = {
    "plain": "",
    "slots": "        __slots__ = ('rows',)\n",
    "dict_subclass": None,
    "dataclass": None,
}


def _factory(kind):
    if kind == "dict_subclass":
        return (
            "def make():\n"
            "    class Made(dict):\n"
            "        def __init__(self):\n"
            "            self.rows = list(range(1000))\n"
            "            self['rows'] = self.rows\n"
            "    return Made()"
        )
    if kind == "dataclass":
        return (
            "import dataclasses\n"
            "def make():\n"
            "    @dataclasses.dataclass\n"
            "    class Made:\n"
            "        rows: list = dataclasses.field(default_factory=lambda: list(range(1000)))\n"
            "    return Made()"
        )
    return (
        "def make():\n"
        "    class Made:\n"
        f"{BODIES[kind]}"
        "        def __init__(self):\n"
        "            self.rows = list(range(1000))\n"
        "    return Made()"
    )


@pytest.mark.parametrize("persist", [False, True], ids=["ram", "persist"])
@pytest.mark.parametrize("kind", list(BODIES))
def test_a_statement_run_does_not_keep_the_class_alive(statement_processor, mock_shell, kind, persist):
    """End to end: the statements' walks ask every per-class question of a
    class made per call; once its objects are gone, so is the class. Before,
    eight memos kept it alive (`own_dict`'s readers, the library kinds, the
    special functions, the slot readers, the user-class answers, the held
    function walk's memo and its per-type caches)."""
    from cash.analysis.annotations import CacheAnnotation

    annotation = CacheAnnotation(persist=True) if persist else None
    for source in (_factory(kind), "obj = make()", "n = len(obj.rows)", "pair = [obj, obj]"):
        statement_processor.process_statement(source, annotation=annotation)
    gone = weakref.ref(type(mock_shell.user_ns["obj"]))
    statement_processor.process_statement("del obj, pair", annotation=annotation)
    statement_processor.process_statement("x = 1", annotation=annotation)
    if persist:
        # The stored ``obj = make()`` holds an object of the class, as a
        # cache entry should: with the entries dropped, nothing else does.
        statement_processor.cash_instance.backend.clear()
    gc.collect()
    assert gone() is None, [type(r).__name__ for r in gc.get_referrers(gone())]


def test_slots_are_still_read():
    obj = _make("slots")
    assert shared_objects.attributes_of(obj) == {"rows": list(range(1000))}
