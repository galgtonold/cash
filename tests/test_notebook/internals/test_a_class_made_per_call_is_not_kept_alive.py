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
    _collect()
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


#: Statements that hand the object on: a key reads it, a function gets it,
#: a copy is made of it.
HANDED = {
    "read": ("n = len(obj.rows)", "pair = [obj, obj]"),
    "passed": ("def use(o):\n    return len(o.rows)", "u = use(obj)"),
    "copied": ("import copy", "c = copy.copy(obj)", "del c"),
}


def _collect():
    """Python 3.12's ``inspect.getattr_static`` keeps the MRO tuples it was
    asked about (an ``lru_cache``; 3.13 keys it weakly): not cash's memo."""
    import inspect

    clear = getattr(getattr(inspect, "_shadowed_dict_from_mro_tuple", None), "cache_clear", None)
    if clear is not None:
        clear()
    gc.collect()


@pytest.mark.parametrize("handed", list(HANDED))
@pytest.mark.parametrize("persist", [False, True], ids=["ram", "persist"])
@pytest.mark.parametrize("kind", list(BODIES))
def test_a_statement_run_does_not_keep_the_class_alive(statement_processor, mock_shell, kind, persist, handed):
    """End to end: the statements' walks ask every per-class question of a
    class made per call; once its objects are gone, so is the class. Before,
    eight memos kept it alive (`own_dict`'s readers, the library kinds, the
    special functions, the slot readers, the user-class answers, the held
    function walk's memo and its per-type caches)."""
    from cash.analysis.annotations import CacheAnnotation

    annotation = CacheAnnotation(persist=True) if persist else None
    for source in (_factory(kind), "obj = make()", *HANDED[handed]):
        statement_processor.process_statement(source, annotation=annotation)
    gone = weakref.ref(type(mock_shell.user_ns["obj"]))
    statement_processor.process_statement("del obj" + (", pair" if handed == "read" else ""), annotation=annotation)
    statement_processor.process_statement("x = 1", annotation=annotation)
    # A stored ``obj = make()`` holds an object of the class, as a cache
    # entry should (in RAM too, when ``make()`` ran past the time floor):
    # with the entries dropped, nothing else does.
    statement_processor.cash_instance.backend.clear()
    _collect()
    assert gone() is None, [type(r).__name__ for r in gc.get_referrers(gone())]


def test_slots_are_still_read():
    obj = _make("slots")
    assert shared_objects.attributes_of(obj) == {"rows": list(range(1000))}


def test_a_class_given_new_bases_is_read_by_its_new_mro():
    """Its slots were remembered by place along its MRO: after
    ``B.__bases__ = (M, A)`` the place named another class, and reading an
    object of it raised ``KeyError``."""
    from cash import closure_sharing
    from cash.notebook import held_functions

    class A:
        __slots__ = ("x",)

    class M:
        __slots__ = ()

    class B(A):
        __slots__ = ("y",)

    b = B()
    b.x, b.y = 1, 2
    readers = (
        lambda: shared_objects.attributes_of(b),
        lambda: sorted(m.__get__(b) for m in closure_sharing._slot_readers(B)),
        lambda: held_functions._slot_getter(B, "x")(b),
    )
    before = [read() for read in readers]
    B.__bases__ = (M, A)
    assert [read() for read in readers] == before


@pytest.mark.parametrize("changes", [1, 2, 3, 4])
def test_an_answer_for_an_mro_given_up_is_not_found(changes):
    """Remembered by the MRO tuple's address: three ``__bases__``
    assignments in a row gave the new MRO the first one's memory, and the
    memo answered for bases the class no longer had."""
    from cash.notebook import held_functions

    class Compares:
        def __eq__(self, other):
            return True

        def __hash__(self):
            return 0

    wrong = 0
    for _ in range(50):
        bases = [type(f"B{i}", (), {}) for i in range(changes - 1)] + [Compares]
        kind = type("K", (type("A", (), {}),), {})
        assert held_functions._identity_hashed(kind)
        for base in bases:
            kind.__bases__ = (base,)
        wrong += held_functions._identity_hashed(kind) != held_functions._identity_hashed.__wrapped__(kind)
    assert wrong == 0


def helper_a(v):
    return v


def test_a_property_put_where_a_freed_one_was_is_read_afresh():
    """A property takes no weak reference: remembered by address, a new
    one at a freed one's address was read as the old one."""
    from cash.analysis import handed_callables

    def calls_a(self):
        return helper_a(self.a)

    def plain(self):
        return self.a

    class C:
        pass

    C.total = property(calls_a)
    assert handed_callables._reaching_special_functions(C) == [calls_a]
    for _ in range(50):
        old = id(vars(C)["total"])
        del C.total
        C.total = property(plain)
        assert handed_callables._reaching_special_functions(C) == []
        if id(vars(C)["total"]) == old:
            break
        C.total = property(calls_a)
        assert handed_callables._reaching_special_functions(C) == [calls_a]
    else:
        pytest.skip("no property was made at the freed address")
