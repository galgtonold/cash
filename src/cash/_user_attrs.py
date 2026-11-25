"""Reading an attribute of a notebook's object without its errors becoming cash's."""

from __future__ import annotations

import abc
import collections
import functools
import operator
import sys
import types
from itertools import compress, repeat
from typing import Any

from ._type_memo import TypeMemo, class_dict, class_mro, mro_mark, mro_of, per_type, same_mro, type_key, type_lookup
from .loaded_modules import loaded_module


def attribute_or(obj: Any, name: str, default: Any = None, *, asked: bool = False) -> Any:
    """``getattr(obj, name, default)``, with any error the lookup raises taken
    as the attribute missing.

    Python's own default covers AttributeError only. A class whose
    ``__getattr__`` raises KeyError for a name it does not hold (a record
    reading its fields from a dict) failed cash's own question about it --
    ``fit``, ``__code__`` -- and the cell with it, where plain Python never
    asks. The question is cash's, so the answer is: not there.

    ``__code__`` is there only as a code object: a config class whose
    ``__getattr__`` answers every name (``self._d.get(name, 0)``) hands back
    ``0``, which cash's walk of what a function runs read as code and failed
    the cell with.

    A class whose metaclass looks attributes up with code of its own (a
    ``__getattribute__`` or ``__getattr__`` in Python) is read by
    ``type``'s lookup: plain Python never runs that code for cash's
    questions. With *asked*, for a read of an attribute the user's code
    names (``Defaults.retries``, a dotted path a cell reads), a name that
    ``type`` does not find is asked of the metaclass's ``__getattr__`` as
    the user's code reading it would: what it answers is the value. A
    ``__dunder__`` name is never asked.
    """
    kind = type(obj)
    own = _OWN_LOOKUP.get(type_key(kind))
    if own is None:
        own = own_lookup(kind)
    try:
        if own:
            if issubclass(kind, type):
                return _class_lookup(obj, name, default, asked)
            found = object.__getattribute__(obj, name)
        else:
            found = getattr(obj, name, default)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return default
    if name == "__code__" and type(found) is not types.CodeType:
        return default
    return found


def plain_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr(obj, name, default)``, but a class whose metaclass looks its
    attributes up with code of its own is read by ``type``'s lookup
    (`attribute_or`): cash's question of it (its ``__module__``, a marker of
    cash's) runs nothing of the user's, as plain Python never asks it.
    Anything else is asked exactly as ``getattr`` asks it."""
    kind = type(obj)
    if kind is not type and issubclass(kind, type):
        own = _OWN_LOOKUP.get(type_key(kind))
        if own is None:
            own = own_lookup(kind)
        if own:
            return _class_lookup(obj, name, default, False)
    return getattr(obj, name, default)


def plain_hasattr(obj: Any, name: str) -> bool:
    """``hasattr(obj, name)`` as `plain_getattr` reads the name."""
    return plain_getattr(obj, name, _NO_DEFAULT) is not _NO_DEFAULT


def named_attr(obj: Any, name: str, *default: Any) -> Any:
    """``getattr(obj, name[, default])`` for a name the user's code reads (a
    step of a dotted path a body or a qualname names): a class whose
    metaclass hooks its lookups is read by ``type``'s, a name it lacks asked
    of the metaclass's ``__getattr__`` as the user's code would
    (`attribute_or` with *asked*). Anything else exactly as ``getattr``
    reads it; without *default* a missing name raises AttributeError."""
    kind = type(obj)
    if kind is not type and issubclass(kind, type):
        own = _OWN_LOOKUP.get(type_key(kind))
        if own is None:
            own = own_lookup(kind)
        if own:
            found = _class_lookup(obj, name, default[0] if default else _NO_DEFAULT, True)
            if found is _NO_DEFAULT:
                raise AttributeError(name)
            return found
    return getattr(obj, name, *default)


def _class_lookup(cls: type, name: str, default: Any, asked: bool) -> Any:
    """`attribute_or` of a class whose metaclass hooks its lookups."""
    try:
        found = type_lookup(cls, name)
    except AttributeError:
        if not asked or (name.startswith("__") and name.endswith("__")):
            return default
        hook = _in_mro(type(cls), "__getattr__")
        if type(hook) is not types.FunctionType:
            return default
        try:
            found = hook(cls, name)
        except Exception:  # noqa: BLE001 - the class's own error, not one of cash's
            return default
    except Exception:  # noqa: BLE001 - a descriptor's own error, not one of cash's
        return default
    if name == "__code__" and type(found) is not types.CodeType:
        return default
    return found


def importable_attr(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr(obj, name, default)``, and *default* too when the read
    raises ImportError.

    A lazily imported optional dependency (``resampy =
    lazy.load("resampy")`` with resampy not installed) is a module object
    whose every attribute raises ImportError: a call through it cannot run
    in this process, so it reaches no code to key. Installed later, the
    name resolves and the code it reaches changes the caller's key. Any
    other error still propagates: what it hides could be code the call
    runs. A lazily loaded module that IS installed is read with
    ``getattr`` like any module, so it loads and its functions are keyed.
    """
    try:
        if hooked_class(obj):
            return class_attr(obj, name, default)
        return getattr(obj, name, default)
    except ImportError:
        return default


def hooked_class(obj: Any) -> bool:
    """Is *obj* a class whose metaclass looks its attributes up with code of
    its own (`own_lookup`)? Read such a class with `class_attr`."""
    kind = type(obj)
    own = _OWN_LOOKUP.get(type_key(kind))
    if own is None:
        own = own_lookup(kind)
    return own and issubclass(kind, type)


def class_attr(cls: type, name: str, default: Any = None) -> Any:
    """``getattr(cls, name, default)`` for a class, found by ``type``'s own
    lookup: a ``__getattribute__`` or ``__getattr__`` of its metaclass's is
    not run (plain Python asks neither for what cash asks). Any error the
    lookup raises (a descriptor's) is the attribute missing.

    A class of a metaclass with no lookup of its own is read with
    ``getattr``, which finds the same and raises nothing for a name it
    lacks: most classes, and the reads that ask a class for a marker it
    rarely holds."""
    meta = type(cls)
    if meta is not type:
        own = _OWN_LOOKUP.get(type_key(meta))
        if own is None:
            own = own_lookup(meta)
        if own:
            try:
                return type_lookup(cls, name)
            except Exception:  # noqa: BLE001 - the class's own error, not one of cash's
                return default
    try:
        return getattr(cls, name, default)
    except Exception:  # noqa: BLE001 - the class's own error, not one of cash's
        return default


def dataclass_fields(kind: type) -> tuple | None:
    """``dataclasses.fields(kind)`` when *kind* is a dataclass
    (``dataclasses.is_dataclass``), else None: its ``__dataclass_fields__``
    read by `class_attr`, as ``dataclasses`` would ask the metaclass's
    hook of a class of the user's for it."""
    found = class_attr(kind, "__dataclass_fields__")
    if found is None:
        return None
    import dataclasses

    return dataclasses.fields(types.SimpleNamespace(__dataclass_fields__=found))


def unwrapped(func: Any) -> Any:
    """``unwrapped(func)``: the end of its ``__wrapped__`` chain, each
    link read by `attribute_or`, so a class's metaclass's hook is not asked
    for one. Raises ValueError on a chain that loops, as ``inspect`` does."""
    f = func
    memo = {id(f): f}
    limit = sys.getrecursionlimit()
    while True:
        inner = attribute_or(f, "__wrapped__", _ABSENT)
        if inner is _ABSENT:
            return f
        f = inner
        if id(f) in memo or len(memo) >= limit:
            raise ValueError(f"wrapper loop when unwrapping {func!r}")
        memo[id(f)] = f


_NO_DEFAULT: Any = object()


def static_attr(obj: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    """``static_attr(obj, name, default)``, with every class's
    namespace read through ``type``'s own descriptor.

    ``inspect.getattr_static`` reads ``klass.__dict__`` for each class
    along the MRO, which runs a ``__getattribute__`` of the metaclass's:
    plain Python never asks it. Raises AttributeError like it without a
    *default*."""
    kind = type(obj)
    if type in class_mro(kind):
        klass = obj
        held = _ABSENT
    else:
        klass = kind
        attributes = own_dict(obj, subclasses=True)
        held = dict.get(attributes, name, _ABSENT) if attributes is not None else _ABSENT
    found = _in_mro(klass, name)
    if held is not _ABSENT:
        if found is not _ABSENT:
            found_kind = type(found)
            if _in_mro(found_kind, "__get__") is not _ABSENT and (
                _in_mro(found_kind, "__set__") is not _ABSENT or _in_mro(found_kind, "__delete__") is not _ABSENT
            ):
                return found
        return held
    if found is not _ABSENT:
        return found
    if obj is klass:
        # For a class, its metaclass's too.
        found = _in_mro(kind, name)
        if found is not _ABSENT:
            return found
    if default is not _NO_DEFAULT:
        return default
    raise AttributeError(name)


def _in_mro(klass: type, name: str) -> Any:
    """What the first class along *klass*'s MRO holding *name* holds, or
    `_ABSENT`."""
    for entry in class_mro(klass):
        held = class_dict(entry)
        if name in held:
            return held[name]
    return _ABSENT


def module_file(module: Any) -> str | None:
    """*module*'s ``__file__``, None without a string one, read from its
    namespace: a module's own ``__getattr__`` is never asked. A lazily
    imported module (``optional_pkg`` not installed) raises
    ModuleNotFoundError from it for every name it lacks, and a builtin or
    namespace module lacks ``__file__``: asked with ``getattr``, that failed
    the key of a cell that never touched the module. Anything but a module
    is asked through `attribute_or`."""
    if isinstance(module, types.ModuleType):
        try:
            found = object.__getattribute__(module, "__dict__").get("__file__")
        except Exception:  # noqa: BLE001 - a module subclass's own error: no file known
            return None
    else:
        found = attribute_or(module, "__file__")
    return found if isinstance(found, str) else None


def metadata_attribute(obj: Any, name: str, default: Any = None) -> Any:
    """`attribute_or` for a fact cash only reads about *obj* (its
    ``nbytes``, its ``shape``), never running the user's code for it.

    Looked up in the class dicts first, as Python does, without asking the
    object: a property or other descriptor a class of the user's defines
    (``@property def nbytes``) is not run, and a name no class holds is
    read from the instance's own ``__dict__``, never passed to a
    ``__getattr__`` of the user's. Plain Python runs neither unless the
    user's code asks. A library's descriptor (numpy's ``nbytes``) is read
    as before.
    """
    kind = type(obj)
    fallback = None
    for klass in _MRO(kind):
        held = _CLASS_DICT(klass)
        found = held.get(name, _ABSENT)
        if found is not _ABSENT:
            if _users_class(klass) and _descriptor(found):
                return default
            return attribute_or(obj, name, default)
        if fallback is None and "__getattr__" in held:
            fallback = klass
    if fallback is not None and _users_class(fallback):
        held = own_dict(obj)
        return held.get(name, default) if held is not None else default
    return attribute_or(obj, name, default)


def _descriptor(found: Any) -> bool:
    """Does *found*, held by a class, compute its value (``__get__``)?"""
    return any("__get__" in _CLASS_DICT(klass) for klass in _MRO(type(found)))


def from_family(kind: type, family: str) -> bool:
    """Is *kind* defined in the package *family* (``"pandas"``,
    ``"scipy.sparse"``) or one of its modules? Read off the class itself,
    never the user's code: a class of the user's named ``DataFrame`` or
    ``csr_matrix`` had its ``memory_usage``, ``_mgr`` or ``data`` run by a
    size check that went by the name alone."""
    module = _module_of(kind)
    return type(module) is str and (module == family or module.startswith(family + "."))


def user_defines(kind: type, name: str) -> bool:
    """Is *name* on *kind* found in a class of the user's (the first in its
    MRO whose own dict holds it)?"""
    for klass in _MRO(kind):
        if name in _CLASS_DICT(klass):
            return _users_class(klass)
    return False


@per_type
def _users_class(klass: type) -> bool:
    """Is *klass* defined in a module of the user's? Per class: its module
    is fixed when it is made."""
    from .install_paths import is_user_code_module

    name = _module_of(klass)
    if name is None:
        # ``type('W', (), {})`` run in a dict with no ``__name__`` has no
        # module at all: only the user's code makes one, so it is the user's.
        return True
    module = sys.modules.get(name) if type(name) is str else None
    return module is not None and is_user_code_module(module)


def _module_of(klass: type) -> Any:
    """*klass*'s ``__module__``, or None for a class made with none."""
    try:
        return _MODULE(klass)
    except AttributeError:
        return None


def forget_user_classes() -> None:
    """Forget which classes are the user's (`_users_class`): for
    `install_paths.clear_caches`, when what counts as the user's changes."""
    _users_class.memo.clear()


_MRO = mro_of
_CLASS_DICT = type.__dict__["__dict__"].__get__
_MODULE = type.__dict__["__module__"].__get__
_ABSENT = object()


#: Per type, `own_lookup`'s answer, held weakly: a class made per call
#: (``df.itertuples()``) is not kept alive.
_OWN_LOOKUP: TypeMemo[bool] = TypeMemo()


def own_lookup(kind: type) -> bool:
    """Does *kind* look its attributes up with a ``__getattribute__`` of
    its own, written in Python? cash's questions about such an object
    (``fit``, ``__code__``, the ``__class__`` an ``isinstance`` asks for
    when the type does not match) ran it, where plain Python never asks:
    they are answered by Python's generic lookup and the real type."""
    own = _OWN_LOOKUP.get(type_key(kind))
    if own is None:
        # Found along the MRO, as ``kind.__getattribute__`` finds it (the
        # metaclass holds no data descriptor by that name) but without
        # asking the metaclass of a class of the user's. A class (*kind* a
        # metaclass) counts one with a ``__getattr__`` in Python too: cash's
        # questions of a class go to ``type``'s lookup (`attribute_or`).
        hook = _in_mro(kind, "__getattribute__")
        own = type(hook) is types.FunctionType
        if not own and issubclass(kind, type):
            own = type(_in_mro(kind, "__getattr__")) is types.FunctionType
        own = _OWN_LOOKUP.put(kind, own)
    return own


_MISSING = object()


def has_attribute(obj: Any, name: str) -> bool:
    """``hasattr(obj, name)``, with any error the lookup raises taken as the
    attribute missing (`attribute_or`)."""
    return attribute_or(obj, name, _MISSING) is not _MISSING


def instance_of(obj: Any, kinds: type | tuple[type, ...]) -> bool:
    """``isinstance(obj, kinds)``, with any error the object's own lookups
    raise answered by its real type.

    When the type does not match, ``isinstance`` asks the object for its
    ``__class__`` (a proxy's way to pass for what it wraps), and so does an
    ABC's check. An object whose ``__getattribute__`` raises for it failed
    cash's own question and the cell with it, where plain Python never asks.
    An object with a ``__getattribute__`` of its own in Python is not asked
    at all (`own_lookup`): its real type answers, so a proxy written that
    way passes for itself, and none passes for a function, class, module
    or array by its ``__class__`` alone (`_passable`).
    """
    kind = type(obj)
    passable = _PASSABLE.get(kinds)
    if passable is None:
        passable = _passable(kinds)
    if passable[1]:
        # Kinds that answer ``isinstance`` as ``issubclass`` of ``__class__``.
        if issubclass(kind, kinds):
            return True
        plain = _PLAIN_CLASS.get(type_key(kind))
        if plain is None:
            plain = _plain_class(kind)
        if plain:
            # Its ``__class__`` is its type: ``isinstance`` would answer as above.
            return False
    elif own_lookup(kind):
        return issubclass(kind, kinds)
    try:
        # Not by its real type: by the ``__class__`` it answers, but never
        # for a kind whose parts cash reads off the object itself.
        if isinstance(obj, passable[0]):
            return True
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        pass
    if passable[1]:
        return False
    try:
        return issubclass(kind, kinds)
    except TypeError:  # a protocol with data members takes no subclass check
        return False


#: Per type, `_plain_class`'s answer, held weakly.
_PLAIN_CLASS: TypeMemo[bool] = TypeMemo()


def _plain_class(kind: type) -> bool:
    """Do *kind*'s instances answer their real type for ``__class__``, and
    look attributes up as Python does (`own_lookup`)? Then ``isinstance``
    is ``issubclass`` of the type, and the object is never asked."""
    return _PLAIN_CLASS.put(
        kind,
        own_lookup(kind) or not any("__class__" in _CLASS_DICT(klass) for klass in mro_of(kind) if klass is not object),
    )


def answers_its_type(kind: type) -> bool:
    """Do *kind*'s objects answer ``isinstance`` by their real type
    (`_plain_class`), never asked for a ``__class__`` of their own?"""
    plain = _PLAIN_CLASS.get(type_key(kind))
    return _plain_class(kind) if plain is None else plain


def _passable(kinds: type | tuple[type, ...]) -> tuple[tuple[type, ...], bool]:
    """*kinds* without the ones an object passes for only by being one: a
    function, method, code object, cell, frame, module, class or numpy
    array. cash reads their parts straight off the object (a function's
    ``__closure__``, a class's ``__qualname__``, an array's ``.base``), and
    one whose ``__class__`` answers ``FunctionType`` has none of them: the
    read failed the cell. A proxy that passes for one is read as the object
    it is. And whether every kind checks an instance as ``type`` and
    ``ABCMeta`` do, by its ``__class__``'s subclass check."""
    found = _PASSABLE.get(kinds)
    if found is None:
        if len(_PASSABLE) >= _PASSABLE_MAX:
            _PASSABLE.clear()
        np = loaded_module("numpy")
        layouts = (*_LAYOUT_KINDS, np.ndarray) if np is not None else _LAYOUT_KINDS
        every = _flat(kinds)
        found = (
            tuple(k for k in every if not issubclass(k, layouts)),
            all(type(k) in _PLAIN_METACLASSES for k in every),
        )
        _PASSABLE[kinds] = found
    return found


def _flat(kinds: Any) -> tuple[type, ...]:
    """*kinds* as ``isinstance`` reads them: tuples within tuples, and the
    members of a ``X | Y``, as one tuple of classes."""
    if isinstance(kinds, tuple):
        return tuple(kind for part in kinds for kind in _flat(part))
    if isinstance(kinds, types.UnionType):
        return _flat(kinds.__args__)
    return (kinds,)


_PLAIN_METACLASSES = (type, abc.ABCMeta)


#: Kinds whose parts cash reads off the object (`_passable`).
_LAYOUT_KINDS = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.CodeType,
    types.CellType,
    types.FrameType,
    types.ModuleType,
    type,
)
#: Per kinds, `_passable`'s answer: cash's own kinds, so a few dozen.
_PASSABLE: dict[Any, tuple[tuple[type, ...], bool]] = {}
_PASSABLE_MAX = 4096


def is_routine(obj: Any) -> bool:
    """``inspect.isroutine(obj)``, with any error the object's own lookups
    raise (`instance_of`) taken as not a routine: a function, method or
    builtin is not an object whose lookups raise.

    A ``functools.partial`` is not one: it holds the arguments its answer
    turns on, as a callable object does. Python 3.14 made it a method
    descriptor, so ``inspect.isroutine`` says yes there and no below, and a
    partial at a module's attribute was keyed by its type alone:
    ``partial(base, 1)`` served for ``partial(base, 2)``."""
    import functools
    import inspect

    kind = type(obj)
    if own_lookup(kind) or issubclass(kind, functools.partial):
        return False
    if hooked_class(kind):
        # The object's class's metaclass hooks its lookups: ``inspect``
        # would ask it for ``__get__``. Answered by the real type alone.
        return issubclass(kind, _ROUTINE_KINDS) or (
            _in_mro(kind, "__get__") is not _ABSENT and _in_mro(kind, "__set__") is _ABSENT
        )
    try:
        return inspect.isroutine(obj)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return False


#: A class's ``__dict__`` as ``type`` reads it: no metaclass of the user's runs.
_CLASS_DICT = type.__dict__["__dict__"].__get__
#: What Python's own ``__dict__`` slot is, on a class or a module.
_SLOT_READERS = (types.GetSetDescriptorType, types.MemberDescriptorType)
#: Per type, whether the ``__dict__`` its MRO finds first is the slot
#: Python gives an instance (`own_dict`), held weakly.
_DICT_READERS: TypeMemo[bool] = TypeMemo()


def has_dict_slot(kind: type) -> bool:
    """Is the ``__dict__`` *kind*'s MRO finds first the slot Python gives
    an instance (`own_dict`)? Read through ``type``'s own descriptors: a
    metaclass's hook is not asked."""
    slot = _DICT_READERS.get(type_key(kind))
    if slot is None:
        slot = False
        for klass in mro_of(kind):
            found = _CLASS_DICT(klass).get("__dict__")
            if found is not None:
                slot = type(found) in _SLOT_READERS
                break
        _DICT_READERS.put(kind, slot)
    return slot


def own_dicts(objs: list, kind: type) -> list[dict]:
    """The attribute dicts `own_dict` gives of *objs*, all of class *kind*,
    those it gives None for left out. Read a list at a time: a tree of a
    few hundred thousand small nodes is read in C, not a call per node."""
    if not objs or not has_dict_slot(kind):
        return []
    try:
        found = list(map(object.__getattribute__, objs, repeat("__dict__")))
    except Exception:  # noqa: BLE001 - one object's own slot: read one by one
        return [d for d in map(own_dict, objs) if d is not None]
    return list(compress(found, map(operator.is_, map(type, found), repeat(dict))))


def own_dict(obj: Any, subclasses: bool = False) -> dict | None:
    """The attribute dict Python keeps for *obj*, read without running code
    of the object's, or None when it has none.

    ``obj.__dict__`` runs a ``__dict__`` the class defines itself (a
    property computing it), and a lazy tree's makes a new child at each
    read: a walk over what an object holds read one more each time, and
    never ended, where plain Python never asks. The slot Python gives an
    instance is read instead; a class whose ``__dict__`` is its own, not
    that slot, has none that cash reads.

    A ``dict`` subclass set as the slot (``self.__dict__ = Attrs(...)``)
    is given only with *subclasses*: read it with ``dict``'s own methods
    (``dict.values(own)``), so none of the subclass's runs.
    """
    if not has_dict_slot(type(obj)):
        return None
    try:
        # The slot found first along the MRO is a data descriptor, so this
        # reads through it: not kept, as it refers to its class.
        attributes = object.__getattribute__(obj, "__dict__")
    except Exception:  # noqa: BLE001 - the object's own slot, not cash's error
        return None
    if type(attributes) is dict or (subclasses and isinstance(attributes, dict)):
        return attributes
    return None


#: The builtin containers a class of the user's can derive from.
_BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset, collections.deque)


def builtin_base(value: Any) -> type | None:
    """The builtin container *value*'s real type is or derives from, or None.

    Decided by the real type, not by ``isinstance``: a proxy
    (``wrapt.ObjectProxy({...})``, ``MagicMock(spec=dict)``, a class whose
    ``__class__`` says ``dict``) passes ``isinstance`` for what it claims to
    be but holds no storage of that type. cash does not open it as one; it
    is read as what it really is."""
    kind = type(value)
    if kind in _BUILTIN_CONTAINERS:
        return kind
    if not issubclass(kind, _BUILTIN_CONTAINERS):
        return None
    found = _DERIVED_BASE.get(type_key(kind))
    if found is not None and same_mro(found[0], kind):
        return found[1]
    return _derived_base(kind)


#: Per subclass, ``(its `mro_mark`, the builtin container it derives
#: from)``: read along its MRO once, not at every object a walk meets.
_DERIVED_BASE: TypeMemo[tuple[int, type]] = TypeMemo()


def _derived_base(kind: type) -> type:
    mro = mro_of(kind)
    base = next(c for c in mro if c in _BUILTIN_CONTAINERS)
    _DERIVED_BASE.put(kind, (mro_mark(kind), base))
    return base


def builtin_items(value: Any) -> list[Any] | None:
    """What the builtin container *value* is or derives from holds: a dict's
    keys then values, the items of a list, tuple, set, frozenset or deque;
    None when its real type derives from none of them (`builtin_base`).

    Read through the builtin type's own methods, not the class's: a dict
    subclass whose ``values()`` raises, or a list subclass whose
    ``__iter__`` does, failed cash's look into it, and the cell with it,
    where plain Python never asked."""
    base = type(value)
    if base not in _BUILTIN_CONTAINERS:
        base = builtin_base(value)
        if base is None:
            return None
    if base is dict:
        return [*dict.keys(value), *dict.values(value)]
    return list(base.__iter__(value))


_OBJECT_LOOKUP = object.__dict__["__getattribute__"]
_ROUTINE_KINDS = (
    types.BuiltinFunctionType,
    types.FunctionType,
    types.MethodType,
    types.MethodWrapperType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
)

#: `held_attribute`'s answer for an attribute reading would compute.
NOT_HELD: Any = object()
_TUPLE_GETTER = type(collections.namedtuple("_Probe", "a").__dict__["a"])


def held_attribute(obj: Any, name: str) -> Any:
    """*obj*'s attribute *name* when reading it runs no code: a value in
    its ``__dict__`` no descriptor of its class overrides, a slot, a
    ``NamedTuple`` field. `NOT_HELD` for anything else: a property, a
    ``cached_property`` (filled or not) or other descriptor, a class
    attribute the instance does not shadow, a class with its own
    ``__getattribute__``, a missing attribute (``__getattr__``'s)."""
    cls = type(obj)
    if _in_mro(cls, "__getattribute__") is not _OBJECT_LOOKUP and not isinstance(obj, tuple):
        return NOT_HELD
    found = NOT_HELD
    for klass in class_mro(cls):
        found = class_dict(klass).get(name, NOT_HELD)
        if found is not NOT_HELD:
            break
    if isinstance(found, functools.cached_property):
        # Filled or not depends on what ran before: read before a call and
        # after it, one chain would stop at two places. The object it is
        # cached on is keyed whole, the same either way.
        return NOT_HELD
    if found is not NOT_HELD and hasattr(type(found), "__set__"):
        # A data descriptor wins over the instance: only a slot and a
        # NamedTuple field are plain.
        if type(found) is types.MemberDescriptorType or (type(found) is _TUPLE_GETTER and isinstance(obj, tuple)):
            try:
                return found.__get__(obj, cls)
            except (AttributeError, IndexError):
                return NOT_HELD
        return NOT_HELD
    try:
        state = object.__getattribute__(obj, "__dict__")
    except (AttributeError, TypeError):
        state = None
    if type(state) is dict and name in state:
        return state[name]
    if found is NOT_HELD:
        return _pydantic_private(obj, cls, name)
    return NOT_HELD


def _pydantic_private(obj: Any, cls: type, name: str) -> Any:
    """A pydantic model's private attribute (``PrivateAttr``), read where
    pydantic's own ``BaseModel.__getattr__`` reads it: the model's
    ``__pydantic_private__`` dict, for a name in its class's
    ``__private_attributes__`` whose default is no descriptor. Only for a
    class whose ``__getattr__`` is pydantic's own, so the read runs no
    other code; `NOT_HELD` for anything else."""
    pydantic = loaded_module("pydantic")
    base = getattr(pydantic, "BaseModel", None) if pydantic is not None else None
    if base is None or not isinstance(obj, base):
        return NOT_HELD
    getattr_ = _in_mro(cls, "__getattr__")
    if getattr_ is _ABSENT or getattr_ is not class_dict(base).get("__getattr__"):
        return NOT_HELD
    try:
        declared = object.__getattribute__(obj, "__private_attributes__")
        private = object.__getattribute__(obj, "__pydantic_private__")
    except AttributeError:
        return NOT_HELD
    if type(declared) is not dict or name not in declared or hasattr(declared[name], "__get__"):
        return NOT_HELD
    if type(private) is not dict or name not in private:
        return NOT_HELD
    return private[name]
