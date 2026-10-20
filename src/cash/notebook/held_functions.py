"""The notebook functions a value holds, at any depth, so a statement that
reaches one through a container keys the globals it reads at the call.

``fns = [make]`` above ``r = fns[0](3)``: the statement names ``fns``, a
list, and a call key walks only into what an input names as a function
(``cache_key.called_function_dependencies``). ``make`` reads ``cfg`` when it
runs, so an edited ``cfg`` must re-key ``r``; without a notebook file to
simulate there was nothing else to move it, and the old result was served.
The same holds for a dict of builders, a ``functools.partial``, a lambda in
a dict, a function kept as another's default or in its closure, a class
attribute, and an object of a notebook class whose methods read globals.

A walk that meets more objects than `WALK_LIMIT` stops and says so
(`Held.complete`): what it did not see may hold a function, so the
statement runs uncached, with a reason (``statement.processor``), rather
than be keyed without it.

The walk also notes the classes of the user's own modules it meets
(`Held.module_classes`): ``orders = pickle.load(f)`` holding objects of
``mylib.Order`` runs ``Order.total`` and the module's ``rate()`` when a
statement calls ``o.total()``, though no lineage of ``orders`` passes
through ``mylib``. The key names that code (``cache_key``).
"""

from __future__ import annotations

import collections
import functools
import gc
import itertools
import operator
import sys
import types
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, NamedTuple

from cash._user_attrs import attribute_or

from .. import _native
from .._memo import CODE_OBJECTS, LruMemo
from ..install_paths import is_user_code_module
from ..tracking.function_tracker import is_local_module
from ..value_types import LEAF_TYPES, library_scalar_types
from .compiled_source import is_cash_filename

__all__ = [
    "EXISTENCE_CHECK_AT",
    "WALK_LIMIT",
    "Held",
    "held_notebook_code",
    "module_class_objects_exist",
    "notebook_functions_exist",
]

#: Objects one walk visits before it gives up. A model, a figure, a frame
#: and their containers hold far fewer that are not plain values (a list of
#: numbers is gathered at C speed and stops the walk).
WALK_LIMIT = 2_000_000

#: Objects a walk visits before it asks whether a notebook function exists
#: at all (`notebook_functions_exist`). Asking looks in the namespace, then
#: reads every object Python tracks once, at C speed (0.06 s for two
#: million); a walk of an openpyxl workbook of 400,000 cells took 0.6 s,
#: before each statement that read it, and found nothing a cell could have
#: defined in it.
EXISTENCE_CHECK_AT = 20_000

_LEAVES = frozenset(LEAF_TYPES)


@functools.lru_cache(maxsize=4)
def _leaves_with(scalars: tuple[type, ...]) -> frozenset[type]:
    return _LEAVES.union(scalars)


def _leaves() -> frozenset[type]:
    """`LEAF_TYPES` and numpy's and pandas' scalars (`library_scalar_types`):
    values that hold no function. Parsed rows holding a ``pd.Timestamp``
    were walked a timestamp at a time, for every statement reading them."""
    return _leaves_with(library_scalar_types())


_SEQUENCES = (list, tuple, set, frozenset)
_PLAIN = frozenset((dict, list, tuple))
_MAIN_NAMES = frozenset(("__main__", "builtins"))
#: ``Py_TPFLAGS_HEAPTYPE``: set on every class a ``class`` statement makes.
_HEAP_TYPE = 1 << 9
#: The C walk kernels (`cash._native.kernels`), or None: each step they take
#: gives what the Python step beside it gives, and the Python step runs
#: when they are missing or say NotImplemented (a subclass, a metaclass of
#: its own). A walk of a 350,000-edge graph took 1.2 s in Python.
_K = _native.kernels

#: What ``gc.freeze`` had moved out of the collector's view when cash was
#: imported. Python 3.12 freezes some 375 tuples of its own while it starts,
#: so a nonzero count says nothing of the user's objects by itself: only
#: more than this does. ``gc.freeze`` moves every tracked object, far more
#: than this many, so one that hides a function always counts above it.
_FROZEN_AT_IMPORT = gc.get_freeze_count()


def _objects_frozen_since_import() -> bool:
    """Has ``gc.freeze`` hidden objects the collector's view no longer
    shows, beyond the interpreter's own?"""
    return gc.get_freeze_count() > _FROZEN_AT_IMPORT


class Held(NamedTuple):
    """What `held_notebook_code` found."""

    #: The code of each notebook function held, a notebook class's methods
    #: included, in the order first met.
    codes: tuple[types.CodeType, ...]
    #: False when the walk stopped at `WALK_LIMIT` before it saw everything.
    complete: bool = True
    #: The classes of the user's own modules (not the notebook) along the
    #: MRO of each class held or of an object held, in the order first met:
    #: their methods, and what those read in their module, run when the
    #: statement calls one.
    module_classes: tuple[type, ...] = ()


_NOTHING = Held(())


class _NoNotebookFunctions(Exception):
    """A walk stopped early: no notebook function, and no object of a
    class of the user's modules, exists to be found."""


#: What `_MEMO` keeps for a walk that stopped because no notebook function
#: existed: the answer holds only while that is still so.
_NONE_EXISTED = ((), (), True)

#: ``(id(value), lineage)`` -> the functions and notebook classes a walk
#: found, and whether it saw everything. The lineage moves whenever a cell
#: rebinds or changes the value, so a container walked once is not walked
#: again for every statement reading it. A class's methods are read again
#: each time: ``M.build = other`` changes what an object of ``M`` runs.
_MEMO: LruMemo[tuple[int, str], tuple[tuple[Any, ...], tuple[type, ...], bool]] = LruMemo(CODE_OBJECTS)


def held_notebook_code(value: Any, namespace: Mapping[str, Any], lineage: str | None = None) -> Held:
    """The code of the notebook functions *value* holds, and of the methods
    of the notebook classes it holds or is an object of.

    A notebook function is one whose globals are *namespace*. Followed: the
    items (and non-string keys) of a list, tuple, set or dict; a function's
    defaults, closure and attributes (``__wrapped__``); a bound method's
    function and object; a partial's function and arguments; a static or
    class method; a property's accessors; a notebook class's attributes
    (along its MRO); and the ``__dict__`` and slots of any other object, a
    library's included -- a ``FunctionTransformer(make)`` calls ``make``.
    Not entered: modules, library classes, and what holds no attributes (a
    numpy array).

    *lineage* is the value's lineage, when it has one: the walk is then
    remembered for it.
    """
    if value is None or type(value) in _leaves() or isinstance(value, types.ModuleType):
        return _NOTHING
    if type(value) is types.FunctionType and _plain_function(value):
        return Held((value.__code__,)) if _is_notebook(value, namespace) else _NOTHING
    memo_key = (id(value), lineage) if lineage else None
    found = _MEMO.get(memo_key) if memo_key is not None else None
    if found is _NONE_EXISTED:
        if not _anything_to_find(namespace):
            return _NOTHING
        found = None  # one exists now: walk to the end
    if found is None:
        try:
            found = _walk(value, namespace, check_at=EXISTENCE_CHECK_AT)
        except _NoNotebookFunctions:
            if memo_key is not None:
                _MEMO[memo_key] = _NONE_EXISTED
            return _NOTHING
        if memo_key is not None:
            _MEMO[memo_key] = found
    functions, classes, complete = found
    codes: dict[int, types.CodeType] = {}
    for fn in functions:
        code = attribute_or(fn, "__code__", None)
        if isinstance(code, types.CodeType):
            codes.setdefault(id(code), code)
    module_classes: dict[int, type] = {}
    for cls in classes:
        for code in _class_codes(cls, namespace):
            codes.setdefault(id(code), code)
        for klass in cls.__mro__:
            if _of_a_users_module(klass):
                module_classes.setdefault(id(klass), klass)
    return Held(tuple(codes.values()), complete, tuple(module_classes.values()))


def _plain_function(fn: types.FunctionType) -> bool:
    """A function that holds nothing more to walk: no defaults, no closure,
    no attributes. Most notebook functions; answered without a walk."""
    return not fn.__defaults__ and not fn.__kwdefaults__ and not fn.__closure__ and not fn.__dict__


def _globals_ids(namespace: Mapping[str, Any]) -> frozenset[int]:
    """The ids of the dicts a notebook function's ``__globals__`` can be:
    *namespace*, or each map of a ``ChainMap`` over it."""
    maps = attribute_or(namespace, "maps", None)
    if isinstance(maps, list):
        return frozenset(id(m) for m in maps)
    return frozenset((id(namespace),))


def _is_notebook(fn: Any, namespace: Mapping[str, Any]) -> bool:
    return id(attribute_or(fn, "__globals__", None)) in _globals_ids(namespace)


def notebook_functions_exist(namespace: Mapping[str, Any]) -> bool:
    """Whether any function whose globals are *namespace* exists, anywhere.

    Every notebook function, a method of a notebook class or a lambda kept
    in a list included, references its globals, and Python tracks every
    function for its garbage collector: ``gc.get_referrers`` of the
    namespace finds them all. When there is none, no value holds one, and
    no class's method reads the notebook's globals. Objects ``gc.freeze``
    moved out of the collector's view are not searched, so then the answer
    is yes.

    Asked of all functions at once, not of each: that nothing but the
    namespace refers to ``make`` does not mean no value holds it. A value
    holding the namespace (``{"ns": globals()}``), the shell
    (``get_ipython().user_ns``) or anything of the kernel's that leads to
    it holds every function there; telling which values do means following
    what refers to the namespace back to them -- the shell, its completer,
    the 140 objects that refer to the shell, and on: 1,300 objects and five
    seconds of ``gc.get_referrers`` in a plain IPython shell, more than the
    walk it would spare.

    The statement cash is running is not one (`_running_statement`): since
    Python 3.11 ``exec`` makes a function of the code it runs, which no
    value can hold. Counted, it said yes to every call keyed inside a
    statement, and each walked a whole networkx graph to its end (1.1 s).

    A no is remembered (`_NO_FUNCTIONS`) until code could have made one:
    a function is made only by code running with these globals, which
    starts with an ``exec``, or by ``types.FunctionType``, both seen by an
    audit hook (`_note_new_code`). A no is neither taken nor kept while
    such code is on the stack (a call keyed inside a running statement,
    which may define a lambda after the question). A statement's checks
    asked it up to 13 times, 0.035-0.07 s each beside a big graph.
    """
    if _objects_frozen_since_import():
        return True
    maps = attribute_or(namespace, "maps", None)
    dicts = list(maps) if isinstance(maps, list) else [namespace]
    globals_ids = _globals_ids(namespace)
    running = _code_running_in(globals_ids)
    if not running and _NO_FUNCTIONS.get(globals_ids) == _new_code():
        return False
    seen_at = _new_code()
    function = types.FunctionType
    # A ``def`` binds its function in the namespace: found there, the scan of
    # every tracked object (0.12 s beside a 400,000-cell workbook, at each
    # walk that asked) is not needed. Only a yes is taken from it.
    for mapping in dicts:
        for value in list(mapping.values()):
            if type(value) is function and id(value.__globals__) in globals_ids:
                return True
    for mapping in dicts:
        for referrer in gc.get_referrers(mapping):
            if (
                type(referrer) is function
                and id(referrer.__globals__) in globals_ids
                and not _running_statement(referrer)
            ):
                return True
    if not running:
        if len(_NO_FUNCTIONS) > 64:
            _NO_FUNCTIONS.clear()
        _NO_FUNCTIONS[globals_ids] = seen_at
    return False


def _anything_to_find(namespace: Mapping[str, Any]) -> bool:
    """Whether a walk could find anything: a notebook function, or an
    object of a class of the user's modules (`module_class_objects_exist`)."""
    return notebook_functions_exist(namespace) or module_class_objects_exist()


def module_class_objects_exist() -> bool:
    """Whether any object of a class defined at the top of one of the
    user's loaded modules, or of a class derived from one, exists.

    An object refers to its class, so ``gc.get_referrers`` of the classes
    finds them all, in one pass at C speed. With no module of the user's
    loaded there is nothing to ask. Objects ``gc.freeze`` moved out of the
    collector's view are not searched, so then the answer is yes.
    """
    classes = _users_module_classes()
    if not classes:
        return False
    if _objects_frozen_since_import():
        return True
    kinds: set[type] = set()
    pending = list(classes)
    while pending:
        cls = pending.pop()
        if cls in kinds:
            continue
        kinds.add(cls)
        try:
            pending.extend(type.__subclasses__(cls))
        except TypeError:
            pass
    return any(type(referrer) in kinds for referrer in gc.get_referrers(*kinds))


def _users_module_classes() -> list[type]:
    """The classes defined at the top of the user's loaded modules."""
    found: list[type] = []
    for module in list(sys.modules.values()):
        if not _users_module(module):
            continue
        name = module.__name__
        for value in list(vars(module).values()):
            if isinstance(value, type) and value.__flags__ & _HEAP_TYPE and value.__module__ == name:
                found.append(value)
    return found


#: Module name -> (the module, whether it is one of the user's), asked once
#: per module object.
_USERS_MODULE: dict[str, tuple[types.ModuleType, bool]] = {}


def _users_module(module: Any) -> bool:
    """Is *module* one of the user's own files (`is_local_module`), not the
    notebook's ``__main__``?"""
    if not isinstance(module, types.ModuleType):
        return False
    name = attribute_or(module, "__name__", None)
    if not isinstance(name, str) or name in _MAIN_NAMES:
        return False
    held = _USERS_MODULE.get(name)
    if held is not None and held[0] is module:
        return held[1]
    try:
        verdict = is_local_module(module)
    except (TypeError, AttributeError):
        verdict = False
    _USERS_MODULE[name] = (module, verdict)
    return verdict


#: Class -> whether a module of the user's defined it (`_of_a_users_module`).
_USERS_CLASS: weakref.WeakKeyDictionary[type, bool] = weakref.WeakKeyDictionary()


def _of_a_users_module(cls: type) -> bool:
    """Was *cls* defined by a ``class`` statement in one of the user's own
    modules (not a cell)?"""
    try:
        known = _USERS_CLASS.get(cls)
    except TypeError:
        return False
    if known is not None:
        return known
    verdict = bool(attribute_or(cls, "__flags__", 0) & _HEAP_TYPE) and _users_module(
        sys.modules.get(attribute_or(cls, "__module__", None) or "")
    )
    try:
        _USERS_CLASS[cls] = verdict
    except TypeError:
        pass
    return verdict


#: The ids of the globals a `notebook_functions_exist` no was answered for,
#: and the `_new_code` count it was answered at.
_NO_FUNCTIONS: dict[frozenset[int], int] = {}
_NEW_CODE_EVENTS = frozenset(("exec", "function.__new__"))
#: How many times code started running from an ``exec`` or a function was
#: made from code (``types.FunctionType``) in this process: counted by the C
#: extension's audit hook (``caudit.c``) when it is in, else by a Python one,
#: which Python calls for every audited event (an ``id()`` included).
_counted = [0]


def _note_new_code(event: str, _args: tuple) -> None:
    if event in _NEW_CODE_EVENTS:
        _counted[0] += 1


if _K is not None and _K.audit_hook_install():
    _new_code = _K.audit_new_code
else:
    sys.addaudithook(_note_new_code)

    def _new_code() -> int:
        return _counted[0]


def _code_running_in(globals_ids: frozenset[int]) -> bool:
    """Is a frame on this thread's stack running with one of these globals?"""
    frame = sys._getframe(1)
    while frame is not None:
        if id(frame.f_globals) in globals_ids:
            return True
        frame = frame.f_back
    return False


def _running_statement(fn: types.FunctionType) -> bool:
    """Is *fn* the function ``exec`` made of a statement cash runs (its
    module code, compiled under cash's file name)?"""
    code = fn.__code__
    return code.co_name == "<module>" and is_cash_filename(code.co_filename)


def _walk(
    value: Any, namespace: Mapping[str, Any], check_at: int | None = None
) -> tuple[tuple[Any, ...], tuple[type, ...], bool]:
    """Breadth first from *value*: ``(functions, classes, complete)``.

    With *check_at*, once that many objects were visited it asks whether a
    notebook function exists at all (`notebook_functions_exist`) and raises
    `_NoNotebookFunctions` when none does: the rest of the walk could find
    nothing."""
    globals_ids = _globals_ids(namespace)
    functions: dict[int, Any] = {}
    classes: dict[int, type] = {}
    #: What the walk met, each by `_identity_keys`.
    seen: set[Any] = set(_identity_keys([value], {type(value)}))
    #: The objects to visit next, by exact type.
    level: list[tuple[type, list[Any]]] = [(type(value), [value])]
    visited = 0
    while level:
        visited += sum(len(group) for _kind, group in level)
        if check_at is not None and visited > check_at:
            check_at = None
            if not _anything_to_find(namespace):
                raise _NoNotebookFunctions
        if visited > WALK_LIMIT:
            return tuple(functions.values()), tuple(classes.values()), False
        children: list[Any] = []
        # One step per TYPE in the level, not per object: the items of a
        # list of 100,000 records are gathered at C speed.
        for kind, group in level:
            _visit_group(kind, group, globals_ids, functions, classes, children)
        if not children:
            break
        if check_at is not None and visited + len(children) > check_at:
            # Asked before the next level is sorted: a sheet's 400,000
            # cells cost more to dedup than the question does.
            check_at = None
            if not _anything_to_find(namespace):
                raise _NoNotebookFunctions
        level = _next_level(children, seen)
    return tuple(functions.values()), tuple(classes.values()), True


def _next_level(children: list[Any], seen: set[Any]) -> list[tuple[type, list[Any]]]:
    """What of *children* the walk visits next, by exact type, each object
    once and none it met before (*seen*, which this adds them to)."""
    kernels = _K
    child_kinds = kernels.type_set(children) if kernels is not None else set(map(type, children))
    # An object that leads only to its class (`_inert`: an openpyxl cell's
    # style array, 400,000 of them) is not visited: its class is.
    inert = {kind for kind in child_kinds if _inert(kind)}
    skipped = _leaves() | inert if inert else _leaves()
    groups = kernels.split_level(children, skipped, inert) if kernels is not None else NotImplemented
    if groups is NotImplemented:
        kinds_of = list(map(type, children))
        others = list(itertools.compress(children, map(operator.not_, map(skipped.__contains__, kinds_of))))
        others.extend(inert)
        # Split by type once, without another pass when one type is left.
        groups = _by_type(others, (child_kinds - skipped) | set(map(type, inert)))
    level = []
    for kind, group in groups:
        if kind in _PLAIN and _plain_records(kind, group):
            continue
        if kernels is not None:
            new = kernels.new_in(group, _identity_hashed(kind), seen)
            if new:
                level.append((kind, new))
            continue
        # Dedup by the objects themselves where that is identity: an
        # openpyxl sheet's 400,000 cells were each asked for their ``id``.
        found = dict(zip(group if _identity_hashed(kind) else map(id, group), group))
        for key in seen & found.keys():
            del found[key]
        if found:
            seen.update(found)
            level.append((kind, list(found.values())))
    return level


def _plain_records(kind: type, group: list[Any]) -> bool:
    """Whether every one of *group*, plain dicts, lists or tuples of type
    *kind*, holds only plain values: a million records read as rows are
    dropped in one pass at C speed, before the walk dedups them (most of
    what a walk over them cost)."""
    if _K is not None and type(group) is list:
        found = _K.leaf_items(group, 2 if kind is dict else 0, _leaves())
        if found is not NotImplemented:
            return found
    if kind is dict:
        items = itertools.chain(
            itertools.chain.from_iterable(group), itertools.chain.from_iterable(map(dict.values, group))
        )
    else:
        items = itertools.chain.from_iterable(group)
    return set(map(type, items)) <= _leaves()


def _visit_group(
    kind: type,
    group: list[Any],
    globals_ids: frozenset[int],
    functions: dict[int, Any],
    classes: dict[int, type],
    children: list[Any],
) -> None:
    """Note the notebook functions and classes in *group*, objects of type
    *kind*, and put what they lead to in *children*."""
    if issubclass(kind, types.ModuleType):
        return
    if _K is not None and type(group) is list:
        # The two branches below for exact dicts and sequences, in C.
        if kind is dict:
            if _K.dict_children(group, _leaves(), children) is not NotImplemented:
                return
        elif kind in _SEQUENCES and _K.seq_children(group, _leaves(), children) is not NotImplemented:
            return
    if issubclass(kind, dict):
        keys = list(itertools.chain.from_iterable(group))
        key_kinds = set(map(type, keys))
        # Keys that are tuples of plain values (an openpyxl sheet's cells by
        # ``(row, column)``) lead nowhere: one pass says so, as for a list.
        if not key_kinds <= _leaves() and not (
            key_kinds <= _leaves() | {tuple}
            and set(map(type, itertools.chain.from_iterable(_of_type(keys, tuple)))) <= _leaves()
        ):
            children.extend(keys)
        if not set(map(type, itertools.chain.from_iterable(map(dict.values, group)))) <= _leaves():
            children.extend(itertools.chain.from_iterable(map(dict.values, group)))
        if kind is dict:
            return
    elif issubclass(kind, _SEQUENCES):
        # A list of numbers or strings leads nowhere: one pass at C speed
        # says so, and adds none of its items.
        if not set(map(type, itertools.chain.from_iterable(group))) <= _leaves():
            children.extend(itertools.chain.from_iterable(group))
        if kind in _SEQUENCES:
            return
    elif kind is types.FunctionType:
        for fn in group:
            if id(fn.__globals__) in globals_ids:
                functions.setdefault(id(fn), fn)
            children.extend(fn.__defaults__ or ())
            children.extend((fn.__kwdefaults__ or {}).values())
            children.extend(_cell_contents(fn.__closure__))
            children.extend(fn.__dict__.values())
        return
    elif kind is types.MethodType:
        for method in group:
            children.extend((method.__func__, method.__self__))
        return
    elif issubclass(kind, functools.partial):
        for part in group:
            children.append(part.func)
            children.extend(part.args)
            children.extend(part.keywords.values())
        if kind is functools.partial:
            return
    elif kind in (staticmethod, classmethod):
        children.extend(member.__func__ for member in group)
        return
    elif kind is property:
        for prop in group:
            children.extend(f for f in (prop.fget, prop.fset, prop.fdel) if f is not None)
        return
    elif issubclass(kind, type):
        for cls in group:
            if _notebook_class(cls, globals_ids):
                classes.setdefault(id(cls), cls)
                for klass in _own_mro(cls, globals_ids):
                    children.extend(vars(klass).values())
            elif any(map(_of_a_users_module, cls.__mro__)):
                # Noted, not entered: its methods' code is keyed by its
                # module (`Held.module_classes`).
                classes.setdefault(id(cls), cls)
        return
    # Any other object: its class (a notebook class's methods run on it),
    # its attributes and its slots.
    children.append(kind)
    if _may_have_dict(kind):
        for obj in group:
            try:
                attrs = object.__getattribute__(obj, "__dict__")
            except (AttributeError, TypeError):
                attrs = None
            if isinstance(attrs, dict):
                children.extend(attrs.values())
    for name in _slot_names(kind):
        getter = _slot_getter(kind, name)
        if getter is not None:
            try:
                # What ``object.__getattribute__`` reads, at C speed: the
                # slot's own descriptor (400,000 openpyxl cells, 8 slots each).
                column = list(map(getter, group))
            except (AttributeError, TypeError):
                pass  # a slot not set on some object: one at a time
            else:
                _add_column(column, children)
                continue
        for obj in group:
            try:
                children.append(object.__getattribute__(obj, name))
            except (AttributeError, TypeError):
                pass


def _add_column(column: list[Any], children: list[Any]) -> None:
    """Put what one slot of a group of objects holds in *children*, at C
    speed: nothing for a column of plain values (each cell's value), the
    class alone for a column of `_inert` objects (each cell's style array),
    each object once (the sheet that 400,000 cells name as their parent)."""
    kinds = set(map(type, column))
    if kinds <= _leaves():
        return
    if all(map(_inert, kinds - _leaves())):
        children.extend(kinds - _leaves())
        return
    if kinds & _leaves():
        column = list(itertools.compress(column, map(operator.not_, map(_leaves().__contains__, map(type, column)))))
        kinds -= _leaves()
    children.extend(dict(zip(_identity_keys(column, kinds), column)).values())


def _identity_keys(objects: list[Any], kinds: set[type]) -> Iterable[Any]:
    """A key for each of *objects* (of the types *kinds*) that no other
    live object shares: the object itself when its type hashes and compares
    by identity (`_identity_hashed`), its ``id`` otherwise.

    A dict keyed by them dedups the objects at C speed: an openpyxl sheet's
    400,000 cells were each asked for their ``id`` -- a new int, and an
    audit event the interpreter hands every audit hook (cash's own included)
    -- twice a walk. An ``id`` and an object never compare equal, so the two
    kinds of key mix in one set."""
    by_identity = {kind for kind in kinds if _identity_hashed(kind)}
    if len(by_identity) == len(kinds):
        return objects
    if not by_identity:
        return map(id, objects)
    return [obj if type(obj) in by_identity else id(obj) for obj in objects]


@functools.lru_cache(maxsize=1024)
def _identity_hashed(kind: type) -> bool:
    """Whether an object of *kind* hashes and compares as ``object`` does:
    by its address, with no code of its class run. Two distinct live
    objects of such kinds never hash alike (the hash is the address,
    rotated), so a set of them never compares two.

    Read from the classes' own dicts, as `_slot_getter` does: looking
    ``__hash__`` up on a ``MagicMock``'s class made a new mock, of a new
    class, which the walk then met -- without end."""
    try:
        for name in ("__hash__", "__eq__"):
            found = next((vars(klass)[name] for klass in kind.__mro__ if name in vars(klass)), None)
            if found is not _OBJECT_SLOTS[name]:
                return False
    except TypeError:
        return False
    return True


_OBJECT_SLOTS = {name: vars(object)[name] for name in ("__hash__", "__eq__")}


def _of_type(objects: list[Any], kind: type) -> Iterator[Any]:
    """The objects of exact type *kind* in *objects*, at C speed."""
    return itertools.compress(objects, map(operator.is_, map(type, objects), itertools.repeat(kind)))


#: A type with more than 1/`_MAJOR` of a level's objects is split out of it
#: in a pass at C speed of its own; the rest are sorted one at a time.
_MAJOR = 16


def _by_type(objects: list[Any], kinds: set[type] | None = None) -> list[tuple[type, list[Any]]]:
    """*objects* grouped by exact type, each group in its order in *objects*.
    The types most of them have are split out at C speed (a level of
    400,000 openpyxl cells and as many index tuples was split by a
    comprehension per type), the few others one at a time. *kinds*, when
    given, holds the types of *objects*, or more: one alone needs no pass."""
    if kinds is not None and len(kinds) == 1:
        return [(next(iter(kinds)), objects)]
    types_of = list(map(type, objects))
    counts = collections.Counter(types_of)
    if len(counts) == 1:
        return [(next(iter(counts)), objects)]
    major = [kind for kind, n in counts.items() if n * _MAJOR > len(objects)]
    groups = {
        kind: list(itertools.compress(objects, map(operator.is_, types_of, itertools.repeat(kind)))) for kind in major
    }
    if len(major) < len(counts):
        majors = frozenset(major)
        for obj in itertools.compress(objects, map(operator.not_, map(majors.__contains__, types_of))):
            groups.setdefault(type(obj), []).append(obj)
    return list(groups.items())


#: What `_visit_group` walks into other than an object's attributes.
_WALKED_KINDS = (
    types.ModuleType,
    dict,
    *_SEQUENCES,
    types.FunctionType,
    types.MethodType,
    functools.partial,
    staticmethod,
    classmethod,
    property,
    type,
)


@functools.lru_cache(maxsize=1024)
def _inert(kind: type) -> bool:
    """An object of *kind* leads `_visit_group` to its class alone: no
    instance dict, no slots, and not a kind it walks into otherwise."""
    if kind in _leaves() or issubclass(kind, _WALKED_KINDS):
        return False
    return not _may_have_dict(kind) and not _slot_names(kind)


@functools.lru_cache(maxsize=1024)
def _may_have_dict(kind: type) -> bool:
    """Can ``object.__getattribute__(obj, "__dict__")`` find anything on an
    object of *kind*: it has an instance dict, or a class along its MRO
    defines ``__dict__``. An openpyxl cell has slots only, and asking each of
    400,000 for its ``__dict__`` raised as many AttributeErrors."""
    if attribute_or(kind, "__dictoffset__", 1):
        return True
    try:
        return any("__dict__" in vars(klass) for klass in kind.__mro__)
    except TypeError:
        return True


@functools.lru_cache(maxsize=1024)
def _slot_getter(kind: type, name: str) -> Callable[[Any], Any] | None:
    """The ``__get__`` of the slot descriptor ``object.__getattribute__``
    finds for *name* on an object of *kind* (the first class along the MRO
    that defines *name*), or None when that is not a plain slot."""
    try:
        for klass in kind.__mro__:
            found = vars(klass).get(name, _MISSING)
            if found is not _MISSING:
                return found.__get__ if type(found) is types.MemberDescriptorType else None
    except TypeError:
        return None
    return None


_MISSING = object()


def _cell_contents(cells: tuple[Any, ...] | None) -> list[Any]:
    found = []
    for cell in cells or ():
        try:
            found.append(cell.cell_contents)
        except ValueError:  # an empty cell
            pass
    return found


@functools.lru_cache(maxsize=1024)
def _slot_names(kind: type) -> tuple[str, ...]:
    """The slots *kind* and its bases declare, but ``__dict__`` and ``__weakref__``."""
    names: list[str] = []
    for klass in kind.__mro__:
        slots = vars(klass).get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        try:
            names.extend(
                _mangled(klass, s) for s in slots if isinstance(s, str) and s not in ("__dict__", "__weakref__")
            )
        except TypeError:
            continue
    return tuple(dict.fromkeys(names))


def _mangled(klass: type, name: str) -> str:
    """The attribute a slot *name* of *klass* is stored under: a private
    ``__x`` is ``_Klass__x``, as for any private name in a class body (read
    by its own name, it was not found and its contents were not walked)."""
    owner = klass.__name__.lstrip("_")
    if name.startswith("__") and not name.endswith("__") and owner:
        return f"_{owner}{name}"
    return name


def _member_functions(member: Any) -> list[Any]:
    """The functions a class attribute runs: a method, a static or class
    method, a property's accessors."""
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        return [f for f in (member.fget, member.fset, member.fdel) if isinstance(f, types.FunctionType)]
    return [member] if isinstance(member, types.FunctionType) else []


def _defined_in(klass: type, globals_ids: frozenset[int]) -> bool:
    """Whether a cell defined *klass*: its module is the notebook's
    (``__main__``), or one of its own methods has the notebook's globals."""
    try:
        members = list(vars(klass).values())
    except TypeError:
        return False
    if any(id(fn.__globals__) in globals_ids for member in members for fn in _member_functions(member)):
        return True
    module = sys.modules.get(attribute_or(klass, "__module__", None) or "")
    return module is not None and id(attribute_or(module, "__dict__", None)) in globals_ids


def _own_mro(cls: type, globals_ids: frozenset[int]) -> list[type]:
    """The classes in *cls*'s MRO a cell defined."""
    return [klass for klass in cls.__mro__ if klass is not object and _defined_in(klass, globals_ids)]


def _notebook_class(cls: type, globals_ids: frozenset[int]) -> bool:
    """Whether a cell defined *cls* or a class it derives from. A class of a
    library is told at once by its module: its 500 members are not read."""
    if not cls.__flags__ & _HEAP_TYPE:
        return False  # a type built into the interpreter or an extension
    name = attribute_or(cls, "__module__", None) or ""
    module = sys.modules.get(name)
    # ``builtins`` too: a class a cell ``exec``s with no ``__name__`` in
    # its namespace says it is from there.
    if name not in _MAIN_NAMES and module is not None and id(attribute_or(module, "__dict__", None)) not in globals_ids:
        if not is_user_code_module(module):
            return False
    return bool(_own_mro(cls, globals_ids))


def _class_codes(cls: type, namespace: Mapping[str, Any]) -> list[types.CodeType]:
    """The code of the methods the notebook's classes in *cls*'s MRO hold now."""
    globals_ids = _globals_ids(namespace)
    codes = []
    for klass in _own_mro(cls, globals_ids):
        for member in list(vars(klass).values()):
            for fn in _member_functions(member):
                if id(fn.__globals__) in globals_ids:
                    codes.append(fn.__code__)
    return codes
