"""The data a user class brings to a key: what it holds at class level, and
what the functions an instance can run read from their modules."""

from __future__ import annotations

import contextvars
import enum
import hashlib
import pickle
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._memo import CODE_OBJECTS, LruMemo
from ..exceptions import CashImpurityWarning
from .arg_hashing import is_opaque
from .call_state import CAPTURE_WATCH
from .closure_fold import iter_code_scopes
from .global_reads import DOCSTRING_READS, MACHINERY_DUNDERS, bytecode_written_attrs
from .global_values import UNHASHABLE_GLOBAL_FIX
from .key_values import SYNC_TYPES, is_immutable_capture, stabilize_for_global_hash
from .user_code import is_cash_wrapper, is_user_class, is_user_code_object, own_package, wraps_code

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher
    from .global_reads import GlobalReads
    from .global_values import GlobalValues
    from .purity_checks import LearnedMutations
    from .registry import FunctionRegistry
    from .reporting import Notices


#: The classes `GlobalsFold.class_parts` has folded during one key build, by
#: id. A class's methods can read a global instance of that same class, and
#: the instance leads back to the class: without this the walk never ended.
CLASSES_FOLDED: contextvars.ContextVar[set[int] | None] = contextvars.ContextVar("_cash_classes_folded", default=None)


def _user_bases(cls: type) -> list[type]:
    """*cls*'s own user classes in method-resolution order, and its metaclass's."""
    found = [b for b in cls.__mro__ if b is not object and not is_opaque(b) and is_user_code_object(b)]
    meta = type(cls)
    if meta is not type:
        found += [b for b in meta.__mro__ if b not in (type, object) and not is_opaque(b) and is_user_code_object(b)]
    return found


def _function_layers(fn: Any) -> list[types.FunctionType]:
    """*fn* and the functions it wraps (``__wrapped__``), each a plain function.

    sklearn wraps every ``transform`` a ``TransformerMixin`` subclass defines,
    and cash wraps a cached method: the class attribute is the wrapper, whose
    globals are the library's, and the user's function is inside it.
    """
    layers: list[types.FunctionType] = []
    walked: set[int] = set()
    while fn is not None and id(fn) not in walked:  # every layer; a cycle ends
        walked.add(id(fn))
        if isinstance(fn, types.FunctionType) and not is_cash_wrapper(fn):
            layers.append(fn)
        fn = getattr(fn, "__wrapped__", None)
    return layers


def _member_functions(member: Any) -> list[types.FunctionType]:
    """The functions a class attribute runs: a method, the function inside a
    staticmethod, classmethod, property, cached_property or partialmethod."""
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        candidates = [member.fget, member.fset, member.fdel]
    elif isinstance(member, types.FunctionType):
        candidates = [member]
    else:
        candidates = [getattr(member, attr, None) for attr in ("__func__", "fget", "func")]
    return [layer for c in candidates for layer in _function_layers(c)]


def class_surface_functions(cls: type) -> list[types.FunctionType]:
    """Every function an instance of *cls* can run: its own methods and every
    user base's, inherited ``__init__`` and ``__init_subclass__`` included,
    property and ``cached_property`` accessors, its metaclass's methods, and
    the methods of user objects it holds as class attributes (a descriptor's
    ``__get__``, a callable instance's ``__call__``) -- and of the user
    objects THEIR classes hold, however deep."""
    found: list[types.FunctionType] = []
    classes = [cls]
    walked: set[type] = {cls}
    while classes:
        current = classes.pop(0)
        for base in _user_bases(current):
            for member in list(vars(base).values()):
                functions = _member_functions(member)
                found.extend(functions)
                if (
                    not functions
                    and not isinstance(member, (type, types.ModuleType))
                    and not wraps_code(member)
                    and type(member) not in walked
                    and is_user_code_object(type(member))
                ):
                    walked.add(type(member))
                    classes.append(type(member))
    seen: set[int] = set()
    return [f for f in found if not (id(f) in seen or seen.add(id(f)))]


#: Dunder methods that build, copy, pickle or change an instance: what an
#: instance runs when it is USED -- an operator, ``[]``, ``in``, ``with``,
#: ``format`` -- is every other one (`implicit_access_functions`).
_NOT_USE_DUNDERS = frozenset(
    {
        "__init__",
        "__new__",
        "__init_subclass__",
        "__set_name__",
        "__class_getitem__",
        "__post_init__",
        "__setattr__",
        "__delattr__",
        "__setitem__",
        "__delitem__",
        "__del__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__reduce__",
        "__reduce_ex__",
        "__copy__",
        "__deepcopy__",
    }
)


def implicit_access_functions(cls: type) -> list[types.FunctionType]:
    """The functions an instance of *cls* runs without a call the code spells
    out: property and ``cached_property`` getters, and the dunder methods
    an operator, a subscript, ``in``, ``with``, ``str()`` or ``format()``
    runs. Its own and every user base's."""
    found: list[types.FunctionType] = []
    for base in _user_bases(cls):
        for name, member in vars(base).items():
            if isinstance(member, property):
                found.extend(_function_layers(member.fget))
            elif hasattr(member, "attrname") and isinstance(getattr(member, "func", None), types.FunctionType):
                found.extend(_function_layers(member.func))  # functools.cached_property
            elif name.startswith("__") and name.endswith("__") and name not in _NOT_USE_DUNDERS:
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if isinstance(member, types.FunctionType):
                    found.extend(_function_layers(member))
    seen: set[int] = set()
    return [f for f in found if not (id(f) in seen or seen.add(id(f)))]


def _is_class_machinery(name: str) -> bool:
    """A class-body name the class machinery owns (``__module__``,
    ``__slots__``, ``_abc_impl``...), never data the user reads."""
    return (name.startswith("__") and name.endswith("__")) or name.startswith("_abc_")


def class_data_items(
    cls: type, skip: frozenset[str] = frozenset(), *, bases: list[type] | None = None
) -> list[tuple[str, Any]]:
    """``(label, value)`` for the class-level DATA of *cls* and its user bases.

    A constant (``RATE = 1``), a table, a callable instance held as a class
    attribute, a namedtuple's ``_fields`` and ``_field_defaults``: what its
    methods read through ``self`` or the class. Code (functions and the
    descriptors around them), classes and modules are left to the code
    channels. An Enum is its members' names and values: a member pickles by
    name, so ``RED = 1`` -> ``RED = 5`` changed nothing a pickle sees.
    Names in *skip* -- what the class's own code writes -- are left out.
    *bases* limits the walk to those classes.
    """
    items: list[tuple[str, Any]] = []
    for base in _user_bases(cls) if bases is None else bases:
        prefix = base.__qualname__
        if isinstance(base, enum.EnumMeta):
            members = [(name, member._value_) for name, member in base.__members__.items()]
            items.append((f"{prefix}.__members__", members))
            continue
        for name, value in list(vars(base).items()):
            if _is_class_machinery(name) or name in skip:
                continue
            if isinstance(value, (types.FunctionType, type, types.ModuleType)) or wraps_code(value):
                continue
            if is_cash_wrapper(value):
                continue
            items.append((f"{prefix}.{name}", value))
    return items


class ClassDataFold:
    """Key parts for the data user classes bring: their class-level data,
    hashed and watched, and the globals their functions read."""

    def __init__(
        self,
        args: ArgHasher,
        reads: GlobalReads,
        values: GlobalValues,
        registry: FunctionRegistry,
        mutations: LearnedMutations,
        notices: Notices,
    ) -> None:
        self._args = args
        self._reads = reads
        self._values = values
        self._registry = registry
        self._mutations = mutations
        self._notices = notices
        #: The fold of the globals one function reads
        #: (`GlobalsFold.fold_read_globals`), which a class's functions go
        #: through; bound by `GlobalsFold`, which is built after this.
        self._fold_reads: Callable[..., str] | None = None
        # class -> (its surface functions, the names their code reads); see
        # `class_parts`. A redefined class is a new key.
        self._class_code_cache: LruMemo[type, tuple[tuple, frozenset]] = LruMemo(CODE_OBJECTS)
        # class -> (the immutable data values last hashed, their digest); see
        # `class_data_digest`.
        self._class_data_memo: LruMemo[type, tuple[tuple, str]] = LruMemo(CODE_OBJECTS)
        # class -> its user bases and their data names; see `_class_layout`.
        self._class_layout_cache: LruMemo[type, tuple] = LruMemo(CODE_OBJECTS)

    def bind_reads_fold(self, fold: Callable[..., str]) -> None:
        """Set the fold the globals a class's functions read go through."""
        self._fold_reads = fold

    def class_parts(
        self,
        cls: type,
        func_name: str,
        *,
        owner_code: Any = None,
        seen: set | None = None,
        reader: Any = None,
    ) -> list[tuple[str, str]]:
        """Key parts for the DATA a user class brings: what it holds, and what
        its code reads.

        A class's code reaches the key through its source and its bases';
        the data behind it is folded here:

        * a module global read by an inherited method, a property, a mixin,
          ``__init__`` or ``cached_property`` (``x * RATE`` in ``Base.scale``,
          called as ``Model().scale(x)``);
        * a class attribute set at run time (``Settings.RATE =
          int(sys.argv[1])``) and read through ``self``;
        * ``cfg.Cfg.RATE`` and ``cfg.Color.RED.value`` through ``import cfg``;
        * a class with no source to read: ``namedtuple(...)`` (its fields and
          defaults), ``make_dataclass``, ``type(...)``;
        * a callable instance held as a class attribute.

        Once per class per key build (`CLASSES_FOLDED`). The class data is one
        hash, re-read on every call and watched: a class attribute the call
        itself moves (a counter, a registry) is dropped from the key after the
        first miss that moves it, member by member. The globals the class's
        functions read go through `GlobalsFold.fold_read_globals`, with the
        same drift guard (*owner_code*, the cached function's code) and the
        same dedup (*seen*); a global any of them writes is state, not input,
        and is left out. *reader* is the function that reached the class, for
        naming a member that cannot be hashed.
        """
        folded = CLASSES_FOLDED.get()
        token = None
        if folded is None:
            folded = set()
            token = CLASSES_FOLDED.set(folded)
        try:
            if id(cls) in folded:
                return []
            folded.add(id(cls))
            return self._class_parts(cls, func_name, owner_code, seen, reader)
        finally:
            if token is not None:
                CLASSES_FOLDED.reset(token)

    def _class_parts(
        self, cls: type, func_name: str, owner_code: Any, seen: set | None, reader: Any
    ) -> list[tuple[str, str]]:
        if owner_code is None:
            owner = self._registry.functions.get(func_name)
            owner_code = getattr(owner, "__code__", None)
        learned = self._mutations.of(owner_code, "global") if owner_code is not None else frozenset()
        label = f"class:{cls.__module__}.{cls.__qualname__}"
        parts, unhashable = self._held_data_parts(cls, label, learned)
        functions, excluded, read_names, _ = self._class_code(cls)
        if DOCSTRING_READS & read_names:
            # `self.__doc__` / `inspect.getdoc(type(self))` in its own code.
            parts.extend(self._docstring_parts(cls, label))
        if unhashable:
            self._warn_unhashable_class_data(func_name, unhashable, read_names, reader)
        parts.extend(self._function_read_parts(functions, excluded, label, func_name, owner_code, seen))
        return parts

    def _held_data_parts(self, cls: type, label: str, learned: frozenset) -> tuple[list[tuple[str, str]], list[str]]:
        """``(parts, unhashable labels)`` for what *cls* holds at class level,
        each part watched by the drift guard (`CAPTURE_WATCH`).

        One digest for all of it, unless a call was seen to move it
        (*learned*): then each member is keyed alone, so the one that moves
        is found and left out, and the rest stay keyed.
        """
        parts: list[tuple[str, str]] = []
        watch: dict[str, tuple] = {}
        if label not in learned:
            digest, unhashable = self.class_data_digest(cls)
            if digest is not None:
                parts.append((label, digest))
                watch[label] = (digest, "classdata", (cls, None), None)
        else:
            unhashable = []
            for item_label, _ in self._class_data_items(cls):
                member_label = f"{label}:{item_label}"
                if member_label in learned:
                    continue
                digest, bad = self.class_data_digest(cls, item_label)
                unhashable.extend(bad)
                if digest is not None:
                    parts.append((member_label, digest))
                    watch[member_label] = (digest, "classdata", (cls, item_label), None)
        pending = CAPTURE_WATCH.get()
        if pending is not None and watch:
            pending.update(watch)
        return parts, unhashable

    def _docstring_parts(self, cls: type, label: str) -> list[tuple[str, str]]:
        """Key parts for the docstrings of *cls* and its user bases."""
        parts: list[tuple[str, str]] = []
        for base, _, _ in self._class_layout(cls):
            doc = vars(base).get("__doc__")
            if isinstance(doc, str):
                parts.append((f"{label}:{base.__qualname__}.__doc__", hashlib.sha256(doc.encode("utf-8")).hexdigest()))
        return parts

    def _function_read_parts(
        self,
        functions: tuple,
        excluded: frozenset,
        label: str,
        func_name: str,
        owner_code: Any,
        seen: set | None,
    ) -> list[tuple[str, str]]:
        """Key parts for the globals the class's *functions* read, folded by
        the function fold (`bind_reads_fold`) with the globals they write
        (*excluded*) left out; what they fold joins *seen*."""
        parts: list[tuple[str, str]] = []
        class_seen = set(seen) if seen is not None else set()
        class_seen |= excluded
        for fn in functions:
            if not self._reads.may_read_data(fn):
                continue
            h = self._fold_reads(fn, func_name, "", owner_code=owner_code, seen=class_seen)
            if h:
                parts.append((f"{label}#reads:{fn.__qualname__}", h))
        if seen is not None:
            seen |= class_seen - excluded
        return parts

    def _class_code(self, cls: type) -> tuple[tuple, frozenset, frozenset, frozenset]:
        """``(functions, written globals, names read, attributes written)``
        for *cls*, per class.

        The functions are `class_surface_functions`. The written globals are
        ``(id(module globals), name)`` for every global one of them assigns or
        mutates in place -- state the class keeps, excluded for all of them.
        The names are what their code looks up, for naming a class attribute
        that is read but cannot be hashed. The attributes written are those
        its code stores or mutates in place (`bytecode_written_attrs`): a
        class-level counter or registry is state, not an input.
        """
        cached = self._class_code_cache.get(cls)
        if cached is not None:
            return cached
        functions = tuple(class_surface_functions(cls))
        excluded: set[tuple[int, str]] = set()
        names: set[str] = set()
        written: set[str] = set()
        for fn in functions:
            g = getattr(fn, "__globals__", None)
            if not isinstance(g, dict):
                continue
            data = set(self._reads.read_global_data_names(fn))
            for scope in iter_code_scopes(fn.__code__):
                written |= bytecode_written_attrs(scope)
                names.update(scope.co_names or ())
                names.update(c for c in scope.co_consts or () if isinstance(c, str) and c.isidentifier())
                for n in scope.co_names or ():
                    if n in g and n not in data and n not in MACHINERY_DUNDERS:
                        excluded.add((id(g), n))
        result = (functions, frozenset(excluded), frozenset(names), frozenset(written))
        self._class_code_cache[cls] = result
        return result

    def _class_layout(self, cls: type) -> tuple[tuple[type, int, tuple[str, ...] | None], ...]:
        """``(base, size of its namespace, its data names)`` per user base of
        *cls* (`class_data_items`; None for an Enum), per class.

        Finding the bases and telling data from code costs far more than
        reading the values, and changes only when a name is added to or
        removed from a class, which the namespace sizes stand guard for.
        """
        cached = self._class_layout_cache.get(cls)
        if cached is not None and all(len(vars(base)) == size for base, size, _ in cached):
            return cached
        skip = self._class_code(cls)[3]
        layout = []
        for base in _user_bases(cls):
            if isinstance(base, enum.EnumMeta):
                layout.append((base, len(vars(base)), None))
                continue
            names = tuple(label.split(".")[-1] for label, _ in class_data_items(base, skip, bases=[base]))
            layout.append((base, len(vars(base)), names))
        result = tuple(layout)
        self._class_layout_cache[cls] = result
        return result

    def _class_data_items(self, cls: type) -> list[tuple[str, Any]]:
        """`class_data_items` of *cls*, read through its memoized layout."""
        items: list[tuple[str, Any]] = []
        for base, _, names in self._class_layout(cls):
            prefix = base.__qualname__
            if names is None:
                members = [(name, member._value_) for name, member in base.__members__.items()]
                items.append((f"{prefix}.__members__", members))
                continue
            namespace = vars(base)
            for name in names:
                if name in namespace:
                    items.append((f"{prefix}.{name}", namespace[name]))
        return items

    def class_data_digest(self, cls: type, only: str | None = None) -> tuple[str | None, list[str]]:
        """``(digest, unhashable labels)`` of *cls*'s class-level data
        (`class_data_items`), or of the one member *only*.

        A digest of immutable values is reused while every value is the same
        object, so an unchanged class of constants costs a scan, not a hash.
        A member that cannot be hashed is left out and named.
        """
        items = self._class_data_items(cls)
        if only is not None:
            items = [(label, value) for label, value in items if label == only]
        if not items:
            return None, []
        if only is None:
            entry = self._class_data_memo.get(cls)
            if (
                entry is not None
                and len(entry[0]) == len(items)
                and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(entry[0], items))
            ):
                return entry[1], []
        unhashable: list[str] = []
        try:
            stabilized = {
                label: stabilize_for_global_hash(value, self._values.data_callable_identity) for label, value in items
            }
            digest = self._args.hash_payload((stabilized,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            kept: dict[str, str] = {}
            for label, value in items:
                try:
                    kept[label] = self._args.hash_payload(
                        (stabilize_for_global_hash(value, self._values.data_callable_identity),), {}
                    )
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                    if not isinstance(value, SYNC_TYPES):
                        unhashable.append(label)
            if not kept:
                return None, unhashable
            digest = hashlib.sha256(repr(sorted(kept.items())).encode("utf-8")).hexdigest()
        if only is None and not unhashable and all(is_immutable_capture(value) for _, value in items):
            self._class_data_memo[cls] = (tuple(items), digest)
        return digest, unhashable

    def _warn_unhashable_class_data(
        self, func_name: str, labels: list[str], read_names: frozenset, reader: Any
    ) -> None:
        """Warn once per member that a class attribute the code reads could not be hashed."""
        reader_code = getattr(reader, "__code__", None)
        if reader_code is not None:
            read_names = read_names | {n for scope in iter_code_scopes(reader_code) for n in scope.co_names or ()}
        for label in labels:
            if label.rsplit(".", 1)[-1] not in read_names:
                continue
            self._notices.warn_once(
                CashImpurityWarning,
                func_name,
                label,
                f"@cash.cache on {func_name}: reads the class attribute '{label}' whose "
                f"value could not be hashed, so changes to it will NOT invalidate the cache.",
                code="KEY-UNHASHABLE-GLOBAL",
                fix=UNHASHABLE_GLOBAL_FIX,
            )

    def fold_class_parts(
        self, cls: type, func: Callable, func_name: str, state_hash: str, owner_code: Any, seen: set
    ) -> str:
        """`ClassDataFold.class_parts` for a class the helper walk reached, into *state_hash*."""
        if not is_user_class(cls, own_package(func)):
            return state_hash
        parts = self.class_parts(cls, func_name, owner_code=owner_code, seen=seen)
        if not parts:
            return state_hash
        payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
        return hashlib.sha256(f"{state_hash}:classes:{payload}".encode("utf-8")).hexdigest()
