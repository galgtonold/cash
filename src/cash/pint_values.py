"""How a key holds a pint quantity, unit or unit registry.

A pint ``Quantity`` is its magnitude, its units (a mapping of unit name to
exponent) and the registry that says what those names mean. Its pickle
drops the registry -- pint unpickles into the application registry -- so
``1 widget`` from a registry where ``widget = 2 m`` and from one where
``widget = 3 m`` pickled alike and shared one cache entry.

A registry is keyed by its definitions: the unit, prefix, suffix and
dimension tables, its groups, systems and contexts, the contexts active,
and its settings. Not by its whole state: it also holds caches of what it
derived from them (parsed names, dimensionalities, root units, the weak
index of a context's relations) that grow as it is used and say nothing
the definitions do not. A context transformation or a preprocessor written
in Python (MetPy adds three preprocessors to the application registry) is
keyed by its compiled body and every value it can read: its defaults,
cells and the globals its code names; a C method bound to an object
(xclim's ``partial(re.compile(...).sub, "**")``) by that object's value, a
compiled pattern by its pattern and flags. One that reads what has no value
form -- a class of the user's, an object known only by its address --
makes the registry unkeyable: the call runs uncached, with a warning
(`KEY-UNHASHABLE-ARG`), never under a key that leaves it out.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import operator
import re
import sys
import types
import weakref
from collections.abc import Mapping
from typing import Any

from . import _weak
from ._type_memo import class_dict, class_module, class_name, class_qualname, per_type
from ._user_attrs import plain_getattr
from .install_paths import is_user_code_module

#: The registry's settings that change what a quantity computes or prints.
_SCALAR_SETTINGS = (
    "case_sensitive",
    "autoconvert_offset_to_baseunit",
    "autoconvert_to_preferred",
    "auto_reduce_dimensions",
    "default_as_delta",
    "force_ndarray",
    "force_ndarray_like",
    "separate_format_defaults",
    "mpl_formatter",
    "_default_system_name",
    "_non_int_type",
)
#: The settings of a registry held in a dict or list.
_TABLE_SETTINGS = ("_defaults", "_base_units")

#: The registry's definition tables, name -> definition.
_TABLES = ("_units", "_prefixes", "_suffixes", "_dimensions")

#: A memory address in a repr: a part with no value form.
_ADDRESS = re.compile(r"\bat 0x[0-9a-fA-F]+")


def _pint_kind(t: type) -> str | None:
    """``"quantity"``, ``"unit"`` or ``"registry"`` for pint's own classes
    (the ones it builds per registry included), else None. A subclass of the
    user's may hold state of its own and is keyed as any other object."""
    pint = sys.modules.get("pint")
    if pint is None or (plain_getattr(t, "__module__", None) or "").partition(".")[0] != "pint":
        return None
    try:
        from pint.facets.plain import GenericPlainRegistry, PlainQuantity, PlainUnit
        from pint.registry import ApplicationRegistry
    except Exception:  # noqa: BLE001 - another pint layout: keyed as before
        return None
    if issubclass(t, PlainQuantity):
        return "quantity"
    if issubclass(t, PlainUnit):
        return "unit"
    if issubclass(t, (GenericPlainRegistry, ApplicationRegistry)):
        return "registry"
    return None


#: `_pint_kind`, per class: asked of every object a key walks.
pint_kind = per_type(_pint_kind)


def _class_registry(cls: type) -> bool:
    """Is *cls* a quantity or unit class pint built for one registry
    (``ureg.Quantity``)? It names its module ``pint`` and pickles by that
    name, which finds pint's own class: two registries' classes keyed alike."""
    return "_REGISTRY" in class_dict(cls) and _pint_kind(cls) in ("quantity", "unit")


#: `_class_registry`, per class.
is_registry_class = per_type(_class_registry)


def quantity_parts(value: Any) -> tuple[str, Any, tuple]:
    """``(registry digest, magnitude, units)`` of a pint quantity or unit;
    the magnitude is None for a unit. Raises TypeError when the registry
    cannot be keyed."""
    units = tuple(sorted(dict(value._units).items()))
    magnitude = value._magnitude if hasattr(value, "_magnitude") else None
    return registry_digest(type(value)._REGISTRY), magnitude, units


def _unwrapped(registry: Any) -> Any:
    """The registry an application-registry wrapper stands for now."""
    get = getattr(type(registry), "get", None)
    if class_name(type(registry)) == "ApplicationRegistry" and callable(get):
        return registry.get()
    return registry


#: registry -> (fingerprint, definitions seen, digest). Weak: a registry
#: dropped is forgotten.
_DIGESTS: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


#: id(registry) -> digest, for one key build: a call whose arguments and
#: defaults hold ten quantities checks their registry once. Set by the
#: decorator around each key build; None outside one.
BUILD_DIGESTS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_cash_pint_digests", default=None)


def registry_digest(registry: Any) -> str:
    """The digest of *registry*'s definitions, worked out again only when
    a setting moved (`_fingerprint`) or one of its definition tables holds
    another definition (`_unchanged`)."""
    registry = _unwrapped(registry)
    build = BUILD_DIGESTS.get()
    if build is None:
        return _checked_digest(registry)
    found = build.get(id(registry))
    if found is None or found[0] is not registry:
        found = build[id(registry)] = (registry, _checked_digest(registry))
    return found[1]


def _checked_digest(registry: Any) -> str:
    fingerprint = _fingerprint(registry)
    tables = _tables(registry)
    try:
        entry = _DIGESTS.get(registry)
    except TypeError:
        entry = None
    if entry is not None and entry[0] == fingerprint and _unchanged(entry[1], tables):
        return entry[2]
    digest = hashlib.sha256(repr(_registry_form(registry)).encode("utf-8")).hexdigest()
    try:
        _DIGESTS[registry] = (fingerprint, tuple((t, tuple(_items(t))) for t in tables), digest)
    except TypeError:
        pass
    return digest


def _items(table: Any) -> Any:
    return table.values() if isinstance(table, Mapping) else table


def _tables(reg: Any) -> list:
    """Every dict and list *reg*'s definitions live in: the definition
    tables (each map of a ChainMap), the defaults, base units and
    preprocessors, and the transformations, defaults and redefinitions of
    each context, registered or active."""
    found: list = []
    for name in _TABLES:
        table = getattr(reg, name, None)
        if table is not None:
            found.extend(getattr(table, "maps", None) or [table])
    for name in ("_defaults", "_base_units", "preprocessors"):
        table = getattr(reg, name, None)
        if table is not None:
            found.append(table)
    contexts = list((getattr(reg, "_contexts", None) or {}).values())
    contexts.extend(getattr(getattr(reg, "_active_ctx", None), "contexts", ()))
    for ctx in contexts:
        for name in ("funcs", "defaults", "redefinitions"):
            table = getattr(ctx, name, None)
            if table is not None:
                found.append(table)
    return found


def _unchanged(seen: tuple, tables: list) -> bool:
    """Are *tables* the containers *seen*, each holding the same objects?
    pint's ``define`` replaces a definition under its name when
    redefinition is allowed, which no length shows. By identity, without
    building anything: an object kept is a value kept."""
    if len(seen) != len(tables):
        return False
    is_ = operator.is_
    for (table, values), now in zip(seen, tables):
        if table is not now or len(values) != len(now) or not all(map(is_, values, _items(now))):
            return False
    return True


def _fingerprint(reg: Any) -> tuple:
    """*reg*'s settings, and the sizes of its groups and systems."""
    parts: list = [getattr(reg, name, None) for name in _SCALAR_SETTINGS]
    for name in ("_groups", "_systems"):
        for g in (getattr(reg, name, None) or {}).values():
            own = g.__dict__
            parts.append((id(g), *[len(own.get(n) or ()) for n in _GROUP_PARTS]))
    formatter = getattr(reg, "formatter", None)
    parts.append(getattr(formatter, "default_format", None))
    parts.append(repr(getattr(formatter, "locale", None)))
    parts.append(_foreign_functions_form(reg))
    return tuple(parts)


def _foreign_functions_form(reg: Any) -> tuple:
    """`_function_form` of each function in *reg* that pint did not build
    (a preprocessor or a context transformation a library or the user
    added). Part of the fingerprint: what such a function reads can change
    while the same function object stays in the registry."""
    found: list = list(getattr(reg, "preprocessors", None) or ())
    contexts = list((getattr(reg, "_contexts", None) or {}).values())
    contexts.extend(getattr(getattr(reg, "_active_ctx", None), "contexts", ()))
    for ctx in contexts:
        found.extend((getattr(ctx, "funcs", None) or {}).values())
    return tuple(
        _function_form(fn, "function")
        for fn in found
        if not (isinstance(fn, types.FunctionType) and (fn.__module__ or "").partition(".")[0] == "pint")
    )


#: What a group or system of units is (pint's ``Group`` and ``System``
#: forward unknown attributes to the registry, so they are read from the
#: instance's own dict).
_GROUP_PARTS = ("_unit_names", "_used_groups", "base_units", "derived_units")


def _own(obj: Any, name: str) -> Any:
    value = vars(obj).get(name)
    return sorted(value) if isinstance(value, (set, frozenset)) else value


def _setting(reg: Any, name: str) -> Any:
    value = getattr(reg, name, None)
    if isinstance(value, type):
        return f"{class_module(value)}.{class_qualname(value)}"
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    if isinstance(value, list):
        return tuple(value)
    return value


def _value_repr(value: Any, where: str) -> str:
    """repr of a definition, or TypeError when it has no value form."""
    text = repr(value)
    if _ADDRESS.search(text):
        raise TypeError(f"the pint registry's {where} has no value form ({text[:120]})")
    return text


def _function_form(fn: Any, where: str) -> tuple:
    """A function in the registry -- a context transformation or a
    preprocessor. pint's own, built from a definition, by its name and what
    it closes over (the parsed relation, an equation's text); any other by
    its code and every value it can read (`_code_form`)."""
    module = plain_getattr(fn, "__module__", None) or ""
    if isinstance(fn, types.FunctionType) and module.partition(".")[0] == "pint":
        cells = tuple(_value_repr(c.cell_contents, where) for c in fn.__closure__ or ())
        return (module, fn.__qualname__, fn.__code__.co_code, cells)
    return _held_form(fn, where, set())


def _code_names(code: types.CodeType) -> set[str]:
    """The global names *code* and the code nested in it can load."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


def _code_form(fn: types.FunctionType, where: str, walking: set[int]) -> tuple:
    """A Python function by value: its compiled body (`bytecode_identity`,
    stable across processes), its defaults, the cells it closes over and
    the globals its code names, each by `_held_form`. Read again at every
    digest: a global rebound or a cell written changes what it does with
    the same function object in the registry."""
    from .source_norm import bytecode_identity

    body = bytecode_identity(fn)
    if body is None:
        raise TypeError(f"the pint registry's {where} holds {fn.__qualname__}, whose code cash cannot read")
    walking.add(id(fn))
    try:
        held = [("default", _held_form(v, where, walking)) for v in fn.__defaults__ or ()]
        held += [("kwdefault", k, _held_form(v, where, walking)) for k, v in sorted((fn.__kwdefaults__ or {}).items())]
        held += [("cell", _held_form(c.cell_contents, where, walking)) for c in fn.__closure__ or ()]
        namespace = fn.__globals__
        for name in sorted(_code_names(fn.__code__)):
            if name in namespace:
                held.append(("global", name, _held_form(namespace[name], where, walking)))
    except ValueError:  # an empty cell: a name the function binds after it is made
        raise TypeError(
            f"the pint registry's {where} holds {fn.__qualname__}, which closes over an unset name"
        ) from None
    finally:
        walking.discard(id(fn))
    return ("code", fn.__module__, fn.__qualname__, body, tuple(held))


def _held_form(value: Any, where: str, walking: set[int]) -> Any:
    """What a function in the registry can read, as plain values: a function
    by `_code_form`, a partial by what it calls and binds, a module or a
    library class or builtin by its name, any other value by its repr.
    TypeError for what has none of these -- a class of the user's, whose
    methods could change, or an object whose repr is its address."""
    if isinstance(value, types.FunctionType):
        if id(value) in walking:
            return ("recursive", value.__module__, value.__qualname__)
        return _code_form(value, where, walking)
    if isinstance(value, functools.partial):
        keywords = sorted(value.keywords.items())
        return (
            "partial",
            _held_form(value.func, where, walking),
            tuple(_held_form(a, where, walking) for a in value.args),
            tuple((k, _held_form(v, where, walking)) for k, v in keywords),
        )
    if isinstance(value, types.ModuleType):
        return ("module", value.__name__)
    bound = plain_getattr(value, "__self__", None) if isinstance(value, types.BuiltinMethodType) else None
    if bound is not None and not isinstance(bound, types.ModuleType):
        # A C method bound to an object (``re.compile(...).sub``): what it
        # does is its name on that object's built-in class, read from the
        # object's value -- a compiled pattern by its pattern and flags.
        if id(bound) in walking:
            raise TypeError(f"the pint registry's {where} reads a method bound to an object that contains it")
        walking.add(id(bound))
        try:
            return ("bound", _held_form(bound, where, walking), value.__name__)
        finally:
            walking.discard(id(bound))
    if isinstance(value, (type, types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType)):
        owner = plain_getattr(value, "__module__", None) or getattr(
            getattr(value, "__objclass__", None), "__module__", None
        )
        if owner is None or is_user_code_module(sys.modules.get(owner)):
            name = plain_getattr(value, "__qualname__", repr(value))
            raise TypeError(f"the pint registry's {where} reads {name}, which cash cannot key by value")
        return ("named", owner, class_qualname(value))
    t = type(value)
    if t in _PLAIN_LEAVES:
        return (t.__name__, repr(value))
    if t is re.Pattern:
        return ("pattern", value.pattern, value.flags)
    if t in _CONTAINERS:
        if id(value) in walking:
            raise TypeError(f"the pint registry's {where} reads a {t.__qualname__} that contains itself")
        walking.add(id(value))
        try:
            if t is dict:
                items = tuple((_held_form(k, where, walking), _held_form(v, where, walking)) for k, v in value.items())
            elif t in (set, frozenset):
                items = tuple(sorted(repr(_held_form(v, where, walking)) for v in value))
            else:
                items = tuple(_held_form(v, where, walking) for v in value)
        finally:
            walking.discard(id(value))
        return (t.__name__, items)
    raise TypeError(
        f"the pint registry's {where} reads a {t.__qualname__}, which cash cannot key by value; "
        f"register a hasher for the quantity"
    )


#: Containers a registry function reads, keyed by what they hold.
_CONTAINERS = frozenset({tuple, list, dict, set, frozenset})

#: Values a registry function reads that are their repr.
_PLAIN_LEAVES = frozenset({type(None), bool, int, float, complex, str, bytes})


def _context_form(ctx: Any) -> tuple:
    name = getattr(ctx, "name", None)
    funcs = sorted(
        (_value_repr(key, f"context {name}"), _function_form(fn, f"context {name}"))
        for key, fn in getattr(ctx, "funcs", {}).items()
    )
    return (
        name,
        tuple(getattr(ctx, "aliases", ())),
        _value_repr(sorted(getattr(ctx, "defaults", {}).items()), f"context {name}"),
        _value_repr(list(getattr(ctx, "redefinitions", ())), f"context {name}"),
        tuple(funcs),
    )


def _registry_form(reg: Any) -> tuple:
    """Everything *reg*'s definitions and settings say, as plain values."""
    t = type(reg)
    pint = sys.modules.get("pint")
    parts: list = [f"{t.__module__}.{t.__qualname__}", getattr(pint, "__version__", None)]
    parts.append(_value_repr([(n, _setting(reg, n)) for n in _SCALAR_SETTINGS + _TABLE_SETTINGS], "settings"))
    for name in _TABLES:
        table = getattr(reg, name, None) or {}
        parts.append((name, tuple((k, _value_repr(v, f"{name}[{k!r}]")) for k, v in sorted(table.items()))))
    for name in ("_groups", "_systems"):
        objs = getattr(reg, name, {}) or {}
        parts.append(
            (
                name,
                tuple(
                    (k, _value_repr([_own(g, n) for n in _GROUP_PARTS], f"{name}[{k!r}]"))
                    for k, g in sorted(objs.items())
                ),
            )
        )
    contexts = getattr(reg, "_contexts", {}) or {}
    parts.append(tuple((k, _context_form(c)) for k, c in sorted(contexts.items())))
    active = getattr(reg, "_active_ctx", None)
    parts.append(tuple(_context_form(c) for c in getattr(active, "contexts", ())))
    parts.append(tuple(_function_form(fn, "preprocessor") for fn in getattr(reg, "preprocessors", ())))
    formatter = getattr(reg, "formatter", None)
    parts.append((getattr(formatter, "default_format", None), repr(getattr(formatter, "locale", None))))
    return tuple(parts)
