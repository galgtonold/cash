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
the definitions do not. A definition that has no value form -- a context
transformation or a preprocessor written in Python rather than in a
definitions file -- makes the registry unkeyable: the call runs uncached,
with a warning (`KEY-UNHASHABLE-ARG`), never under a key that leaves it out.
"""

from __future__ import annotations

import contextvars
import hashlib
import operator
import re
import sys
import types
import weakref
from collections.abc import Mapping
from typing import Any

from ._type_memo import per_type

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
    if pint is None or (getattr(t, "__module__", None) or "").partition(".")[0] != "pint":
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
    return "_REGISTRY" in vars(cls) and _pint_kind(cls) in ("quantity", "unit")


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
    if type(registry).__name__ == "ApplicationRegistry" and callable(get):
        return registry.get()
    return registry


#: registry -> (fingerprint, definitions seen, digest). Weak: a registry
#: dropped is forgotten.
_DIGESTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


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
    return tuple(parts)


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
        return f"{value.__module__}.{value.__qualname__}"
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
    """A function pint built from a definition: its name and what it closes
    over (the parsed relation, an equation's text). A function of the
    user's has no value form here."""
    module = getattr(fn, "__module__", None) or ""
    if not isinstance(fn, types.FunctionType) or module.partition(".")[0] != "pint":
        name = getattr(fn, "__qualname__", type(fn).__qualname__)
        raise TypeError(
            f"the pint registry's {where} is the Python function {name}, which cash cannot key by value; "
            f"define it in a definitions string or file, or register a hasher for the quantity"
        )
    cells = tuple(_value_repr(c.cell_contents, where) for c in fn.__closure__ or ())
    return (module, fn.__qualname__, fn.__code__.co_code, cells)


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
