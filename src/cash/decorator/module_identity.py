"""What a module held by a name is keyed by: which module it is."""

from __future__ import annotations

import sys
import types
from typing import Any

_OWN_PACKAGE = __name__.partition(".")[0]


def module_ref(module: types.ModuleType) -> str:
    """The identity of *module* in a key: its import name, the same in every
    process, never its ``id``.

    ``__FFTLIB = numpy.fft`` edited to ``scipy.fft`` must move the key of
    every function that reads it, though neither function's code changed.
    Read from the module's namespace, not through attribute access, so a
    lazy module (`importlib.util.LazyLoader`) is not loaded by keying it and
    keys the same before its first use as after. The import system's name
    (``__spec__.name``) comes first: a module that renames itself on load
    keeps it. A module not importable by that name (made with
    ``types.ModuleType`` or loaded from a file path) adds the file it came
    from, so two such modules sharing a name are told apart.
    """
    try:
        namespace = object.__getattribute__(module, "__dict__")
    except Exception:  # noqa: BLE001 - a module with no namespace: its type stands for it
        namespace = {}
    spec = namespace.get("__spec__")
    spec_name = getattr(spec, "name", None) if spec is not None else None
    names = [n for n in (spec_name, namespace.get("__name__")) if isinstance(n, str)]
    for name in names:
        if sys.modules.get(name) is module:
            return name
    origin = namespace.get("__file__")
    if not isinstance(origin, str):
        origin = getattr(spec, "origin", None) if spec is not None else None
    name = names[0] if names else f"{type(module).__module__}.{type(module).__qualname__}"
    return f"{name}@{origin}" if isinstance(origin, str) else name


def module_ref_part(label: str, value: Any) -> list[tuple[str, str]]:
    """The key part for *label* holding the module *value*:
    ``(label#module, name)``; none when *value* is not a module.

    None for cash's own package either: ``with cash.assume_safe():`` reads
    the name ``cash`` only to mark lines, and wrapping lines in it must
    leave the key alone (`source_norm`). A name rebound from cash to
    another module still moves the key: that module's part appears."""
    if not issubclass(type(value), types.ModuleType):
        return []
    ref = module_ref(value)
    if ref == _OWN_PACKAGE or ref.startswith(_OWN_PACKAGE + "."):
        return []
    return [(f"{label}#module", ref)]
