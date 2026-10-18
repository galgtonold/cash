"""The installed versions of the libraries a cached function calls into.

cash keys a call into a library (anything installed in ``site-packages``
that is not the cached function's own package, a plain-installed team
package included) by name and does not look inside, so after ``pip install
-U`` the results the old version computed were still served, without a word.
The version of every distribution whose code the function, its helpers and
the cached functions it calls reach by a global name is folded into the key:
an upgrade recomputes, an unchanged environment keys as before.

What counts is the library the code names (``np`` in ``np.mean``, ``Ridge``
imported from scikit-learn, ``read_csv`` imported from pandas) and what that
library requires to be installed, as its metadata lists it (a numpy upgrade
under pandas recomputes a ``pd.read_csv`` result too). Optional extras are
left out: they are installed only when asked for, and a library that uses one
is named by the code that asks. A version number is all that is read, so code
reinstalled under the same number keys as before.
"""

from __future__ import annotations

import inspect
import re
import sys
import types
from collections.abc import Callable, Iterable
from typing import Any

from .._lazy_module import LazyModule
from ..install_paths import in_own_package, is_cash_path, is_user_path, top_package

# Imported on first use (6 ms, with zipfile and email): asked only when a
# cached function reaches an installed library.
importlib_metadata = LazyModule("importlib.metadata")

__all__ = ["library_versions"]

#: top-level module name -> the distributions that provide it; filled once.
_DISTS_BY_TOP: dict[str, list[str]] | None = None
#: distribution name -> its installed version, or None when it has none.
_VERSIONS: dict[str, str | None] = {}
#: distribution name -> the installed distributions it requires, transitively.
_CLOSURES: dict[str, frozenset[str]] = {}
#: The project name at the start of a requirement line: ``numpy>=1.26; ...``.
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
#: code object -> the global names it and its nested code read.
_CODE_NAMES: dict[int, tuple[Any, tuple[str, ...]]] = {}


def _packages_distributions() -> dict[str, list[str]]:
    """Top-level module name -> the distributions that provide it.

    ``importlib.metadata.packages_distributions`` on Python 3.10 reads only
    ``top_level.txt``, which wheels built by anything but setuptools (numpy,
    pandas, scikit-learn, ...) do not ship, so it named no distribution for
    them and their upgrades went unseen. Python 3.11+ falls back to the
    distribution's file list; this does the same on every version.
    """
    by_top: dict[str, list[str]] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        tops = (dist.read_text("top_level.txt") or "").split()
        if not tops:
            seen: dict[str, None] = {}
            for f in dist.files or ():
                parts = f.parts
                top = parts[0] if len(parts) > 1 else (inspect.getmodulename(str(f)) or str(f))
                if "." not in top:
                    seen[top] = None
            tops = list(seen)
        for top in tops:
            by_top.setdefault(top, []).append(name)
    return by_top


def _dists_for(top: str) -> list[str]:
    global _DISTS_BY_TOP
    if _DISTS_BY_TOP is None:
        try:
            _DISTS_BY_TOP = _packages_distributions()
        except Exception:  # noqa: BLE001 - broken metadata: no versions to fold
            _DISTS_BY_TOP = {}
    return _DISTS_BY_TOP.get(top, [])


def _version(dist: str) -> str | None:
    if dist not in _VERSIONS:
        try:
            _VERSIONS[dist] = importlib_metadata.version(dist)
        except Exception:  # noqa: BLE001 - no metadata: nothing to fold
            _VERSIONS[dist] = None
    return _VERSIONS[dist]


def _requires(dist: str) -> list[str]:
    """The distributions *dist* requires unconditionally (no ``extra ==``)."""
    try:
        lines = importlib_metadata.requires(dist) or []
    except Exception:  # noqa: BLE001 - no metadata: requires nothing we can name
        return []
    names = []
    for line in lines:
        if "extra" in line.partition(";")[2]:
            continue
        m = _REQ_NAME.match(line)
        if m:
            names.append(m.group(1))
    return names


def _closure(dist: str) -> frozenset[str]:
    """*dist* and every installed distribution it requires, transitively."""
    done = _CLOSURES.get(dist)
    if done is not None:
        return done
    found: set[str] = set()
    stack = [dist]
    while stack:
        d = stack.pop()
        key = d.lower().replace("_", "-").replace(".", "-")
        if key in found or _version(d) is None:
            continue
        found.add(key)
        stack.extend(_requires(d))
    result = frozenset(found)
    _CLOSURES[dist] = result
    return result


def _code_names(code: types.CodeType) -> tuple[str, ...]:
    """Every name *code* and the code nested in it (comprehensions, inner
    functions, lambdas) look up."""
    cached = _CODE_NAMES.get(id(code))
    if cached is not None and cached[0] is code:
        return cached[1]
    names: dict[str, None] = {}
    stack = [code]
    while stack:
        c = stack.pop()
        names.update(dict.fromkeys(c.co_names))
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    found = tuple(names)
    _CODE_NAMES[id(code)] = (code, found)
    return found


def _defining_module(value: Any) -> types.ModuleType | None:
    if isinstance(value, types.ModuleType):
        return value
    if isinstance(value, type) or callable(value):
        name = getattr(value, "__module__", None)
        return sys.modules.get(name) if isinstance(name, str) else None
    return None


def _library_top(module: types.ModuleType, own_pkg: str | None) -> str | None:
    """*module*'s top-level package when it is a library: not the user's
    code, not cash, not the standard library."""
    name = getattr(module, "__name__", None)
    top = top_package(name)
    if top is None or top in sys.stdlib_module_names or top == "builtins":
        return None
    if in_own_package(name, own_pkg):
        return None
    file = getattr(module, "__file__", None)
    if file is None or is_cash_path(file) or is_user_path(file):
        return None
    return top


def library_tops(functions: Iterable[Callable], own_pkg: str | None) -> set[str]:
    """The library packages *functions* name in their global lookups."""
    tops: set[str] = set()
    for fn in functions:
        code = getattr(fn, "__code__", None)
        g = getattr(fn, "__globals__", None)
        if not isinstance(code, types.CodeType) or not isinstance(g, dict):
            continue
        for name in _code_names(code):
            if name not in g:
                continue
            module = _defining_module(g[name])
            if module is None:
                continue
            top = _library_top(module, own_pkg)
            if top is not None:
                tops.add(top)
    return tops


def library_versions(functions: Iterable[Callable], own_pkg: str | None) -> list[tuple[str, str]]:
    """``(distribution, version)`` for each installed library *functions*
    reach, sorted; empty when they reach none."""
    found: set[tuple[str, str]] = set()
    for top in library_tops(functions, own_pkg):
        for dist in _dists_for(top):
            for name in _closure(dist):
                version = _version(name)
                if version is not None:
                    found.add((name, version))
    return sorted(found)


def forget() -> None:
    """Read the installed distributions again (tests that install one)."""
    global _DISTS_BY_TOP
    _DISTS_BY_TOP = None
    _VERSIONS.clear()
    _CLOSURES.clear()
