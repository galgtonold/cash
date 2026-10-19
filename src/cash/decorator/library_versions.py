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

import csv
import importlib.machinery as importlib_machinery
import inspect
import json
import os
import re
import sys
import types
from collections.abc import Callable, Iterable
from typing import Any

from .._lazy_module import LazyModule
from .._location import per_user_cache_root
from ..backends.cache_dir import LIBRARY_MAP_FILENAME
from ..install_paths import in_own_package, is_cash_path, is_user_path, top_package
from ..tracking.tracker_context import untracked

# Imported on first use (6 ms, with zipfile and email): asked only when a
# cached function reaches an installed library.
importlib_metadata = LazyModule("importlib.metadata")

__all__ = ["library_versions"]

#: top-level module name -> ``(distribution, how to check it is there)`` for
#: each distribution that may provide it, filled once: None when its
#: ``top_level.txt`` names the module, ``(its metadata, the files it lists
#: under that name)`` when built here, its metadata folder when read from the
#: saved map (`_saved_map`).
_DISTS_BY_TOP: dict[str, list[tuple[str, Any]]] | None = None
#: top-level module name -> the distributions that provide it (`_dists_for`).
_PROVIDERS: dict[str, list[str]] = {}
#: distribution name -> its installed version, or None when it has none.
_VERSIONS: dict[str, str | None] = {}
#: distribution name -> the installed distributions it requires, transitively.
_CLOSURES: dict[str, frozenset[str]] = {}
#: The project name at the start of a requirement line: ``numpy>=1.26; ...``.
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
#: code object -> the global names it and its nested code read.
_CODE_NAMES: dict[int, tuple[Any, tuple[str, ...]]] = {}


def _listed_files(dist: Any) -> list[str] | None:
    """The files *dist*'s ``RECORD`` lists, as ``dist.files`` reads them but
    before it drops the ones not on disk, which it finds by a stat of every
    file (20,000 in an ordinary environment, half a second, and each process
    a parallel call starts paid it). None without a ``RECORD``."""
    text = dist.read_text("RECORD")
    if text is None:
        return None
    return [row[0] for row in csv.reader(text.splitlines()) if row]


def _packages_distributions() -> tuple[dict[str, list[tuple[str, Any]]], dict[str, list] | None]:
    """Top-level module name -> the distributions that may provide it, each
    with the files to check it by (`_DISTS_BY_TOP`); and the same map as it
    can be saved (`_save_map`), or None when it cannot be.

    ``importlib.metadata.packages_distributions`` on Python 3.10 reads only
    ``top_level.txt``, which wheels built by anything but setuptools (numpy,
    pandas, scikit-learn, ...) do not ship, so it named no distribution for
    them and their upgrades went unseen. Python 3.11+ falls back to the
    distribution's file list; this does the same on every version. A listed
    file counts only if it is on disk, as in ``dist.files``; that is checked
    for the names a cached function reaches, when it does (`_dists_for`).
    """
    by_top: dict[str, list[tuple[str, Any]]] = {}
    saved: dict[str, list] | None = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        tops = (dist.read_text("top_level.txt") or "").split()
        if tops:
            for top in tops:
                by_top.setdefault(top, []).append((name, None))
                if saved is not None:
                    saved.setdefault(top, []).append([name, None])
            continue
        listed = _listed_files(dist)
        if listed is None:  # an egg-info install: its own file list, checked now
            listed, checked = [str(f) for f in dist.files or ()], True
            saved = None  # checked now, so not for another process
        else:
            checked = False
        folder = _metadata_folder(dist)
        if folder is None:
            saved = None
        seen: dict[str, list[str]] = {}
        for f in listed:
            top = _top_of(f)
            if "." not in top:
                seen.setdefault(top, []).append(f)
        for top, files in seen.items():
            by_top.setdefault(top, []).append((name, None if checked else (dist, files)))
            if saved is not None:
                saved.setdefault(top, []).append([name, folder])
    return by_top, saved


def _metadata_folder(dist: Any) -> str | None:
    """The ``.dist-info`` folder *dist* is read from, when it is one on disk:
    what `_on_disk` reads it from again in a process given the saved map."""
    path = getattr(dist, "_path", None)
    if type(dist) is not importlib_metadata.PathDistribution or not isinstance(path, os.PathLike):
        return None
    folder = os.fspath(path)
    return folder if os.path.isdir(folder) else None


def _top_of(f: str) -> str:
    """The top-level name a listed file *f* is under, as ``PackagePath(f)``
    gives it; without building one when splitting on ``/`` gives the same
    parts (no root, no empty or ``.`` part): 20,000 of them cost 0.2 s."""
    if "//" in f or "/." in f or f.startswith(("/", "./")) or f.endswith("/") or f in ("", "."):
        path = importlib_metadata.PackagePath(f)
        parts, f = path.parts, str(path)
    else:
        parts = f.split("/")
    return parts[0] if len(parts) > 1 else (inspect.getmodulename(f) or f)


def _on_disk(listed: Any, top: str) -> bool:
    """Whether any of a distribution's files listed under *top* is there."""
    if listed is None:
        return True
    if isinstance(listed, str):  # from the saved map: its file list read now
        dist = importlib_metadata.Distribution.at(listed)
        listed = (dist, [f for f in _listed_files(dist) or () if _top_of(f) == top])
    dist, files = listed
    for f in files:
        try:
            if dist.locate_file(importlib_metadata.PackagePath(f)).exists():
                return True
        except (OSError, ValueError):
            continue
    return False


def _dists_for(top: str) -> list[str]:
    global _DISTS_BY_TOP
    if _DISTS_BY_TOP is None:
        _DISTS_BY_TOP = _distribution_map()
    names = _PROVIDERS.get(top)
    if names is None:
        names = _PROVIDERS[top] = [name for name, listed in _DISTS_BY_TOP.get(top, ()) if _on_disk(listed, top)]
    return names


#: How many environments' maps the saved file keeps.
_SAVED_ENVIRONMENTS = 8


def _distribution_map() -> dict[str, list[tuple[str, Any]]]:
    """The map `_dists_for` reads: saved by an earlier process of this
    environment while its installs are unchanged, else built and saved.

    Building it reads every installed distribution's metadata and file list,
    0.15 s for 150 of them, and every process paid it on its first key: each
    worker of a parallel call, each run of a script. What it is built from
    is the metadata folders on ``sys.path`` (`_install_fingerprint`): while
    those are the same, so is the map, and so is every key.
    """
    try:
        fingerprint = _install_fingerprint()  # before the build: an install during it rebuilds next time
        if fingerprint is not None:
            saved = _saved_map(fingerprint)
            if saved is not None:
                return saved
        by_top, saveable = _packages_distributions()
    except Exception:  # noqa: BLE001 - broken metadata: no versions to fold
        return {}
    if fingerprint is not None and saveable is not None:
        _save_map(fingerprint, saveable)
    return by_top


def _install_fingerprint() -> list | None:
    """What the installed distributions are read from: each ``sys.path``
    entry and the metadata folders in it, by name and stat (``pip install``,
    ``uninstall`` and ``-U`` each create or remove one). None when something
    else may name distributions too (a finder of its own on ``sys.meta_path``).
    """
    for finder in sys.meta_path:
        if getattr(finder, "find_distributions", None) is not None and finder is not importlib_machinery.PathFinder:
            return None
    found: list = [sys.version, sys.prefix, sys.executable]
    for entry in sys.path:
        path = os.path.abspath(entry or ".")
        try:
            st = os.stat(path)
        except (OSError, ValueError):
            found.append([path])
            continue
        if not os.path.isdir(path):  # a zip: its own stat
            found.append([path, st.st_mtime_ns, st.st_size, st.st_ino])
            continue
        folders = []
        try:
            with os.scandir(path) as children:
                for child in children:
                    if child.name.lower().endswith(("dist-info", "egg-info")):
                        cst = child.stat()
                        folders.append([child.name, cst.st_mtime_ns, cst.st_size, cst.st_ino])
        except OSError:
            return None
        folders.sort()
        found.append([path, folders])
    return found


def _map_path() -> str:
    return os.path.join(per_user_cache_root(), LIBRARY_MAP_FILENAME)


def _environment_id() -> str:
    return f"{sys.prefix}|{sys.executable}"


def _read_saved() -> dict:
    try:
        with untracked(), open(_map_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _saved_map(fingerprint: list) -> dict[str, list[tuple[str, Any]]] | None:
    """The map an earlier process saved for these installs, or None."""
    saved = _read_saved().get(_environment_id())
    if not isinstance(saved, dict) or saved.get("fingerprint") != fingerprint:
        return None
    by_top = saved.get("map")
    if not isinstance(by_top, dict):
        return None
    return {top: [(name, folder) for name, folder in entries] for top, entries in by_top.items()}


def _save_map(fingerprint: list, by_top: dict[str, list]) -> None:
    """Keep *by_top* for the next process of this environment. Best-effort."""
    path = _map_path()
    try:
        with untracked():
            data = _read_saved()
            data.pop(_environment_id(), None)
            data[_environment_id()] = {"fingerprint": fingerprint, "map": by_top}
            for env in list(data)[:-_SAVED_ENVIRONMENTS]:
                del data[env]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temp = f"{path}.{os.getpid()}.tmp"
            with open(temp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(temp, path)
    except (OSError, TypeError, ValueError):
        pass


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
    _PROVIDERS.clear()
    _VERSIONS.clear()
    _CLOSURES.clear()
