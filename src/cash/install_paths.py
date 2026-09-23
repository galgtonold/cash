"""Is this path the user's code, or part of the Python installation?

One answer for every part of cash that asks: the file tracker (is a read the
user's data or a library's?), the notebook's function tracker (is an imported
module local, so its edits should invalidate?), the decorator (does a
module's global belong in the key?), cacheability analysis and the config's
cache-directory anchor.

The installation is the standard library (with its compiled extensions and a
zipped stdlib), site-packages -- the interpreter's, the user site, and any
directory named ``site-packages`` or ``dist-packages`` -- and the directories
console scripts are installed into. NOT the interpreter prefix itself: that
is the standard library's parent, or, for a virtualenv created as the project
folder (``python -m venv .``), the user's whole project. A module next to the
venv's ``bin/`` and ``lib/`` is the user's.
"""

from __future__ import annotations

import functools
import os
import site
import sys
import sysconfig

from ._paths import normalize_path

__all__ = [
    "installed_roots",
    "interpreter_roots",
    "is_installed_path",
    "is_user_path",
    "norm_dir",
    "normcase_path",
    "site_roots",
    "clear_caches",
]


def normcase_path(path: str) -> str:
    """*path* case-folded where the OS is, with forward slashes.

    ``normcase`` on Windows turns ``/`` back into ``\\``, so it has to come
    first, or a directory prefix would never match a path under it.
    """
    return normalize_path(os.path.normcase(path))


def norm_dir(path: str) -> str:
    """*path* as a directory prefix: absolute, `normcase_path`'d, one trailing slash."""
    return normcase_path(os.path.abspath(path)).rstrip("/") + "/"


def _norm_dirs(path: str) -> set[str]:
    """*path* as a directory prefix, spelled as given and with links resolved:
    a caller may hand over either."""
    dirs = {norm_dir(path)}
    try:
        dirs.add(norm_dir(os.path.realpath(path)))
    except OSError:
        pass
    return dirs


def _prefixes() -> set[str]:
    dirs: set[str] = set()
    for p in {sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix}:
        dirs |= _norm_dirs(p)
    return dirs


@functools.lru_cache(maxsize=1)
def interpreter_roots() -> tuple[str, ...]:
    """The standard library, its compiled extensions and zipped stdlib."""
    roots: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib"):
        if paths.get(key):
            roots |= _norm_dirs(paths[key])
    for prefix in {sys.base_prefix, sys.base_exec_prefix}:
        roots |= _norm_dirs(os.path.join(prefix, "DLLs"))
    for entry in sys.path:
        if entry and entry.lower().endswith(".zip"):
            roots |= _norm_dirs(entry)
    return tuple(sorted(roots))


@functools.lru_cache(maxsize=1)
def site_roots() -> tuple[str, ...]:
    """Where installed third-party packages live (site-packages).

    Often INSIDE the standard library directory (``Lib/site-packages`` on
    Windows, ``lib/python3.X/site-packages`` in conda), so a test for the
    interpreter has to exclude these, or it would swallow every installed
    package -- and with it the own-package exemption.
    """
    roots: set[str] = set()
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        if paths.get(key):
            roots |= _norm_dirs(paths[key])
    try:
        for entry in site.getsitepackages():
            roots |= _norm_dirs(entry)
    except AttributeError:  # virtualenv's old site.py
        pass
    try:
        roots |= _norm_dirs(site.getusersitepackages())
    except (AttributeError, TypeError):
        pass
    # On Windows `getsitepackages()` also lists the installation prefix itself.
    # That is the standard library's parent -- or, for a venv created as the
    # project folder, the user's whole project -- never a package directory.
    return tuple(sorted(roots - _prefixes()))


@functools.lru_cache(maxsize=1)
def _script_roots() -> tuple[str, ...]:
    """Where console scripts are installed: ``bin/`` or ``Scripts/`` of the
    environment and of the user base."""
    roots: set[str] = set()
    schemes = [None]
    user_scheme = f"{os.name}_user"
    if user_scheme in sysconfig.get_scheme_names():
        schemes.append(user_scheme)
    for scheme in schemes:
        try:
            scripts = sysconfig.get_path("scripts", scheme) if scheme else sysconfig.get_path("scripts")
        except KeyError:
            continue
        if scripts:
            roots |= _norm_dirs(scripts)
    return tuple(sorted(roots - _prefixes()))


@functools.lru_cache(maxsize=1)
def installed_roots() -> tuple[str, ...]:
    """Where the installation lives: the standard library, site-packages and
    the console-script directories. Never a whole interpreter prefix."""
    return tuple(sorted(set(interpreter_roots()) | set(site_roots()) | set(_script_roots())))


_PACKAGE_DIR_SEGMENTS = ("/site-packages/", "/dist-packages/")

#: cash's own package directory: cash is not the user's code, wherever it is
#: installed -- an editable install included.
_CASH_DIR = norm_dir(os.path.dirname(os.path.abspath(__file__)))


def _is_installed(path_nc: str) -> bool:
    return path_nc.startswith(installed_roots()) or any(seg in path_nc for seg in _PACKAGE_DIR_SEGMENTS)


def is_installed_path(path: str | os.PathLike[str]) -> bool:
    """Does *path* live inside the Python installation (see the module docstring)?"""
    return _is_installed(normcase_path(os.path.abspath(os.fspath(path))))


_USER_PATH: dict[str, bool] = {}


def is_user_path(path: str | os.PathLike[str] | None) -> bool:
    """Is *path* a file of code the user writes: a real path, outside the
    installation and outside cash itself? A pseudo-filename (``<stdin>``,
    ``<ipython-input-3>``) is not a path, so not."""
    if not path:
        return False
    path = os.fspath(path)
    verdict = _USER_PATH.get(path)
    if verdict is not None:
        return verdict
    if path.startswith("<"):
        verdict = False
    else:
        path_nc = normcase_path(os.path.abspath(path))
        verdict = not _is_installed(path_nc) and not path_nc.startswith(_CASH_DIR)
    if len(_USER_PATH) < 8192:
        _USER_PATH[path] = verdict
    return verdict


def clear_caches() -> None:
    """Forget every root and verdict -- for a test that changes the installation."""
    for fn in (interpreter_roots, site_roots, _script_roots, installed_roots):
        fn.cache_clear()
    _USER_PATH.clear()
