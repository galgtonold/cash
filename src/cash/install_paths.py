"""Is this path, module or code the user's, or part of the Python installation?

One answer for every part of cash that asks: the file tracker (is a read the
user's data or a library's?), the notebook's function tracker (is an imported
module local, so its edits should invalidate?), the decorator (does a
module's global belong in the key?), the helper walk (is a callee followed?),
cacheability analysis and the config's cache-directory anchor.

The verdicts, from the path up: `is_user_path` for a file, `is_user_module`
for a module (strict: it has a file, and the file is the user's),
`is_user_code_module` and `is_user_code_file` (lenient: code with no file
-- a notebook cell, ``exec`` -- is the user's too), and `in_own_package`,
the one shortcut that makes the cached function's own package user code
wherever it is installed. The strict verdicts decide what is warned about
and walked; the lenient ones what is keyed.

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

from ._memo import SOURCE_FILES, LruMemo
from ._paths import normalize_path

__all__ = [
    "in_own_package",
    "installed_roots",
    "interpreter_roots",
    "is_installed_path",
    "is_cash_path",
    "is_user_code_file",
    "is_user_code_module",
    "is_user_module",
    "is_user_path",
    "top_package",
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


_USER_PATH: LruMemo[str, bool] = LruMemo(SOURCE_FILES)


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
    _USER_PATH[path] = verdict
    return verdict


def is_cash_path(path: str | os.PathLike[str] | None) -> bool:
    """Is *path* a file of cash itself (an editable install included)?"""
    if not path:
        return False
    path = os.fspath(path)
    return not path.startswith("<") and normcase_path(os.path.abspath(path)).startswith(_CASH_DIR)


def is_user_code_file(filename: str | None) -> bool:
    """Is *filename*, a code object's ``co_filename``, the user's code? A
    real path by `is_user_path`; a pseudo-filename (``<ipython-input-3>``,
    ``<string>``) is a notebook cell or ``exec``'d source, which the user
    writes."""
    if not filename:
        return False
    return filename.startswith("<") or is_user_path(filename)


def top_package(module_name: str | None) -> str | None:
    """The top-level package of *module_name*: ``pkg`` for ``pkg.sub.mod``."""
    top = (module_name or "").split(".")[0]
    return top or None


def in_own_package(module_name: str | None, own_pkg: str | None) -> bool:
    """Is *module_name* inside *own_pkg*, the cached function's top-level
    package? That package is user code wherever it is installed (a tool of
    the user's that is ``pip install``ed, an editable install), so this
    comes before any path test. A script (``__main__``) is not a package:
    only the script itself counts, not what it imports."""
    if not module_name or not own_pkg:
        return False
    return module_name == own_pkg or module_name.startswith(own_pkg + ".")


#: Modules with no file that are NOT the user's code. Everything else without
#: a file is a notebook cell, a REPL or ``exec``'d source.
_FILELESS_NON_USER = frozenset(sys.builtin_module_names) | {
    "builtins",
    "__future__",
    "_frozen_importlib",
    "_frozen_importlib_external",
}

#: ``__spec__.origin`` of a module compiled into the interpreter or frozen
#: into it: never the user's code, whatever its ``__name__`` says.
_INTERPRETER_ORIGINS = frozenset({"built-in", "frozen"})


def is_user_module(mod: object, own_pkg: str | None = None) -> bool:
    """Is *mod* a module of the user's (strict): inside *own_pkg*
    (`in_own_package`), or loaded from a file that `is_user_path` accepts?
    A module with no file has nothing on disk to judge, so not."""
    if in_own_package(getattr(mod, "__name__", None), own_pkg):
        return True
    path = getattr(mod, "__file__", None)
    return isinstance(path, str) and bool(path) and is_user_path(path)


def is_user_code_module(mod: object, own_pkg: str | None = None) -> bool:
    """`is_user_module`, except that a module with no file is the user's
    (lenient): a notebook's ``__main__`` and ``exec``'d code have none, and
    they are what the user edits. A module built into or frozen into the
    interpreter is judged by its spec, not its name (``_io`` calls itself
    ``io``)."""
    if getattr(mod, "__file__", None) is None and not in_own_package(getattr(mod, "__name__", None), own_pkg):
        name = getattr(mod, "__name__", "") or ""
        origin = getattr(getattr(mod, "__spec__", None), "origin", None)
        return origin not in _INTERPRETER_ORIGINS and name not in _FILELESS_NON_USER
    return is_user_module(mod, own_pkg)


def clear_caches() -> None:
    """Forget every root and verdict -- for a test that changes the installation."""
    for fn in (interpreter_roots, site_roots, _script_roots, installed_roots):
        fn.cache_clear()
    _USER_PATH.clear()
