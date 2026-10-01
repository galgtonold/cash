"""Is the code on disk still the code that is running?

A helper's digest is computed lazily, the first time a call needs it, from
the source on disk. If the file was edited after this process imported it
-- every deploy, every ``git pull`` under a long-running worker -- that
digest describes the new text while the old code runs, and a result of the
old code would be stored, and served after a restart, under the new code's
key. The checks here tell the two apart: from the module's ``.pyc`` when it
proves the import saw this file, else by compiling the file.

Recompiling just the function's text is not a valid comparison: bytecode
depends on the surrounding module (inside its module the compiler can see
that ``hashlib`` is an imported module and emits a different call form
than for the same text compiled alone). Compiling the whole file gives the
compiler the context the import had, and `bytecode_identity` ignores line
numbers, so an edit to a neighbouring function does not read as one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import marshal
import os
import types

from ._memo import COMPILED_MODULES, LruMemo
from .process_start import process_start_time
from .source_norm import bytecode_identity
from .source_reading import SETTLED_SECONDS, read_code_file, stat_has_settled
from .tracking.tracker_context import untracked

__all__ = ["class_functions", "loaded_class_identity", "loaded_code_matches_disk", "loaded_module_matches_disk"]

_MODULE_CODE_CACHE: LruMemo[str, tuple[int, int, types.CodeType | None]] = LruMemo(COMPILED_MODULES)


def _compiled_module(path: str) -> types.CodeType | None:
    """The whole file at *path*, compiled, cached per (path, mtime, size)."""

    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _MODULE_CODE_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    settled = stat_has_settled(st)
    try:
        source = read_code_file(path)
        code: types.CodeType | None = compile(source, path, "exec", dont_inherit=True)
    except (OSError, SyntaxError, ValueError):
        code = None
    if not settled:
        return code
    _MODULE_CODE_CACHE[path] = (st.st_mtime_ns, st.st_size, code)
    return code


def _pyc_postdates_source(st: object, pyc_st: object) -> bool:
    """True when the ``.pyc`` was written after the source's last save had settled.

    The header records the source's mtime in WHOLE seconds and its size, so it
    cannot tell apart two saves of the same size inside one second: import the
    first, save the second (``sum`` -> ``max``) in the same second, and every
    later import -- a kernel restart included -- loads the first save's
    bytecode, because the header still matches. A ``.pyc`` written more than a
    tick after the source's mtime (`stat_has_settled`'s window, which covers
    the coarsest clocks) was compiled from the last save: any later save would
    carry a later mtime, which the header would not match.
    """
    return pyc_st.st_mtime - st.st_mtime > SETTLED_SECONDS


def loaded_module_matches_disk(module: object) -> bool:
    """False when *module* was loaded from a ``.pyc`` that is not its source file.

    Python loads a module from its ``.pyc`` when the header matches the
    source's (mtime in whole seconds, size), which a same-size save inside the
    second of the last import keeps (`_pyc_postdates_source`). The module then
    runs the previous save's code while every key cash builds from the file
    describes the new one, and a value computed by the old code is stored --
    and persisted -- under the new code's key.

    True whenever the import cannot have used stale bytecode: no source file,
    no ``.pyc``, a header that does not match the source (the import compiled
    the source instead), a checked hash-based ``.pyc`` (Python compares it
    with the source's hash), or a ``.pyc`` written after the source had
    settled. Otherwise the ``.pyc``'s code is compared with the source
    compiled, once per file version.
    """

    path = getattr(module, "__file__", None)
    if not path or not path.endswith((".py", ".pyw")):
        return True
    try:
        st = os.stat(path)
        pyc = importlib.util.cache_from_source(path)
        pyc_st = os.stat(pyc)
        # Untracked for the reason `_pyc_proves_unchanged` gives.
        with untracked(), io.FileIO(pyc, "rb") as fh:
            header = fh.read(16)
            if len(header) < 16 or header[:4] != importlib.util.MAGIC_NUMBER:
                return True
            flags = int.from_bytes(header[4:8], "little")
            if flags & 0b1:
                if flags & 0b10:
                    return True  # checked hash-based: validated against the source
            elif int.from_bytes(header[8:12], "little") != (int(st.st_mtime) & 0xFFFFFFFF) or int.from_bytes(
                header[12:16], "little"
            ) != (st.st_size & 0xFFFFFFFF):
                return True
            elif _pyc_postdates_source(st, pyc_st):
                return True
            body = fh.readall()
    except (OSError, ValueError, NotImplementedError):
        return True
    try:
        loaded = marshal.loads(body)
    except (EOFError, ValueError, TypeError):
        return True
    on_disk = _compiled_module(path)
    if on_disk is None:
        return True  # nothing to reload to: the error belongs to the next import
    return loaded == on_disk


def _pyc_proves_unchanged(path: str, st: object) -> bool:
    """True when the module's ``.pyc`` shows *path* is what this process imported.

    A file's mtime alone proved nothing: ``shutil.copy2``, ``cp -p``, rsync,
    robocopy and Explorer all keep the SOURCE file's mtime, so a helper
    replaced under a running process by a copy made two hours earlier looked
    untouched since the process started -- and was keyed by the new text while
    the old code ran.

    The ``.pyc`` is a record of the import: importlib checks its header against
    the source's (mtime, size) and rewrites it when they differ. A ``.pyc``
    older than this process whose header still matches the source means the
    import saw exactly this (mtime, size) -- and, when the ``.pyc`` was written
    after the source's mtime tick had passed (`_pyc_postdates_source`), exactly
    this file. A newer one may have been written by a later import of a
    different file, and a missing one says nothing -- both fall back to
    compiling the file, once per (path, mtime, size).
    """

    started = process_start_time()
    if st.st_mtime > started:
        return False
    try:
        pyc = importlib.util.cache_from_source(path)
        pyc_st = os.stat(pyc)
        if pyc_st.st_mtime > started or not _pyc_postdates_source(st, pyc_st):
            return False
        # Untracked: read inside a cached call's body -- a nested cached
        # call's key being built -- the module's .pyc became that call's
        # input, and editing ANY function in the module re-ran it (a 25 s
        # step on every deploy). The file tracker sees every Python-level
        # open, FileIO included.
        with untracked(), io.FileIO(pyc, "rb") as fh:
            header = fh.read(16)
    except (OSError, ValueError, NotImplementedError):
        return False
    if len(header) < 16 or header[:4] != importlib.util.MAGIC_NUMBER:
        return False
    if int.from_bytes(header[4:8], "little") != 0:
        return False  # hash-based pyc: no timestamp to compare
    return int.from_bytes(header[8:12], "little") == (int(st.st_mtime) & 0xFFFFFFFF) and int.from_bytes(
        header[12:16], "little"
    ) == (st.st_size & 0xFFFFFFFF)


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def loaded_code_matches_disk(fn: object) -> bool:
    """False when *fn*'s source file was edited after this process loaded it.

    True whenever that cannot be shown: no file (a REPL, ``exec``, a notebook
    cell), a file its ``.pyc`` shows is the one imported (the common case; see
    `_pyc_proves_unchanged`), or a file whose compiled form still contains this
    function unchanged. Anything else is compiled, once per file version.
    """

    if isinstance(fn, type):
        # A class has no code of its own; its methods do, and an edit to any
        # of them is an edit to the class.
        return all(loaded_code_matches_disk(m) for m in class_functions(fn))
    code = getattr(fn, "__code__", None)
    if not isinstance(code, types.CodeType):
        return True
    path = code.co_filename
    if not path or path.startswith("<"):
        return True
    if not path.endswith((".py", ".pyw")):
        # Code compiled from something that is not a Python file -- a doc
        # page's fence, a template -- cannot be recompiled whole to compare,
        # and "does not compile" would read as "edited".
        return True
    try:
        if _pyc_proves_unchanged(path, os.stat(path)):
            return True
    except OSError:
        return True
    module = _compiled_module(path)
    if module is None:
        return False  # the file no longer compiles: it is not what runs
    live = bytecode_identity(fn)
    if live is None:
        return True
    qualname = getattr(code, "co_qualname", None)
    for candidate in _code_objects(module):
        if qualname is not None:
            if getattr(candidate, "co_qualname", None) != qualname:
                continue
        elif candidate.co_name != code.co_name:
            continue
        # Not ``types.FunctionType(candidate, {})``: that raises for a nested
        # function with free variables (it needs a closure), and only the
        # code is read anyway.
        probe = types.SimpleNamespace(__code__=candidate)
        if bytecode_identity(probe) == live:
            return True
    return False


def class_functions(cls: type) -> list[types.FunctionType]:
    """The plain functions defined directly in *cls*, unwrapping descriptors."""
    found = []
    for value in vars(cls).values():
        func = getattr(value, "__func__", value)  # staticmethod / classmethod
        if isinstance(value, property):
            func = value.fget
        if isinstance(func, types.FunctionType) and getattr(func, "_cash_cached", False):
            # A cached method's class attribute is cash's wrapper, whose globals
            # are cash's own: walked as the class's code, it reported
            # KEY-UNHASHABLE-GLOBAL for `Model.fit.ACTIVE_CONFIG`, a name in
            # no file of the user's. The user's function is inside.
            func = getattr(func, "__wrapped__", func)
        if isinstance(func, types.FunctionType):
            found.append(func)
    return found


def loaded_class_identity(cls: type) -> str | None:
    """A digest of *cls* from its LOADED methods, for when disk is not them."""
    try:
        parts = [cls.__qualname__]
        for func in sorted(class_functions(cls), key=lambda f: f.__qualname__):
            parts.append(f"{func.__qualname__}={bytecode_identity(func)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    except (AttributeError, TypeError, ValueError):
        return None
