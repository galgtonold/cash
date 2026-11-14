"""Is the code on disk still the code that is running?

A helper's digest is computed lazily, the first time a call needs it, from
the source on disk. If the file was edited after this process imported it
-- every deploy, every ``git pull`` under a long-running worker -- that
digest describes the new text while the old code runs, and a result of the
old code would be stored, and served after a restart, under the new code's
key. The checks here tell the two apart: from the module's ``.pyc`` when it
proves the import saw this file, else by what the file's text compiles to
(compiled once per text, and kept under the text's digest).

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
import sys
import types

from ._memo import COMPILED_MODULES, LruMemo
from ._paths import is_pseudo_filename
from ._user_attrs import module_file
from .process_start import process_start_time
from .source_norm import bytecode_identity
from .source_reading import (
    SETTLED_SECONDS,
    code_is_where_it_says,
    read_code_file,
    relocated_source_lines,
    stat_has_settled,
)
from .tracking.tracker_context import untracked

__all__ = [
    "class_functions",
    "loaded_class_identity",
    "loaded_code_matches_disk",
    "loaded_module_matches_disk",
    "relocated_definition",
]

#: path -> (mtime_ns, size, digest of the text, its compiled code, its lines).
#: The stat is -1 for a text read before it had settled: only its digest
#: (`_compiled_text`) finds it then.
_MODULE_CODE_CACHE: LruMemo[str, tuple[int, int, str, types.CodeType | None, tuple[str, ...]]] = LruMemo(
    COMPILED_MODULES
)


def _compiled_module(path: str) -> types.CodeType | None:
    """The whole file at *path*, compiled, cached per (path, mtime, size)."""
    return _compiled_module_and_lines(path)[0]


def _compiled_module_and_lines(path: str) -> tuple[types.CodeType | None, tuple[str, ...]]:
    """`_compiled_module`, with the lines of the text it compiled."""

    try:
        st = os.stat(path)
    except OSError:
        return None, ()
    cached = _MODULE_CODE_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[3], cached[4]
    settled = stat_has_settled(st)
    text = _read_text(path)
    if text is None:
        return None, ()
    code, lines = _compiled_text(path, text)
    if settled:
        _MODULE_CODE_CACHE[path] = (st.st_mtime_ns, st.st_size, text[1], code, lines)
    return code, lines


def _read_text(path: str) -> tuple[bytes, str] | None:
    """The bytes of *path* and their digest; None when it cannot be read."""
    try:
        source = read_code_file(path)
    except OSError:
        return None
    return source, hashlib.blake2b(source, digest_size=20).hexdigest()


def _compiled_text(path: str, text: tuple[bytes, str]) -> tuple[types.CodeType | None, tuple[str, ...]]:
    """*text* (`_read_text`) compiled as the file at *path*, with its lines;
    kept per path and found again by the text's digest."""
    source, digest = text
    cached = _MODULE_CODE_CACHE.get(path)
    if cached is not None and cached[2] == digest:
        return cached[3], cached[4]
    lines: tuple[str, ...] = ()
    try:
        code: types.CodeType | None = compile(source, path, "exec", dont_inherit=True)
        # As the import decodes it: its coding cookie, universal newlines.
        lines = tuple(io.StringIO(importlib.util.decode_source(source)).readlines())
    except (SyntaxError, ValueError):
        code = None
    _MODULE_CODE_CACHE[path] = (-1, -1, digest, code, lines)
    return code, lines


#: What a module file's text compiles to, kept on disk (`scan_store`) so a
#: fresh process need not compile it again to compare: the optimisation level
#: is part of it (the store's own key holds the Python and cash).
_RECORD_KIND = f"compiled-code-o{sys.flags.optimize}"
#: path -> (digest of the text, its record): read from the store once per text.
_RECORDS: LruMemo[str, tuple[str, dict]] = LruMemo(COMPILED_MODULES)


def _record(path: str, digest: str) -> dict:
    """The stored facts about the text of *path* that digests to *digest*:
    ``{"functions": {name: [bytecode identities]}, "pyc": [digests of .pyc
    bodies whose code is that text's]}``, each worked out from the text
    itself and only ever added to. Content-addressed, so as exact as
    compiling: a different text (a same-size save with the mtime put back)
    has a different digest and finds no record."""
    cached = _RECORDS.get(path)
    if cached is not None and cached[0] == digest:
        return cached[1]
    from .analysis import scan_store  # loaded_code is imported by modules scan_store's imports need

    data = scan_store.load_record(_RECORD_KIND, path, digest)
    functions = data.get("functions") if data is not None else None
    pyc = data.get("pyc") if data is not None else None
    record = {
        "functions": functions if isinstance(functions, dict) else {},
        "pyc": pyc if isinstance(pyc, list) else [],
    }
    _RECORDS[path] = (digest, record)
    return record


def _save_record(path: str, digest: str, record: dict) -> None:
    from .analysis import scan_store

    scan_store.save_record(_RECORD_KIND, path, digest, record)


def relocated_definition(
    code: types.CodeType, qualnames: list[str], paths: list[str]
) -> tuple[str, int, tuple[str, ...]] | None:
    """``(path, first line, the file's lines)`` of the definition in one of
    *paths* that compiles to *code* (`bytecode_identity`, which ignores line
    numbers), looked up under each of *qualnames*, then under its bare name
    when that is one definition only; None when no file holds it.

    For code compiled from other text than its file at its line: IPython's
    ``%autoreload`` compiles an edited function on its own (first line 1) and
    swaps the code into the old function object, a decorated one under the
    decorator's file. The file as it is now is what the code was compiled
    from, so its definition there is the text a fresh process keys it by.
    """
    live = bytecode_identity(types.SimpleNamespace(__code__=code))
    if live is None:
        return None
    for path in paths:
        module, lines = _compiled_module_and_lines(path)
        if module is None or not lines:
            continue
        index = _code_by_name(module)
        for name in qualnames:
            same = [c for c in index.get(name, ()) if bytecode_identity(types.SimpleNamespace(__code__=c)) == live]
            if same:
                # A name defined twice runs as the last definition.
                return path, same[-1].co_firstlineno, lines
        same = [
            c
            for c in index.get("name:" + code.co_name, ())
            if bytecode_identity(types.SimpleNamespace(__code__=c)) == live
        ]
        if len(same) == 1:
            return path, same[0].co_firstlineno, lines
    return None


#: Does ``st_ctime`` move with every write and ``utime``? Not on Windows,
#: where it is the creation time (`_pyc_postdates_source`).
_HAS_CHANGE_TIME = os.name != "nt"


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

    Unless the save put the old mtime back: ``os.utime``, ``cp -p``, ``tar``
    and ``rsync -t`` all do, and a same-size edit restored to the second the
    ``.pyc`` recorded matched the header again -- Python loaded the old
    bytecode in every later process, and cash, taking the ``.pyc`` as proof,
    keyed that old code's results by the new text (a result persisted, and
    served to the process that finally runs the new code). The change time
    no save can set back: on POSIX ``st_ctime`` moves with every write and
    every ``utime``, so a source whose ctime is not strictly older than the
    ``.pyc`` changed after it was compiled -- or within its tick, which no
    clock can order. On Windows ``st_ctime`` is the creation time, which an
    in-place save keeps (and robocopy, unzip and ``git checkout`` restore
    mtimes), so nothing there tells a put-back save from no save: the header
    is no proof, and the callers compare the ``.pyc``'s code, or the loaded
    code, with what the file's text compiles to: compiled once per text and
    kept on disk under the text's digest (`_record`), so a later process
    pays a read and a digest of the file, not a compile.
    """
    if not _HAS_CHANGE_TIME:
        return False
    if pyc_st.st_mtime - st.st_mtime <= SETTLED_SECONDS:
        return False
    return st.st_ctime_ns < pyc_st.st_mtime_ns


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

    path = module_file(module)
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
    text = _read_text(path)
    if text is None:
        return True
    # A .pyc body proven to be this text's code before (here or by an earlier
    # process) is not compiled again: both are compared by digest.
    record = _record(path, text[1])
    body_digest = hashlib.blake2b(body, digest_size=20).hexdigest()
    if body_digest in record["pyc"]:
        return True
    try:
        loaded = marshal.loads(body)
    except (EOFError, ValueError, TypeError):
        return True
    on_disk = _compiled_text(path, text)[0]
    if on_disk is None:
        return True  # nothing to reload to: the error belongs to the next import
    if loaded != on_disk:
        return False
    record["pyc"] = [*record["pyc"][-7:], body_digest]
    _save_record(path, text[1], record)
    return True


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


#: id of a compiled module -> (that module, its code objects by qualified
#: name). The module is held so the id cannot be reused while the entry lives.
_CODE_INDEX: LruMemo[int, tuple[types.CodeType, dict[str, list[types.CodeType]]]] = LruMemo(COMPILED_MODULES)


def _code_by_name(module: types.CodeType) -> dict[str, list[types.CodeType]]:
    """Every code object in the compiled *module*, by ``co_qualname`` (and by
    ``co_name`` under a ``"name:"`` prefix, for Pythons without qualnames).
    Built once per compiled file version, so checking each of a module's
    functions costs a lookup, not a walk of the whole module."""
    cached = _CODE_INDEX.get(id(module))
    if cached is not None and cached[0] is module:
        return cached[1]
    index: dict[str, list[types.CodeType]] = {}
    for candidate in _code_objects(module):
        qualname = getattr(candidate, "co_qualname", None)
        if qualname is not None:
            index.setdefault(qualname, []).append(candidate)
        index.setdefault("name:" + candidate.co_name, []).append(candidate)
    _CODE_INDEX[id(module)] = (module, index)
    return index


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
    if not path or is_pseudo_filename(path):
        return True
    if not path.endswith((".py", ".pyw")):
        # Code compiled from something that is not a Python file -- a doc
        # page's fence, a template -- cannot be recompiled whole to compare,
        # and "does not compile" would read as "edited".
        return True
    if not code_is_where_it_says(code):
        # Compiled from other text than the file holds at its line
        # (`%autoreload` swaps in code compiled from the function alone): the
        # file it is defined in holds it now, or nothing does.
        return relocated_source_lines(fn, code) is not None
    try:
        if _pyc_proves_unchanged(path, os.stat(path)):
            return True
    except OSError:
        return True
    # Compared through the text's record (`_record`): the bytecode identities
    # of its code objects under this function's name, worked out from the
    # text once -- here or by an earlier process -- and found again by the
    # text's digest. Reading and digesting the file costs a fraction of
    # compiling it, which a process paid for every helper file it checked.
    text = _read_text(path)
    if text is None:
        return False  # gone between the stat and the read: not what runs
    record = _record(path, text[1])
    name = getattr(code, "co_qualname", None)
    if name is None:
        name = "name:" + code.co_name
    functions = record["functions"]
    known = functions.get(name, False)
    if known is not None and not isinstance(known, list):
        module = _compiled_text(path, text)[0]
        if module is None:
            known = None
        else:
            # Not ``types.FunctionType(candidate, {})``: that raises for a
            # nested function with free variables (it needs a closure), and
            # only the code is read anyway.
            known = [
                bytecode_identity(types.SimpleNamespace(__code__=candidate))
                for candidate in _code_by_name(module).get(name, ())
            ]
        functions[name] = known
        _save_record(path, text[1], record)
    if known is None:
        return False  # the file no longer compiles: it is not what runs
    live = bytecode_identity(fn)
    if live is None:
        return True
    return live in known


def class_functions(cls: type) -> list[types.FunctionType]:
    """The plain functions defined directly in *cls*, unwrapping descriptors."""
    found = []
    for value in vars(cls).values():
        func = getattr(value, "__func__", value)  # staticmethod / classmethod
        if isinstance(value, property):
            func = value.fget
        if type(func) is types.FunctionType and getattr(func, "_cash_cached", False):
            # A cached method's class attribute is cash's wrapper, whose globals
            # are cash's own: walked as the class's code, it reported
            # KEY-UNHASHABLE-GLOBAL for `Model.fit.ACTIVE_CONFIG`, a name in
            # no file of the user's. The user's function is inside.
            func = getattr(func, "__wrapped__", func)
        if type(func) is types.FunctionType:
            found.append(func)
    return found


def code_objects_of(owner: object) -> tuple:
    """The code objects *owner* runs as itself, in a fixed order: each method a
    class defines (as `class_functions` finds them), or a function's code and
    the code down its ``__wrapped__`` chain. What a digest memoised per object
    (not per code object) must still match: IPython's ``%autoreload`` keeps
    the class and the function objects and swaps each one's ``__code__`` (and
    a wrapper's ``__dict__``) in place.

    Asked per helper on every call, so written for speed: about 0.2 us for a
    plain function, 1-2 us for a class of a few methods."""
    if type(owner) is types.FunctionType:
        attrs = owner.__dict__
        if not attrs or "__wrapped__" not in attrs:
            return (owner.__code__,)
    elif isinstance(owner, type):
        found = []
        try:
            for value in vars(owner).values():
                kind = type(value)
                if kind is types.FunctionType:
                    func = value
                elif kind is property:
                    func = value.fget
                elif kind is staticmethod or kind is classmethod:
                    func = value.__func__
                elif kind in _PLAIN_ATTRIBUTE_TYPES:
                    continue
                else:
                    func = getattr(value, "__func__", value)
                if type(func) is types.FunctionType:
                    attrs = func.__dict__
                    if attrs and attrs.get("_cash_cached", False):
                        func = attrs.get("__wrapped__", func)
                        if type(func) is not types.FunctionType:
                            continue
                    found.append(func.__code__)
        except (AttributeError, TypeError):
            return ()
        return tuple(found)
    found = []
    seen: set[int] = set()
    while owner is not None and id(owner) not in seen:
        seen.add(id(owner))
        code = getattr(owner, "__code__", None)
        if not isinstance(code, types.CodeType):
            break
        found.append(code)
        owner = getattr(owner, "__wrapped__", None) if type(owner) is types.FunctionType else None
    return tuple(found)


#: Class attribute types that hold no function (`code_objects_of`).
_PLAIN_ATTRIBUTE_TYPES = frozenset(
    {str, int, float, bool, type(None), tuple, dict, types.MappingProxyType, types.GetSetDescriptorType}
)


def loaded_class_identity(cls: type) -> str | None:
    """A digest of *cls* from its LOADED methods, for when disk is not them."""
    try:
        parts = [cls.__qualname__]
        for func in sorted(class_functions(cls), key=lambda f: f.__qualname__):
            parts.append(f"{func.__qualname__}={bytecode_identity(func)}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    except (AttributeError, TypeError, ValueError):
        return None
