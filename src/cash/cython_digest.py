"""The identity of a Cython function: the Cython code it can reach, not the whole binary.

A function of a user's compiled extension has no Python source, so it was
keyed by the digest of the built ``.so`` / ``.pyd`` (`extension_file_digest`).
That is safe, but a binary holds every function of its ``.pyx``: an edit to
any one of them moved the key of every cached function that called any
other, and each re-ran for nothing.

A Cython function (a "cyfunction") names its ``.pyx`` and first line in its
``__code__``. When that file is the one the binary was built from, the
function is keyed by what it can reach in it instead
(`cython_function_digest`):

- the Cython version the binary was built with;
- the module header: everything at the top level that is not a function or a
  class (imports, cimports, compiler directives, ctypedefs, structs, globals
  and constants), with the directive comments;
- the top-level blocks (``def`` / ``cpdef`` / ``cdef`` functions, ``cdef``
  classes and classes) it reaches by name, transitively, starting from its
  own block and from the header;
- every ``.pxd`` / ``.pxi`` the module uses (its own ``.pxd``, a local
  ``cimport``, an ``include``), and the ``.pxd`` of a library it cimports;
- the build configuration (``setup.py``, ``setup.cfg``, ``pyproject.toml``)
  next to the package, in a canonical form without the metadata that
  cannot change what is compiled (`canonical_build_config`): a version
  bump does not move the key of every helper.
- for each local module it cimports (a ``cdef`` function called through
  its ``.pxd`` runs from THAT module's binary): the blocks the names in use
  reach in its ``.pyx``, by the same rules and through its own cimports, or
  its whole binary where its ``.pyx`` cannot be used.

The whole-binary key (`cimported_binaries_digest`) folds in the binaries of
the modules it cimports for the same reason.

Blocks and names come from Cython's own parser and tokenizer; a block is
keyed by its token stream, so a comment, a blank line or a line shift does
not move it. Line endings are made ``\n`` before tokenizing, as Cython's
own reader does, so a CRLF checkout keys like an LF one.

``None`` -- the caller keeps the whole-binary digest -- whenever the parts
cannot be trusted to be what the binary holds: no Cython, no ``.pyx``, a
``.pyx`` (or any file keyed above) newer than the binary, a function of the
binary whose name and first line do not match a block of the parsed
``.pyx`` (a stale or foreign build), a parse failure, a ``cdef extern``
in the module's own files from anything but a header of numpy's or
Python's own include folder (which is keyed by its content), extra
``sources`` or compile flags in the build configuration or a
``# distutils:`` comment (macros are keyed text and allowed), a build
system it does not read (meson, CMake), no ``setup.py`` /
``pyproject.toml`` at all (``cythonize -i``, pyximport: the directives
and flags of such a build are recorded nowhere), or a ``cimport`` it
cannot find. Paths are compared in one spelling (``os.path.normcase``),
so a Windows ``.pyd`` and a differently cased path match.

Parsing a module with Cython costs about a second (importing the compiler
alone a fifth of one), which every fresh process paid for every module it
reached. So a module's analysis is stored on disk (`cash.analysis.scan_store`,
one record per built file) with everything it was worked out from: the
binary's content digest, the Cython version, the Python and its
``sys.path``; the content digest of every file it read (the ``.pyx``, its
``.pxd`` / ``.pxi`` files, the cimported ``.pxd`` files, the headers, the
build configuration); and the answer of every check for a file that may or
may not be there (a cimport's candidates, a ``meson.build``). A later process
uses the record only when all of them are the same now -- each file is read
and digested again (a build configuration file in its canonical form),
never trusted by its timestamp -- and then never imports the Cython
compiler. An analysis that failed for any reason but a missing file is
stored as a fallback too, so it is not paid again in every process. The checks that do not depend on content (a keyed
file newer than the binary, every function of the binary where the
analysis places it) run every time.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import importlib.machinery
import json
import os
import sys
import threading
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ._user_attrs import module_file
from .tracking.tracker_context import untracked

__all__ = ["canonical_build_config", "cimported_binaries_digest", "cython_function_digest"]

_CYFUNCTION_TYPES = frozenset({"cython_function_or_method", "fused_cython_function"})

#: Top-level statements that are blocks keyed only when reached by name.
_BLOCK_NODES = frozenset({"DefNode", "CFuncDefNode", "CClassDefNode", "PyClassDefNode"})

#: The files next to the package that say how it is built.
_CONFIG_FILES = ("setup.py", "setup.cfg", "pyproject.toml")

#: Files of a build system whose flags are not read here.
_UNREAD_BUILD_FILES = ("meson.build", "CMakeLists.txt")

#: Extension arguments that add code or change how it compiles.
_FLAG_ARGUMENTS = frozenset(
    {
        "extra_compile_args",
        "extra_link_args",
        "extra_objects",
        "depends",
    }
)

#: The same, as a ``setup.cfg`` / ``pyproject.toml`` spells them.
_CONFIG_FLAG_WORDS = tuple(sorted({w for a in _FLAG_ARGUMENTS for w in (a, a.replace("_", "-"))} | {"sources"}))

#: ``# distutils:`` comment settings that do not add code or flags.
#: Macros are allowed: they are part of the keyed text (the ``.pyx``
#: header, ``setup.py``), and change only with it -- numpy's
#: ``NPY_NO_DEPRECATED_API``, say, which hdbscan sets in every module.
_HARMLESS_DISTUTILS = frozenset({"language", "include_dirs", "define_macros", "undef_macros"})

#: ``setup()`` keywords (and ``setup.cfg`` ``[metadata]`` keys) that describe
#: the distribution and cannot change what is compiled. Only the keyword of
#: a call named ``setup`` is dropped: a value it shares with anything else
#: (a version passed to ``define_macros`` too) stays keyed there.
_SETUP_METADATA = frozenset(
    {
        "version",
        "description",
        "long_description",
        "long_description_content_type",
        "author",
        "author_email",
        "maintainer",
        "maintainer_email",
        "url",
        "project_urls",
        "license",
        "classifiers",
        "keywords",
        "platforms",
    }
)

#: PEP 621 ``[project]`` keys that are metadata only. ``name``,
#: ``requires-python``, the dependencies and ``dynamic`` stay keyed, as do
#: ``[build-system]`` and every ``[tool.*]`` table.
_PROJECT_METADATA = frozenset(
    {
        "version",
        "description",
        "readme",
        "authors",
        "maintainers",
        "license",
        "license-files",
        "keywords",
        "classifiers",
        "urls",
    }
)


class _Fallback(Exception):
    """The per-function key cannot be trusted; the whole binary is keyed."""


@dataclass
class _Block:
    names: frozenset[str]
    path: str
    start: int
    end: float
    digest: str = ""
    refs: frozenset[str] = frozenset()


@dataclass
class _Module:
    """What one built module's ``.pyx`` says, parsed once per loaded module."""

    pyx: str
    blocks: list[_Block]
    by_name: dict[str, list[_Block]]
    header_digest: str
    header_refs: frozenset[str]
    files_digest: str
    keyed_files: tuple[str, ...]
    #: The local ``.pxd`` files of other modules it cimports.
    cimported: tuple[str, ...] = ()
    #: The files that must not be newer than the binary.
    newer_keyed: tuple[str, ...] = ()
    digests: dict[str, str | None] = field(default_factory=dict)


#: The layout of a stored analysis; a change makes every older one a miss.
_STORE_FORMAT = 2
_STORE_KIND = f"cython-module-v{_STORE_FORMAT}"


class _Inputs:
    """What an analysis read: each file's content digest, and the answer of
    each check for whether a file or folder is there."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.probes: dict[tuple[str, str], bool] = {}


#: The inputs of the analysis running now (under `_LOCK`), if any.
_RECORDING: list[_Inputs] = []


def _isfile(path: str) -> bool:
    found = os.path.isfile(path)
    if _RECORDING:
        _RECORDING[-1].probes[("f", path)] = found
    return found


def _isdir(path: str) -> bool:
    found = os.path.isdir(path)
    if _RECORDING:
        _RECORDING[-1].probes[("d", path)] = found
    return found


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        data = fh.read()
    if _RECORDING:
        _RECORDING[-1].files[path] = hashlib.sha256(data).hexdigest()
    return data


#: The prefix of a recorded digest taken of a file's canonical build
#: configuration form rather than of its bytes.
_CONFIG_DIGEST = "config:"


def _config_digest(path: str, data: bytes) -> str:
    return _CONFIG_DIGEST + hashlib.sha256(canonical_build_config(path, data)).hexdigest()


def _read_config(path: str) -> bytes:
    """The canonical form (`canonical_build_config`) of the build
    configuration file *path*, recorded by its digest."""
    with open(path, "rb") as fh:
        data = fh.read()
    form = canonical_build_config(path, data)
    if _RECORDING:
        _RECORDING[-1].files[path] = _CONFIG_DIGEST + hashlib.sha256(form).hexdigest()
    return form


def canonical_build_config(path: str, data: bytes) -> bytes:
    """*data*, the bytes of the build configuration file *path*, without
    what cannot change the compiled result: the distribution metadata
    (version, description, authors, classifiers, ...) of ``pyproject.toml``'s
    ``[project]``, of ``setup.cfg``'s ``[metadata]`` and of the keywords of
    a ``setup(...)`` call in ``setup.py``, and the comments and layout of
    each. Everything else stays. A file that does not parse is its bytes."""
    name = os.path.basename(path)
    try:
        if name == "pyproject.toml":
            return _canonical_pyproject(data)
        if name == "setup.cfg":
            return _canonical_setup_cfg(data)
        if name == "setup.py":
            return _canonical_setup_py(data)
    except Exception:  # noqa: BLE001 - what does not parse is keyed by its bytes
        pass
    return data


def _canonical_pyproject(data: bytes) -> bytes:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # cash depends on tomli there
        import tomli as tomllib

    doc = tomllib.loads(data.decode("utf-8"))
    project = doc.get("project")
    if isinstance(project, dict):
        doc["project"] = {k: v for k, v in project.items() if k not in _PROJECT_METADATA}
    return b"toml:" + json.dumps(doc, sort_keys=True, default=repr).encode("utf-8")


def _canonical_setup_cfg(data: bytes) -> bytes:
    import configparser

    parser = configparser.RawConfigParser(strict=True, delimiters=("=", ":"))
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(data.decode("utf-8"))
    doc = {}
    for section in parser.sections():
        items = dict(parser.items(section, raw=True))
        if section == "metadata":
            items = {k: v for k, v in items.items() if k.replace("-", "_").lower() not in _SETUP_METADATA}
        doc[section] = items
    return b"cfg:" + json.dumps(doc, sort_keys=True).encode("utf-8")


def _canonical_setup_py(data: bytes) -> bytes:
    tree = ast.parse(data)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name == "setup":
            node.keywords = [k for k in node.keywords if k.arg not in _SETUP_METADATA]
    options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    return b"py:" + ast.dump(tree, annotate_fields=True, include_attributes=False, **options).encode("utf-8")


def _source_text(data: bytes) -> str:
    """*data* decoded, with ``\r\n`` and ``\r`` made ``\n`` as Cython's
    own reader makes them: its scanner refuses a bare ``\r``."""
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


#: binary path -> (the module loaded from it, its analysis or None).
_MODULES: dict[str, tuple[types.ModuleType, _Module | None]] = {}
#: binary path -> (the module loaded from it, its content digest).
_BINARIES: dict[str, tuple[types.ModuleType, str]] = {}
_LOCK = threading.RLock()


def is_cyfunction(fn: object) -> bool:
    """Is *fn* a function Cython compiled?"""
    kind = type(fn)
    return kind.__name__ in _CYFUNCTION_TYPES and str(getattr(kind, "__module__", "")).startswith("_cython_")


def cython_function_digest(fn: object) -> str | None:
    """The digest of what *fn*, a Cython function of a user's built extension,
    can reach in its ``.pyx`` (see the module docstring), or ``None`` when that
    cannot be trusted and the whole built file must stand for it. Never raises."""
    fn = getattr(fn, "__func__", fn) if isinstance(fn, types.MethodType) else fn
    if not is_cyfunction(fn):
        return None
    code = getattr(fn, "__code__", None)
    if not isinstance(code, types.CodeType) or not code.co_filename.lower().endswith(".pyx"):
        return None
    module = sys.modules.get(getattr(fn, "__module__", None) or "")
    binary = module_file(module)
    if module is None or not isinstance(binary, str):
        return None
    if not _is_extension(binary):
        return None
    with _LOCK:
        analysis = _analysis_of(module, binary, code.co_filename)
        if analysis is None:
            return None
        qualname = str(getattr(fn, "__qualname__", "") or "")
        key = f"{code.co_filename}:{code.co_firstlineno}:{qualname}"
        if key not in analysis.digests:
            try:
                with untracked():
                    analysis.digests[key] = _function_digest(analysis, fn, code, qualname)
            except Exception:  # noqa: BLE001 - any failure keys the whole binary
                analysis.digests[key] = None
        return analysis.digests[key]


def cimported_binaries_digest(fn: object) -> str | None:
    """For the whole-binary key of *fn*, a Cython function of a user's
    build: the digest of every user-built module its module cimports,
    transitively, or None for a function that is not Cython's.

    A ``cdef`` function of another module, called through its ``.pxd``,
    runs from THAT module's binary: rebuilding it after an edit leaves the
    caller's binary byte for byte the same. The cimports are read from the
    ``.pyx`` with Cython's tokenizer; when that cannot be done, every
    user-built module loaded from the caller's package folder stands in.
    Never raises.
    """
    fn = getattr(fn, "__func__", fn) if isinstance(fn, types.MethodType) else fn
    if not is_cyfunction(fn):
        return None
    module = sys.modules.get(getattr(fn, "__module__", None) or "")
    binary = module_file(module)
    if module is None or not isinstance(binary, str):
        return None
    code = getattr(fn, "__code__", None)
    co_filename = code.co_filename if isinstance(code, types.CodeType) else ""
    with _LOCK:
        try:
            with untracked():
                parts = sorted(_whole_cimports(module, binary, co_filename, {_norm(binary)}))
        except Exception:  # noqa: BLE001 - every loaded sibling build stands in
            with untracked():
                parts = sorted(_sibling_binaries(module, binary))
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _analysis_of(module: types.ModuleType, binary: str, co_filename: str, pyx: str | None = None) -> _Module | None:
    """`_analyse`, memoised per loaded module; None where it falls back."""
    cached = _MODULES.get(binary)
    if cached is not None and cached[0] is module:
        return cached[1]
    try:
        with untracked():
            analysis = _stored_or_analysed(module, binary, co_filename, pyx)
            if analysis is not None:
                # What a stored record cannot say: the timestamps of today's
                # files against the binary, and that the loaded binary's
                # functions are where the analysis places them.
                _refuse_newer_than(analysis.newer_keyed, binary)
                _check_built_from(analysis, module)
    except _Fallback:
        analysis = None
    except Exception:  # noqa: BLE001 - any failure keys the whole binary
        analysis = None
    _MODULES[binary] = (module, analysis)
    return analysis


def _stored_or_analysed(module: types.ModuleType, binary: str, co_filename: str, pyx: str | None) -> _Module | None:
    """`_analyse`, or the analysis an earlier process stored for the same
    inputs (see the module docstring); None where it falls back."""
    from .analysis import scan_store, tree_memo
    from .install_paths import is_user_path

    if not is_user_path(binary):
        raise _Fallback("not the user's build")
    try:
        import Cython
    except Exception as e:
        raise _Fallback("no Cython") from e
    key = hashlib.sha256(
        json.dumps(
            [
                _STORE_KIND,
                _binary_digest(module, binary),
                module.__name__,
                co_filename,
                pyx,
                str(getattr(Cython, "__version__", "")),
                os.path.abspath(str(getattr(Cython, "__file__", ""))),
                sys.prefix,
                sys.exec_prefix,
                list(sys.path),
            ]
        ).encode("utf-8")
    ).hexdigest()
    slot = f"{_norm(binary)}\0{co_filename}\0{pyx}"
    stored = scan_store.load_record(_STORE_KIND, slot, key)
    if stored is not None and _inputs_unchanged(stored):
        try:
            analysis = _module_from_json(stored["analysis"])
        except (KeyError, TypeError, ValueError):
            analysis = None
        else:
            if tree_memo.VERIFY:
                fresh, _inputs = _analyse_recorded(module, binary, co_filename, pyx)
                if _module_to_json(fresh) != _module_to_json(analysis):
                    raise scan_store.StoredScanMismatch(f"stored Cython analysis of {binary} differs from a new one")
            return analysis
    analysis, inputs = _analyse_recorded(module, binary, co_filename, pyx)
    scan_store.save_record(
        _STORE_KIND,
        slot,
        key,
        {
            "files": sorted(inputs.files.items()),
            "probes": sorted([kind, path, found] for (kind, path), found in inputs.probes.items()),
            "analysis": _module_to_json(analysis),
        },
    )
    return analysis


def _analyse_recorded(
    module: types.ModuleType, binary: str, co_filename: str, pyx: str | None
) -> tuple[_Module | None, _Inputs]:
    """`_analyse` and what it read. A `_Fallback` is an answer (None) of
    the inputs it read; any other failure is raised, never stored."""
    inputs = _Inputs()
    _RECORDING.append(inputs)
    try:
        analysis = _analyse(module, binary, co_filename, pyx)
    except _Fallback:
        analysis = None
    except OSError:
        # A file that could not be read is not among the recorded inputs,
        # so a stored answer could outlive its coming back.
        raise
    except Exception as e:  # noqa: BLE001 - a parse failure, stored as a fallback
        # The same inputs fail the same way: stored, so that every process
        # does not pay for it again. A file the parser read on its own (an
        # include) is recorded where the error names it.
        _record_error_file(e, inputs)
        analysis = None
    finally:
        _RECORDING.pop()
    return analysis, inputs


def _record_error_file(error: BaseException, inputs: _Inputs) -> None:
    position = getattr(error, "position", None)
    source = position[0] if isinstance(position, tuple) and position else None
    path = getattr(source, "filename", None)
    if isinstance(path, str) and path not in inputs.files:
        try:
            with open(path, "rb") as fh:
                inputs.files[path] = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            pass


def _inputs_unchanged(stored: dict) -> bool:
    """Is every file a stored analysis read the same now, by content, and
    every file check answered the same?"""
    try:
        for path, digest in stored["files"]:
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
                if str(digest).startswith(_CONFIG_DIGEST):
                    if _config_digest(path, data) != digest:
                        return False
                elif hashlib.sha256(data).hexdigest() != digest:
                    return False
            except OSError:
                return False
        for kind, path, found in stored["probes"]:
            check = os.path.isfile if kind == "f" else os.path.isdir
            if check(path) != found:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _module_to_json(analysis: _Module | None) -> dict | None:
    if analysis is None:
        return None
    return {
        "pyx": analysis.pyx,
        "blocks": [
            {
                "names": sorted(b.names),
                "path": b.path,
                "start": b.start,
                "end": None if b.end == float("inf") else b.end,
                "digest": b.digest,
                "refs": sorted(b.refs),
            }
            for b in analysis.blocks
        ],
        "header_digest": analysis.header_digest,
        "header_refs": sorted(analysis.header_refs),
        "files_digest": analysis.files_digest,
        "keyed_files": list(analysis.keyed_files),
        "cimported": list(analysis.cimported),
        "newer_keyed": list(analysis.newer_keyed),
    }


def _module_from_json(data: dict | None) -> _Module | None:
    if data is None:
        return None
    blocks = [
        _Block(
            names=frozenset(b["names"]),
            path=str(b["path"]),
            start=int(b["start"]),
            end=float("inf") if b["end"] is None else b["end"],
            digest=str(b["digest"]),
            refs=frozenset(b["refs"]),
        )
        for b in data["blocks"]
    ]
    by_name: dict[str, list[_Block]] = {}
    for block in blocks:
        for name in block.names:
            by_name.setdefault(name, []).append(block)
    return _Module(
        pyx=str(data["pyx"]),
        blocks=blocks,
        by_name=by_name,
        header_digest=str(data["header_digest"]),
        header_refs=frozenset(data["header_refs"]),
        files_digest=str(data["files_digest"]),
        keyed_files=tuple(data["keyed_files"]),
        cimported=tuple(data["cimported"]),
        newer_keyed=tuple(data["newer_keyed"]),
    )


def _reach(analysis: _Module, own: Iterable[_Block], names: Iterable[str]) -> tuple[list[_Block], set[str]]:
    """The blocks reached from *own* and *names* (and the header), and every
    name they use."""
    reached: dict[int, _Block] = {id(b): b for b in own}
    pending = set(analysis.header_refs) | set(names)
    for block in reached.values():
        pending |= block.refs
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for block in analysis.by_name.get(name, ()):
            if id(block) not in reached:
                reached[id(block)] = block
                pending |= block.refs - seen
    return list(reached.values()), seen


def _function_digest(analysis: _Module, fn: object, code: types.CodeType, qualname: str) -> str | None:
    own = _own_blocks(analysis, code, qualname)
    if own is None:
        return None
    reached, names = _reach(analysis, own, ())
    parts = [
        "cash-cython-function-v2",
        type(fn).__module__,
        str(getattr(fn, "__module__", "")),
        qualname,
        analysis.header_digest,
        analysis.files_digest,
        *sorted(b.digest for b in reached),
        *_cimported_parts(analysis, names, {analysis.pyx}),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _cimported_parts(analysis: _Module, names: set[str], visited: set[str]) -> list[str]:
    """What the modules *analysis* cimports contribute: for each, the blocks
    the names in use reach in its ``.pyx`` (by the same rules, through its
    own cimports), or its whole binary where its ``.pyx`` cannot be used."""
    parts = []
    for pxd in analysis.cimported:
        pyx = os.path.splitext(pxd)[0] + ".pyx"
        if pyx in visited:
            continue
        visited.add(pyx)
        loaded = _module_for_pxd(pxd)
        label = os.path.basename(pyx)
        if loaded is None:
            if os.path.isfile(pyx):
                # Not loaded, so nothing of it runs from its binary; its
                # source still stands for it.
                with open(pyx, "rb") as fh:
                    parts.append(f"{label}:source:{hashlib.sha256(fh.read()).hexdigest()}")
            continue
        module, binary = loaded
        sub = _analysis_of(module, binary, os.path.basename(pyx), pyx if os.path.isfile(pyx) else None)
        if sub is None:
            whole = sorted(_whole_cimports(module, binary, os.path.basename(pyx), {_norm(binary)}))
            parts.append(f"{label}:built:{_binary_digest(module, binary)}:{','.join(whole)}")
            continue
        reached, sub_names = _reach(sub, (), names)
        nested = _cimported_parts(sub, sub_names, visited)
        digest = hashlib.sha256(
            "\x1f".join([sub.header_digest, sub.files_digest, *sorted(b.digest for b in reached), *nested]).encode()
        ).hexdigest()
        parts.append(f"{label}:cython:{digest}")
    return sorted(parts)


def _module_for_pxd(pxd: str) -> tuple[types.ModuleType, str] | None:
    """The loaded extension module built from the ``.pyx`` beside *pxd*."""
    folder = os.path.dirname(_norm(pxd))
    stem = os.path.splitext(os.path.basename(_norm(pxd)))[0]
    for module in list(sys.modules.values()):
        path = module_file(module) if isinstance(module, types.ModuleType) else None
        if not isinstance(path, str) or not _is_extension(path):
            continue
        full = _norm(path)
        if os.path.basename(full).split(".", 1)[0] == stem and os.path.dirname(full) == folder:
            return module, path
    return None


def _binary_digest(module: types.ModuleType, binary: str) -> str:
    cached = _BINARIES.get(binary)
    if cached is not None and cached[0] is module:
        return cached[1]
    from .pe_digest import extension_bytes_digest

    with open(binary, "rb") as fh:
        digest = extension_bytes_digest(fh.read())
    _BINARIES[binary] = (module, digest)
    return digest


def _whole_cimports(module: types.ModuleType, binary: str, co_filename: str, visited: set[str]) -> list[str]:
    """``name=digest`` of the binary of every loaded module *module*'s ``.pyx``
    cimports, transitively, read from its tokens. Raises where the
    cimports cannot be read."""
    pyx = _find_pyx(module, binary, co_filename)
    package_root = _package_root(module.__name__, pyx)
    includes = _cython_includes()
    texts = [pyx]
    own = os.path.splitext(pyx)[0] + ".pxd"
    if os.path.isfile(own):
        texts.append(own)
    pxds: list[str] = []
    done: set[str] = set()
    while texts:
        path = texts.pop()
        if path in done:
            continue
        done.add(path)
        with open(path, "rb") as fh:
            toks = _tokens(path, _source_text(fh.read()))
        for i, (sy, _text, _line) in enumerate(toks):
            if sy == "include":
                follows = [t for s_, t, _l in toks[i + 1 : i + 4] if s_ == "CHARS"]
                if not follows:
                    raise _Fallback("an include not understood")
                texts.append(os.path.join(os.path.dirname(path), follows[0]))
        for name, level in _token_cimports(toks):
            for target in _resolve_cimport(name, level, path, package_root, includes):
                if target != own and target not in pxds and _is_local(target, package_root):
                    pxds.append(target)
                    texts.append(target)
    parts = []
    for pxd in pxds:
        loaded = _module_for_pxd(pxd)
        if loaded is None:
            continue
        sub, sub_binary = loaded
        key = _norm(sub_binary)
        if key in visited:
            continue
        visited.add(key)
        parts.append(f"{os.path.basename(pxd)}={_binary_digest(sub, sub_binary)}")
        parts.extend(_whole_cimports(sub, sub_binary, os.path.basename(os.path.splitext(pxd)[0] + ".pyx"), visited))
    return parts


def _sibling_binaries(module: types.ModuleType, binary: str) -> list[str]:
    """``path=digest`` of every user-built module loaded from the folder of
    *module*'s package: what it might cimport, when that cannot be read."""
    from .install_paths import is_user_path

    root = os.path.dirname(os.path.abspath(binary))
    for _ in range(module.__name__.count(".")):
        root = os.path.dirname(root)
    root = os.path.normcase(root) + os.sep
    parts = []
    for other in list(sys.modules.values()):
        path = module_file(other) if isinstance(other, types.ModuleType) else None
        if not isinstance(path, str) or not _is_extension(path) or other is module:
            continue
        full = os.path.abspath(path)
        if not os.path.normcase(full).startswith(root) or not is_user_path(full):
            continue
        try:
            parts.append(f"{os.path.relpath(full, root)}={_binary_digest(other, full)}")
        except OSError:
            parts.append(f"{os.path.relpath(full, root)}=unreadable")
    return parts


def _package_root(module_name: str, pyx: str) -> str:
    root = os.path.dirname(pyx)
    for _ in range(module_name.count(".")):
        root = os.path.dirname(root)
    return root


def _cython_includes() -> str:
    import Cython

    return os.path.join(os.path.dirname(os.path.abspath(Cython.__file__)), "Includes")


def _own_blocks(analysis: _Module, code: types.CodeType, qualname: str) -> list[_Block] | None:
    """The top-level block(s) *code* is in, or None when the parse does not
    place it where its name says (a ``.pyx`` that is not the one built)."""
    head = qualname.split(".", 1)[0]
    path = _code_path(analysis, code)
    if path is None:
        return None
    line = code.co_firstlineno
    own = [b for b in analysis.blocks if b.path == path and b.start <= line < b.end]
    if not own:
        # Defined at the top level outside any def or class (inside an
        # ``if``, say): its code is in the header, keyed for every function.
        return [] if head and not analysis.by_name.get(head) else None
    if not head.startswith("<") and not any(head in b.names for b in own):
        return None
    return own


def _code_path(analysis: _Module, code: types.CodeType) -> str | None:
    name = os.path.normcase(os.path.normpath(code.co_filename))
    for path in analysis.keyed_files:
        if os.path.normcase(path).endswith(name) and os.path.basename(os.path.normcase(path)) == os.path.basename(name):
            return path
    return None


# -- building the analysis ---------------------------------------------------


def _analyse(module: types.ModuleType, binary: str, co_filename: str, pyx: str | None = None) -> _Module:
    """What *module*'s ``.pyx`` says (`_Module`). Raises `_Fallback` where
    the per-function key cannot be trusted. It reads files only through
    `_read`, `_isfile` and `_isdir`, which record them for the stored
    analysis; the timestamps and the binary's functions are checked by
    the caller, every time."""
    try:
        import Cython  # noqa: F401 - only whether it imports
        from Cython.Compiler.Main import Context  # noqa: F401 - only whether it imports
    except Exception as e:
        raise _Fallback("no Cython") from e
    if pyx is None:
        pyx = _find_pyx(module, binary, co_filename)
    pyx_text = _source_text(_read(pyx))
    distutils = _refuse_distutils_settings(pyx_text)
    tree = _parse(module.__name__, pyx)
    stats = list(_top_level(tree.body))
    files = [pyx]
    for stat in stats:
        path = _node_path(stat)
        if path not in files:
            files.append(path)
    texts = {pyx: pyx_text}
    for path in files[1:]:
        texts[path] = _source_text(_read(path))
    tokens = {path: _tokens(path, texts[path]) for path in files}
    headers: list[str] = []
    for path in files:
        headers.extend(_extern_headers(tokens[path], path))
    blocks = _blocks(stats, tokens)
    header: list[str] = [repr(sorted(getattr(tree, "directive_comments", {}).items())), distutils]
    header_refs: set[str] = set()
    owned = {path: _owner_map(tokens[path], [b for b in blocks if b.path == path]) for path in files}
    for path in files:
        toks = tokens[path]
        per_block: dict[int, list[tuple[str, str]]] = {}
        header.append(f"file:{os.path.basename(path)}")
        for i, (sy, text, _line) in enumerate(toks):
            owner = owned[path][i]
            if owner is None:
                header.append(f"{sy}\x1e{text}")
                if sy in ("IDENT", "CHARS"):
                    header_refs.add(text)
            else:
                per_block.setdefault(owner, []).append((sy, text))
        for index, block in enumerate(b for b in blocks if b.path == path):
            body = per_block.get(index, [])
            block.digest = hashlib.sha256(
                "\x1f".join(f"{sy}\x1e{text}" for sy, text in body).encode("utf-8")
            ).hexdigest()
            block.refs = frozenset(text for sy, text in body if sy in ("IDENT", "CHARS"))
    by_name: dict[str, list[_Block]] = {}
    for block in blocks:
        for name in block.names:
            by_name.setdefault(name, []).append(block)
    root, configs = _build_config(os.path.dirname(pyx))
    pxds = _used_pxds(module.__name__, pyx, tree, files, root, headers)
    keyed = [*files, *configs, *(p for p, local in pxds if local)]
    file_parts = []
    for path in configs:
        file_parts.append(f"{_label(path, root)}={hashlib.sha256(_read_config(path)).hexdigest()}")
    for path in [*(p for p, _ in pxds), *sorted(set(headers))]:
        digest = hashlib.sha256(_read(path)).hexdigest()
        file_parts.append(f"{_label(path, root)}={digest}")
    analysis = _Module(
        pyx=pyx,
        blocks=blocks,
        by_name=by_name,
        header_digest=hashlib.sha256("\x1f".join(header).encode("utf-8")).hexdigest(),
        header_refs=frozenset(header_refs),
        files_digest=hashlib.sha256("\x1f".join(sorted(file_parts)).encode("utf-8")).hexdigest(),
        keyed_files=tuple(files),
        cimported=tuple(p for p, local in pxds if local and p != os.path.splitext(pyx)[0] + ".pxd"),
        newer_keyed=tuple(keyed),
    )
    return analysis


def _find_pyx(module: types.ModuleType, binary: str, co_filename: str) -> str:
    """The ``.pyx`` the binary names, found from the binary's folder up."""
    stem = module.__name__.rsplit(".", 1)[-1]
    if os.path.normcase(os.path.splitext(os.path.basename(co_filename))[0]) != os.path.normcase(stem):
        raise _Fallback("the code names another file")
    if os.path.isabs(co_filename):
        candidates = [co_filename]
    else:
        candidates = []
        folder = os.path.dirname(os.path.abspath(binary))
        while True:
            candidates.append(os.path.join(folder, co_filename))
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(binary)), os.path.basename(co_filename)))
    for candidate in candidates:
        if _isfile(candidate):
            return _norm(candidate)
    raise _Fallback("no .pyx")


def _refuse_distutils_settings(text: str) -> str:
    """The ``# distutils:`` settings of *text*, for the header's key; refused
    where one adds code or flags."""
    from Cython.Build.Dependencies import DistutilsInfo

    values = DistutilsInfo(source=text).values
    refused = sorted(set(values) - _HARMLESS_DISTUTILS)
    if refused:
        raise _Fallback(f"# distutils: {', '.join(refused)}")
    return repr(sorted(values.items()))


def _parse(module_name: str, pyx: str) -> Any:
    from Cython.Compiler.Main import Context
    from Cython.Compiler.Options import CompilationOptions
    from Cython.Compiler.Scanning import FileSourceDescriptor

    context = Context.from_options(CompilationOptions())
    source = FileSourceDescriptor(pyx, os.path.basename(pyx))
    scope = context.find_module(module_name, pos=(source, 1, 0), need_pxd=False)
    return context.parse(source, scope, pxd=False, full_module_name=module_name)


def _top_level(node: Any) -> Iterator[Any]:
    if type(node).__name__ == "StatListNode":
        for stat in node.stats:
            yield from _top_level(stat)
    else:
        yield node


def _node_path(node: Any) -> str:
    source = node.pos[0]
    path = getattr(source, "filename", None)
    if not isinstance(path, str):
        raise _Fallback("a statement from no file")
    return _norm(path)


def _tokens(path: str, text: str) -> list[tuple[str, str, int]]:
    """``(kind, text, line)`` for every token of *text*, by Cython's scanner.
    Line endings are made ``\n`` first: the scanner refuses a ``\r``."""
    from io import StringIO

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    from Cython.Compiler.Scanning import PyrexScanner, StringSourceDescriptor
    from Cython.Compiler.TreeFragment import StringParseContext

    name = os.path.splitext(os.path.basename(path))[0] or "m"
    context = StringParseContext(name)
    source = StringSourceDescriptor(name, text)
    scope = context.find_module(name, pos=(name, 1, 0), need_pxd=False)
    scanner = PyrexScanner(
        StringIO(text), source, source_encoding="UTF-8", scope=scope, context=context, initial_pos=(name, 1, 0)
    )
    out = []
    while scanner.sy != "EOF":
        out.append((scanner.sy, scanner.systring or "", scanner.position()[1]))
        scanner.next()
    return out


def _extern_headers(tokens: list[tuple[str, str, int]], path: str) -> list[str]:
    """The library headers *tokens* declare ``cdef extern from``, to key by
    their content.

    A header of the user's own C code is not followed (nor what it
    includes), so only a header found in numpy's or Python's own include
    folder -- fixed by that installation -- is taken; any other extern
    (``math.h``, ``"mylib.h"``, ``extern from *`` with verbatim C) keys
    the whole binary.
    """
    found = []
    for i, (sy, text, _line) in enumerate(tokens):
        if sy == "CHARS" or text != "extern":
            continue
        form = [t for t in tokens[i - 1 : i + 6]]
        if (
            len(form) < 7
            or form[0][0] != "cdef"
            or form[2][0] != "from"
            or form[3][0] != "BEGIN_STRING"
            or form[4][0] != "CHARS"
            or form[5][0] != "END_STRING"
        ):
            raise _Fallback(f"cdef extern in {path}")
        header = _library_header(form[4][1])
        if header is None:
            raise _Fallback(f"cdef extern from {form[4][1]!r} in {path}")
        found.append(header)
    return found


def _library_header(name: str) -> str | None:
    """*name* found in numpy's or Python's include folder, or None."""
    import sysconfig

    if not name or os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
        return None
    folders = [sysconfig.get_paths().get("include"), sysconfig.get_paths().get("platinclude")]
    if name.replace("\\", "/").startswith("numpy/"):
        try:
            import numpy

            folders.insert(0, numpy.get_include())
        except Exception:  # noqa: BLE001 - no numpy, so not its header
            return None
    for folder in folders:
        if folder:
            candidate = os.path.join(folder, *name.replace("\\", "/").split("/"))
            if _isfile(candidate):
                return _norm(candidate)
    return None


def _is_extension(path: str) -> bool:
    """Is *path* a built extension (``.so``, ``.pyd`` in any case)?"""
    return os.path.normcase(path).endswith(tuple(os.path.normcase(s) for s in importlib.machinery.EXTENSION_SUFFIXES))


def _norm(path: str) -> str:
    """*path* as one spelling: absolute, and on Windows in one case."""
    return os.path.normcase(os.path.abspath(path))


def _logical_starts(tokens: list[tuple[str, str, int]]) -> list[tuple[int, str]]:
    """``(line, first token)`` of every top-level logical line."""
    starts = []
    depth = 0
    at_start = True
    for sy, _text, line in tokens:
        if sy == "INDENT":
            depth += 1
            continue
        if sy == "DEDENT":
            depth -= 1
            continue
        if sy == "NEWLINE":
            at_start = True
            continue
        if sy == "commentline":
            continue
        if at_start and depth == 0:
            starts.append((line, sy))
        at_start = False
    return starts


def _blocks(stats: list[Any], tokens: dict[str, list[tuple[str, str, int]]]) -> list[_Block]:
    starts = {path: _logical_starts(toks) for path, toks in tokens.items()}
    blocks = []
    for stat in stats:
        if type(stat).__name__ not in _BLOCK_NODES:
            continue
        path = _node_path(stat)
        line = stat.pos[1]
        lines = [start for start, _ in starts[path]]
        if line not in lines:
            raise _Fallback("a block that starts inside a line")
        index = lines.index(line)
        first = index
        while first > 0 and starts[path][first - 1][1] == "@":
            first -= 1
        end = lines[index + 1] if index + 1 < len(lines) else float("inf")
        blocks.append(_Block(names=_defined_names(stat), path=path, start=lines[first], end=end))
    return blocks


def _defined_names(stat: Any) -> frozenset[str]:
    kind = type(stat).__name__
    if kind == "CClassDefNode":
        names = {stat.class_name, getattr(stat, "as_name", None)}
    elif kind == "CFuncDefNode":
        declarator = stat.declarator
        while not hasattr(declarator, "name") and getattr(declarator, "base", None) is not None:
            declarator = declarator.base
        names = {getattr(declarator, "name", None)}
    else:
        names = {getattr(stat, "name", None)}
    names.discard(None)
    names.discard("")
    if not names:
        raise _Fallback("a block with no name")
    return frozenset(str(n) for n in names)


def _owner_map(tokens: list[tuple[str, str, int]], blocks: list[_Block]) -> list[int | None]:
    """Which block (index into *blocks*) each token is in; None for the
    header. A DEDENT belongs with the token before it, so how deeply one
    block ends does not move the next."""
    order = sorted(range(len(blocks)), key=lambda i: blocks[i].start)
    starts = [blocks[i].start for i in order]
    owners: list[int | None] = []
    previous: int | None = None
    for sy, _text, line in tokens:
        if sy == "DEDENT" and owners:
            owners.append(previous)
            continue
        at = bisect.bisect_right(starts, line) - 1
        owner = order[at] if at >= 0 and line < blocks[order[at]].end else None
        owners.append(owner)
        previous = owner
    return owners


def _build_config(folder: str) -> tuple[str, list[str]]:
    """The project root (the first folder up from *folder* with a
    ``setup.py`` or ``pyproject.toml``) and the config files on the way."""
    configs = []
    while True:
        for name in _UNREAD_BUILD_FILES:
            if _isfile(os.path.join(folder, name)):
                raise _Fallback(f"built by {name}")
        found = [os.path.join(folder, n) for n in _CONFIG_FILES if _isfile(os.path.join(folder, n))]
        configs.extend(found)
        if any(os.path.basename(p) in ("setup.py", "pyproject.toml") for p in found):
            break
        parent = os.path.dirname(folder)
        if parent == folder:
            raise _Fallback("no build configuration")
        folder = parent
    for path in configs:
        # Recorded by its canonical form, as it is keyed.
        form = _read_config(path)
        if path.endswith("setup.py"):
            with open(path, "rb") as fh:
                _refuse_setup_flags(fh.read().decode("utf-8", "replace"))
            continue
        # The canonical form: a word in a description or a classifier is
        # not a flag.
        text = form.decode("utf-8", "replace")
        if any(word in text for word in _CONFIG_FLAG_WORDS):
            raise _Fallback(f"flags or sources in {path}")
    return folder, configs


def _refuse_setup_flags(text: str) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise _Fallback("setup.py does not parse") from e
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in _FLAG_ARGUMENTS:
                raise _Fallback(f"setup.py sets {keyword.arg}")
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if not name.endswith("Extension"):
            continue
        if any(k.arg is None for k in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
            raise _Fallback("setup.py builds an Extension from unpacked arguments")
        sources = [k.value for k in node.keywords if k.arg == "sources"] + node.args[1:2]
        for value in sources:
            if not isinstance(value, (ast.List, ast.Tuple)) or len(value.elts) != 1:
                raise _Fallback("setup.py lists other sources")
            only = value.elts[0]
            if not (isinstance(only, ast.Constant) and isinstance(only.value, str) and only.value.endswith(".pyx")):
                raise _Fallback("setup.py lists other sources")


def _used_pxds(
    module_name: str, pyx: str, tree: Any, files: list[str], root: str, headers: list[str]
) -> list[tuple[str, bool]]:
    """``(path, is local)`` of every ``.pxd`` the module uses: its own, and
    what it cimports, transitively through the local ones. A cimport found
    only in Cython's own ``Includes`` is fixed by the Cython version."""
    includes = _cython_includes()
    package_root = _package_root(module_name, pyx)
    found: dict[str, bool] = {}
    own = os.path.splitext(pyx)[0] + ".pxd"
    pending: list[tuple[str, Any]] = []
    if _isfile(own):
        found[own] = True
        pending.append((own, None))
    for name, level in _cimports(tree):
        pending.extend((p, None) for p in _resolve_cimport(name, level, pyx, package_root, includes))
    while pending:
        path, _ = pending.pop()
        if path in found and path != own:
            continue
        local = _is_local(path, root)
        found[path] = local
        if not local:
            continue
        toks = _tokens(path, _source_text(_read(path)))
        headers.extend(_extern_headers(toks, path))
        if any(sy == "include" for sy, _t, _l in toks):
            raise _Fallback(f"include in {path}")
        for name, level in _token_cimports(toks):
            for target in _resolve_cimport(name, level, path, package_root, includes):
                if target not in found:
                    pending.append((target, None))
    return sorted(found.items())


def _is_local(path: str, root: str) -> bool:
    from .install_paths import is_user_path

    inside = os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(os.path.abspath(root)) + os.sep)
    return inside and is_user_path(path)


def _cimports(tree: Any) -> Iterator[tuple[str, int]]:
    """``(dotted name, relative level)`` of every cimport in the parse tree,
    each ``from X cimport y`` also as ``X.y`` (y may be a submodule)."""
    stack = [tree]
    while stack:
        node = stack.pop()
        kind = type(node).__name__
        if kind == "CImportStatNode":
            yield node.module_name, 0
        elif kind == "FromCImportStatNode":
            level = getattr(node, "relative_level", 0) or 0
            base = node.module_name or ""
            if base:
                yield base, level
            for _pos, name, _as in node.imported_names:
                yield (f"{base}.{name}" if base else name), level
        for attr in getattr(node, "child_attrs", ()) or ():
            child = getattr(node, attr, None)
            if isinstance(child, list):
                stack.extend(c for c in child if hasattr(c, "pos"))
            elif child is not None and hasattr(child, "pos"):
                stack.append(child)


def _token_cimports(tokens: list[tuple[str, str, int]]) -> Iterator[tuple[str, int]]:
    """The cimports of a ``.pxd`` from its tokens: ``cimport a.b`` and
    ``from [.]a cimport b, c``."""
    i = 0
    n = len(tokens)
    while i < n:
        sy = tokens[i][0]
        if sy == "cimport" and (i == 0 or tokens[i - 1][0] in ("NEWLINE", "INDENT", "DEDENT")):
            i += 1
            while i < n and tokens[i][0] != "NEWLINE":
                dotted, i = _dotted(tokens, i)
                if dotted:
                    yield dotted, 0
                if i < n and tokens[i][0] == "IDENT" and tokens[i][1] == "as":
                    i += 2
                if i < n and tokens[i][0] == ",":
                    i += 1
                elif i < n and tokens[i][0] != "NEWLINE":
                    raise _Fallback("a cimport not understood")
        elif sy == "from":
            i += 1
            level = 0
            while i < n and tokens[i][0] == ".":
                level += 1
                i += 1
            base, i = _dotted(tokens, i)
            if i >= n or tokens[i][0] != "cimport":
                continue
            if base:
                yield base, level
            i += 1
            while i < n and tokens[i][0] != "NEWLINE":
                if tokens[i][0] == "IDENT" and tokens[i][1] != "as" and tokens[i - 1][1] != "as":
                    yield (f"{base}.{tokens[i][1]}" if base else tokens[i][1]), level
                i += 1
        i += 1


def _dotted(tokens: list[tuple[str, str, int]], i: int) -> tuple[str, int]:
    parts = []
    while i < len(tokens) and tokens[i][0] == "IDENT":
        parts.append(tokens[i][1])
        i += 1
        if i < len(tokens) and tokens[i][0] == ".":
            i += 1
        else:
            break
    return ".".join(parts), i


def _resolve_cimport(name: str, level: int, importer: str, package_root: str, includes: str) -> list[str]:
    """The ``.pxd`` a cimport of *name* reads, as a list of at most one.

    A ``from X cimport y`` yields ``X.y`` too, where ``y`` is usually a name
    in ``X.pxd``, not a module: a dotted name of that shape that is found
    nowhere is no error. A plain module found nowhere is.
    """
    parts = [p for p in name.split(".") if p]
    if level:
        base = os.path.dirname(importer)
        for _ in range(level - 1):
            base = os.path.dirname(base)
        roots = [base]
    else:
        roots = [package_root, os.path.dirname(importer), *(p for p in sys.path if p and _isdir(p))]
        roots.append(includes)
    for root in roots:
        for candidate in (
            os.path.join(root, *parts) + ".pxd",
            os.path.join(root, *parts, "__init__.pxd"),
        ):
            if _isfile(candidate):
                if os.path.normcase(os.path.abspath(candidate)).startswith(os.path.normcase(includes) + os.sep):
                    return []
                return [_norm(candidate)]
    if not parts or (not level and name == "cython"):
        return []
    if len(parts) > 1 or (level and len(parts) == 1):
        # ``from X cimport y`` where y is a name declared in X.pxd, not a
        # module: fine when X itself is found (it raises when it is not).
        parent = ".".join(parts[:-1])
        if parent or level:
            _resolve_cimport(parent, level, importer, package_root, includes)
            return []
    raise _Fallback(f"cimport {name} not found")


def _refuse_newer_than(paths: Iterable[str], binary: str) -> None:
    built = os.stat(binary).st_mtime_ns
    for path in paths:
        if os.stat(path).st_mtime_ns > built:
            raise _Fallback(f"{path} is newer than the build")


def _label(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root).replace(os.sep, "/")
    except ValueError:
        return os.path.basename(path)


def _module_functions(module: types.ModuleType) -> Iterator[Any]:
    """The Cython functions the module defines, its classes' included."""
    for value in list(vars(module).values()):
        if is_cyfunction(value):
            yield value
        elif isinstance(value, type) and getattr(value, "__module__", None) == module.__name__:
            for member in list(vars(value).values()):
                member = getattr(member, "__func__", member)
                if is_cyfunction(member):
                    yield member


def _check_built_from(analysis: _Module, module: types.ModuleType) -> None:
    """Every function of the binary from this ``.pyx`` must start in the
    block its name says, at the line the binary records: a ``.pyx`` edited
    and rebuilt elsewhere, or never rebuilt, does not match."""
    pyx_name = os.path.basename(os.path.normcase(analysis.pyx))
    for fn in _module_functions(module):
        code = getattr(fn, "__code__", None)
        if not isinstance(code, types.CodeType):
            continue
        if os.path.basename(os.path.normcase(code.co_filename)) != pyx_name and _code_path(analysis, code) is None:
            continue
        qualname = str(getattr(fn, "__qualname__", "") or "")
        if _own_blocks(analysis, code, qualname) is None:
            raise _Fallback(f"{qualname} is not where the binary says")
