"""The one digest that stands for a callable's code (`callable_identity`)
or a module's (`module_identity`), wherever cash keys on it.

Read from the source file and reduced to its canonical form
(`cash.source_norm`), so a comment or a reformat does not move it; failing
that from the compiled body (`bytecode_identity`), and for a callable with
no code from its name and, for a user's compiled extension, the built
file. A ``functools.wraps`` wrapper's digest holds both its own code and
what it wraps. The extension and module digests are memoised per file.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import os
import sys
import types

from .exceptions import SOURCE_RETRIEVAL_ERRORS
from .source_norm import bytecode_identity, module_text_identity, source_identity_digest
from .source_reading import own_source, read_code_file, stat_has_settled
from .tracking.tracker_context import untracked

__all__ = [
    "callable_identity",
    "compiled_identity",
    "extension_file_digest",
    "module_identity",
    "own_source_digest",
    "source_digest",
    "unwrap_partials",
]


def callable_identity(fn: object) -> str:
    """The one digest that stands for a callable's code, wherever cash keys on it.

    Its own source text reduced by `source_identity_digest` (a comment, a
    reformat or cash's own decorator arguments do not move it); failing that
    its compiled body (`bytecode_identity`: a REPL, ``exec``, a moved file);
    failing that its `opaque_identity`. For a ``functools.wraps`` wrapper,
    its OWN code (`own_source`) together with the identity of what it wraps,
    so an edit to either half moves it and two functions wrapped by one
    decorator never share it. Never raises.
    """
    return _callable_identity(fn, frozenset())


def _callable_identity(fn: object, walked: frozenset[int]) -> str:
    digest = _source_digest(fn, walked)
    return digest if digest is not None else _compiled_identity(fn, walked)


def source_digest(fn: object) -> str | None:
    """*fn*'s identity read from its source file -- `source_identity_digest`
    of its own source (`own_source`), with what a ``functools.wraps`` wrapper
    wraps folded in -- or ``None`` when there is no source to read."""
    return _source_digest(fn, frozenset())


def _source_digest(fn: object, walked: frozenset[int]) -> str | None:
    own = own_source_digest(fn)
    return None if own is None else _with_wrapped(own, fn, walked)


def own_source_digest(fn: object) -> str | None:
    """`source_identity_digest` of `own_source` alone -- the text in *fn*'s
    own file, without what a wrapper wraps -- or ``None`` without source.
    What a check that one FILE still holds the code that runs compares."""
    try:
        return source_identity_digest(own_source(fn))
    except SOURCE_RETRIEVAL_ERRORS:
        return None


def _wrapped_of(fn: object) -> object | None:
    """What a ``functools.wraps`` wrapper FUNCTION wraps, or None."""
    if isinstance(fn, types.FunctionType):
        return getattr(fn, "__wrapped__", None)
    return None


def _with_wrapped(own: str, fn: object, walked: frozenset[int]) -> str:
    """*own*, the digest of *fn*'s own code, joined with the identity of the
    function it wraps, if it is a ``functools.wraps`` wrapper.

    A wrapper's own code is shared by every function its decorator wraps, so
    on its own it cannot tell ``@timed def a`` from ``@timed def b`` -- nor see
    an edit to either body. The wrapped function is what the wrapper runs.

    Every layer is followed, however many decorators are stacked; *walked*
    (the layers already in this identity) ends a ``__wrapped__`` cycle.
    """
    wrapped = _wrapped_of(fn)
    if wrapped is None:
        return own
    walked = walked | {id(fn)}
    if id(wrapped) in walked:
        return hashlib.sha256(f"{own}:wraps:cycle".encode("utf-8")).hexdigest()
    inner = _callable_identity(wrapped, walked)
    return hashlib.sha256(f"{own}:wraps:{inner}".encode("utf-8")).hexdigest()


def compiled_identity(fn: object) -> str:
    """*fn*'s identity when its source cannot be read: its `bytecode_identity`
    (a wrapper's with what it wraps folded in), or for a callable with no
    code at all a digest of its `opaque_identity`."""
    return _compiled_identity(fn, frozenset())


def _compiled_identity(fn: object, walked: frozenset[int]) -> str:
    own = bytecode_identity(fn)
    built = extension_file_digest(fn)
    if own is None:
        opaque = f"__cash_opaque__:{opaque_identity(fn)}"
        if built is not None:
            opaque = f"{opaque}:built:{built}"
        return hashlib.sha256(opaque.encode("utf-8")).hexdigest()
    if built is not None:
        own = hashlib.sha256(f"{own}:built:{built}".encode()).hexdigest()
    return _with_wrapped(own, fn, walked)


def opaque_identity(fn: object) -> str:
    """A stable ``module.qualname`` for a callable with no source and no code:
    a builtin, a C-extension function, a ufunc, a ``functools.partial``.

    A partial reprs as ``functools.partial(<function slow at 0x...>, 1)``: an
    ADDRESS, so its identity differed in every process and a cached partial
    never hit across processes. What it wraps is stable; what it binds reaches
    a key through the arguments and the function's own namespace name.
    """
    fn = unwrap_partials(fn)
    module = getattr(fn, "__module__", None) or "?"
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    return f"{module}.{qualname}"


def unwrap_partials(fn: object) -> object:
    """What a chain of ``functools.partial`` objects finally calls, however long.

    CPython flattens a partial of a plain partial, but not of a subclass, so a
    chain can be any length; one that stopped after eight left the function
    it wraps out of every identity built from the result. A partial whose
    ``func`` leads back to itself (only ``__setstate__`` can build one) ends
    at the first repeat.
    """
    seen: set[int] = set()
    while isinstance(fn, functools.partial) and id(fn) not in seen:
        seen.add(id(fn))
        fn = fn.func
    return fn


#: extension file path -> (the module loaded from it, its content digest).
_EXTENSION_DIGESTS: dict[str, tuple[types.ModuleType, str]] = {}


def extension_file_digest(fn: object) -> str | None:
    """Digest of the compiled extension *fn* was loaded from, when that file is the user's.

    A C or Cython function has no source, and a C one no bytecode either,
    so it was keyed by its name: ``fastops.scale`` built in place (``build_ext
    --inplace``, an editable install) and rebuilt to compute something else
    was served the old result. The built file stands for its code. Read once
    per loaded module: a process cannot load a rebuilt file in place of the
    one it runs, so the first digest is the one that matches the code.
    None for a library's extension, which is fixed for an environment.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    module = sys.modules.get(getattr(fn, "__module__", None) or "")
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or not path.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        return None
    cached = _EXTENSION_DIGESTS.get(path)
    if cached is not None and cached[0] is module:
        return cached[1]
    from .install_paths import is_user_path

    if not is_user_path(path):
        return None
    try:
        with untracked(), open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None
    _EXTENSION_DIGESTS[path] = (module, digest)
    return digest


#: ``{path: (mtime_ns, size, identity_digest)}`` for `module_identity`. One
#: entry per file, replaced when it moves. The identity below parses the file, which is far too much to
#: repeat per statement -- and even the plain read it replaces was one file
#: read per statement per module.
_MODULE_IDENTITY_CACHE: dict[str, tuple[int, int, str]] = {}


def module_identity(module: object) -> str | None:
    """The identity digest of a module's source file -- see
    `module_text_identity` for what it covers -- or ``None`` when the file
    cannot be read. *module* is a module object or the path of its file.
    Memoised on the file's stat."""
    path = module if isinstance(module, str) else getattr(module, "__file__", None)
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    cached = _MODULE_IDENTITY_CACHE.get(path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    settled = stat_has_settled(st)
    try:
        raw = read_code_file(path)
    except OSError:
        return None
    # Keyed on the same signal `FunctionTracker.check_tracked_modules` uses to
    # notice a module changed at all, so a change this memo would miss is one
    # cash would not have reloaded for either -- once the file has settled,
    # since a same-size save inside one mtime tick keeps that stat too.
    digest = hashlib.sha256(module_text_identity(raw)).hexdigest()
    if settled:
        _MODULE_IDENTITY_CACHE[path] = (st.st_mtime_ns, st.st_size, digest)
    return digest
