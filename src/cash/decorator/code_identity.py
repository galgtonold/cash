"""The identity of code: a function's own source, the classes and
callables it reaches, and which code counts as the user's."""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import inspect
import logging
import os
import pickle
import re
import sys
import textwrap
import threading
import types
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._annotation_refs import annotation_referents
from .._memo import CODE_OBJECTS, LruMemo
from .._paths import MAIN_MODULE_NAMES, resolve_main_module
from ..analysis.helper_code import UnwalkableLayers, callable_layers
from ..canonical_form import stable_key_repr
from ..code_digest import callable_identity, compiled_identity, own_source_digest, source_digest, unwrap_partials
from ..content_hashers import BUILTIN_CONTENT
from ..diagnostics import warn_diagnostic
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashCacheIneffectiveWarning
from ..install_paths import in_own_package, is_cash_path, is_user_code_module, is_user_module, top_package
from ..loaded_code import loaded_class_identity, loaded_code_matches_disk
from ..source_norm import bytecode_identity, code_consts_without_docstring
from ..source_reading import getsource
from .arg_hashing import is_opaque
from .call_state import KeyBuildFailed

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher

logger = logging.getLogger(__name__)


# id(code object) -> (the object itself, its source digest). Keyed by IDENTITY,
# and the object is retained so the id cannot be recycled under us -- the same
# guard `function_tracker._source_cache` uses.
#
# NOT keyed on the code object directly: CodeType implements __eq__/__hash__
# BY VALUE, and co_filename is not part of that equality, so two helpers with
# the same body in different modules would share one slot.
#
# A redefinition (reloaded module, re-run cell) compiles a NEW code object, so
# identity keying still cannot serve a digest for code that is no longer
# running. Editing a .py file WITHOUT reloading leaves the old code object
# live, and the old digest is then the correct answer.
#
# Load-bearing, not a micro-optimisation. `hash_callable_source` is the live
# per-call identity of every transitive helper, and it calls
# `inspect.getsource`, which re-reads and RE-TOKENISES the source block on
# every call: without the memo, most of a hit's key computation.
#
# Module-level rather than per-instance: the digest depends only on the code
# object, so two Cash instances cannot legitimately disagree about it.
SOURCE_HASH_MEMO: LruMemo[int, tuple[Any, str]] = LruMemo(CODE_OBJECTS)
#: ``id(code) -> (code, path, size, mtime_ns, text digest)``: the stat of the
#: file whose text a function's key was read from, taken just before reading
#: it, and that text's digest. The store compares both with the file now
#: (`FileDeps.code_moved_since_keyed`).
CODE_KEYED_STATS: LruMemo[int, tuple[Any, str, int, int, str]] = LruMemo(CODE_OBJECTS)


#: Source files already reported as edited-since-load, one notice per file.
_SOURCE_CHANGED_WARNED: set[str] = set()


def stat_code_file(fn: Any) -> tuple[Any, str, int, int] | None:
    """``(code, path, size, mtime_ns)`` for *fn*'s source file, or None."""
    code = getattr(fn, "__code__", None)
    path = getattr(code, "co_filename", "") or ""
    if not path or path.startswith("<"):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (code, path, st.st_size, st.st_mtime_ns)


def warn_source_changed_since_load(fn: Callable) -> None:
    """Say, once per file, that a helper is keyed by its loaded code."""
    code = getattr(fn, "__code__", None)
    path = getattr(code, "co_filename", "") or ""
    if path in _SOURCE_CHANGED_WARNED:
        return
    _SOURCE_CHANGED_WARNED.add(path)
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "a function")
    try:
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "KEY-SOURCE-CHANGED",
            f"{path} was edited after this process loaded it, so the code running "
            f"{name}() is the old version while the file holds a new one. cash "
            f"keys it by the code actually running, so results stay correct for "
            f"this process -- but they are not the new code's results, and they "
            f"will not be reused once the process restarts.",
            "restart the process to run the new code. If a deploy puts new files "
            "on disk before the restart, this is the window it opens.",
        )
    except Exception:  # a notice must never break a call
        logger.debug("Could not emit the source-changed notice", exc_info=True)


#: Pydantic v2 compiles these onto every model class. They are derived from the
#: field declarations and their digest differs in every process, so folding them
#: made a pydantic spec un-cacheable across runs. `CodeIdentity._pydantic_field_parts`
#: folds the declarations they were standing in for.
PYDANTIC_COMPILED = frozenset(
    {
        "__pydantic_core_schema__",
        "__pydantic_serializer__",
        "__pydantic_validator__",
    }
)


def func_key(func: Callable) -> str:
    """Return a module-qualified key for a function.

    Uses ``func.__module__ + '.' + func.__qualname__`` to avoid collisions
    when different modules define functions with the same ``__qualname__``
    (e.g. a notebook's ``dep()`` vs a library module's ``dep()``).

    ``__main__`` is resolved to the name the module would have when
    imported — see `resolve_main_module`.

    A ``functools.partial`` is named after the function it wraps plus a
    digest of what it binds, a callable instance after its class. Anything
    else without ``__qualname__`` or ``__name__`` falls back to ``repr`` so
    keying it never crashes.
    """
    if isinstance(func, functools.partial):
        # `repr(partial)` holds the wrapped function's ADDRESS, so every
        # process took a fresh namespace and none of them ever hit. Name it after what it
        # wraps, plus what it binds -- two partials of one function stay
        # two namespaces, and each is the same in every process.
        # The bound values' content, not their ``repr``: that holds an
        # ordinary object's address, and a new process never found the
        # namespace again. Only a name: the values themselves are keyed per
        # call (`ClosureFold.fold_bound_partial`).
        inner = func_key(func.func)
        try:
            bound = hashlib.sha256(
                pickle.dumps(stable_key_repr((func.args, sorted(func.keywords.items())), BUILTIN_CONTENT), protocol=4),
            ).hexdigest()[:12]
        except Exception:  # noqa: BLE001 - an unpicklable argument keys on its type
            shape = [type(v).__qualname__ for v in (*func.args, *func.keywords.values())]
            bound = hashlib.sha256(repr((shape, sorted(func.keywords))).encode("utf-8")).hexdigest()[:12]
        return f"{inner}[partial:{bound}]"
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if not qualname and callable(func) and not isinstance(func, type):
        # A callable INSTANCE (`cash.cache(Scaler(2))`) reprs with its address,
        # so every process took a fresh namespace and none ever hit. Named
        # after its class; what it holds is keyed per call
        # (`ClosureFold.fold_bound_self`).
        cls = type(func)
        module = getattr(cls, "__module__", None) or "__unknown__"
        if module in MAIN_MODULE_NAMES:
            call = getattr(cls, "__call__", None)
            module = resolve_main_module(call) if hasattr(call, "__globals__") else module
        return f"{module}.{cls.__qualname__}[instance]"
    module = getattr(func, "__module__", None) or "__unknown__"
    if module in MAIN_MODULE_NAMES:
        module = resolve_main_module(func)
    return f"{module}.{qualname or repr(func)}"


def hash_callable_source(fn: Callable) -> str:
    """Return a stable hex digest representing *fn*'s body.

    Used by `register_hasher` to embed the hasher's source
    identity in the cache key, so that changing a hasher's body
    invalidates dependent cache entries even when the hasher's
    output coincidentally matches the old one. Also the
    ``hash_callable`` injected into ``SysModulesHelperResolver``,
    which makes it the live per-call identity of every transitive
    HELPER -- so what this returns decides whether editing a helper
    recomputes its callers.

    This is `cash.code_digest.callable_identity`, plus a memo per code
    object and a check that the file still holds the code that runs.

    Resolution order:

    1. ``own_source(fn)`` - primary (for a ``functools.wraps`` wrapper its
       own code, with the identity of what it wraps folded in), reduced via
       ``source_identity_digest`` so that a comment, a reformat, or a
       change to cash's own ``@....cache`` decorator arguments in a
       helper does not invalidate the functions that call it. Works
       for module-level functions and lambdas defined in a
       discoverable source file.
    2. ``bytecode_identity(fn)`` - fallback. Works for functions defined
       in a REPL or via ``exec()``, and for callable instances (it reads
       ``__call__``), so two instances of the same callable class share
       one identity. Folds consts/names/varnames, NOT ``co_code`` alone:
       a const load's operand is an index, so bare ``co_code`` cannot
       see ``return "alpha"`` become ``return "omega"``. Bytecode is
       stable within a Python version; an upgrade conservatively
       invalidates the cache.
    3. ``opaque_identity(fn)`` (``module.qualname``) - last resort, for a
       builtin, a ufunc or a partial. Doesn't differentiate instances of
       the same class; stable across processes, but coarse.
    """
    memo_owner: Any = getattr(fn, "__code__", None)
    if (memo_owner is None and isinstance(fn, type)) or (
        isinstance(fn, types.FunctionType) and hasattr(fn, "__wrapped__")
    ):
        # A class has no code object; a `functools.wraps` wrapper shares
        # its code with every function its decorator wraps, and its
        # identity includes the one it wraps -- so it is memoized as itself.
        memo_owner = fn
    memo_key = id(memo_owner) if memo_owner is not None else None
    if memo_key is not None:
        entry = SOURCE_HASH_MEMO.get(memo_key)
        # ``is``, not ``==``: confirms this is the SAME object and not a
        # recycled id, and sidesteps CodeType's by-value equality.
        if entry is not None and entry[0] is memo_owner:
            return entry[1]

    # The source on disk may no longer be the code that is running: a file
    # edited after this process imported it (new files land, the restart
    # comes later) gives the NEW text for the OLD code object; an entry keyed
    # by the new text but computed by the old code would be served to the
    # restarted process. Key such a helper by what actually runs.
    # One os.stat in the normal case; see `loaded_code_matches_disk`.
    if not loaded_code_matches_disk(fn):
        digest = loaded_class_identity(fn) if isinstance(fn, type) else compiled_identity(fn)
        if digest is not None:
            warn_source_changed_since_load(fn)
            if memo_key is not None:
                SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
            return digest

    # `callable_identity`, in its two halves: only a digest read from the
    # file is recorded against the file's stat.
    keyed_stat = stat_code_file(fn)
    digest = source_digest(fn)
    if digest is None:
        digest = compiled_identity(fn)
        # A function with no readable source (a dataclass's generated
        # `__init__`, an exec'd helper) is its bytecode, which its code
        # object fixes: memoized like a source digest, or every hit re-ran a
        # failing `inspect.getsource` per generated method.
        if memo_key is not None and isinstance(memo_owner, types.CodeType):
            SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
        return digest
    if memo_key is not None:
        SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
    if keyed_stat is not None:
        own = digest if memo_owner is not fn else own_source_digest(fn)
        if own is not None:
            CODE_KEYED_STATS[id(keyed_stat[0])] = (*keyed_stat, own)
    return digest


def code_fingerprint(code: types.CodeType) -> str:
    """A digest of what a code object DOES, independent of where it sits.

    Source text is not enough on its own for a lambda: two different
    lambdas written on the SAME physical line share their
    ``inspect.getsource`` result, so ``a(lambda: "AAA"), a(lambda: "BBB")``
    fingerprint identically and collide. Their code objects differ, which
    is the signal this reads.

    Deliberately built from ``co_code``/``co_names``/``co_varnames`` and the
    constants, never from ``repr`` of a nested code object -- that carries a
    memory address, which would make the key unstable across processes and
    turn every restart into a miss. Nested code (a lambda inside a lambda)
    recurses instead, however deep: code objects form a finite tree, and a
    level left out would let two lambdas differing only there collide.
    """
    parts: list[str] = [
        code.co_code.hex(),
        repr(code.co_names),
        repr(code.co_varnames),
        repr(code.co_freevars),
    ]
    for const in code_consts_without_docstring(code):
        if isinstance(const, types.CodeType):
            parts.append(code_fingerprint(const))
        elif isinstance(const, frozenset):
            # `x in {"a", "b"}` compiles to a frozenset, whose repr follows
            # the per-process string hash: its sorted members instead.
            parts.append(f"frozenset({sorted(map(repr, const))})")
        else:
            parts.append(repr(const))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


#: How many user-code objects one reference walk (`CodeIdentity._code_ref_closure`)
#: may reach. Not a depth or a count that real code meets: the walk follows
#: every reference, however deep, and its seen set ends cycles. What can pass
#: it is code that makes a NEW object on every read (a module ``__getattr__``
#: building a function per lookup), where the walk would never end. Past it
#: the key would leave code out, so the call runs uncached instead
#: (KEY-HELPERS-UNWALKABLE), as the helper walk does
#: (``HelperWalk.WALK_LIMIT``).
MAX_CODE_REF_TARGETS = 5_000


def walk_nested_code(code: types.CodeType, glb: dict):
    """Yield *code* and the code objects nested in its constants, however deep.

    A comprehension, a lambda, or a nested ``def`` compiles to its own
    code object stored in ``co_consts``; the names IT references do not
    appear in the parent's ``co_names``. ``field(default_factory=lambda:
    B(0))`` is exactly that shape -- ``B`` is reachable only through the
    lambda -- so a walk that stopped at the top level would miss the case
    this exists for.
    """
    stack = [code]
    while stack:
        current = stack.pop()
        yield current, glb
        stack.extend(const for const in reversed(current.co_consts) if isinstance(const, types.CodeType))


def iter_contained(obj: Any):
    """Yield *obj*, or its members if it is a plain container, skipping
    primitives outright (they can hold no user class and are common)."""
    if isinstance(obj, (str, bytes, bytearray, int, float, bool, complex, type(None))):
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        yield from obj
    elif isinstance(obj, dict):
        yield from obj.values()
    else:
        yield obj


def is_cash_wrapper(value: Any) -> bool:
    """Is *value* the wrapper ``@cash.cache`` returns, or a method bound
    from one? A mock answers every attribute, never with True."""
    try:
        return getattr(value, "_cash_cached", False) is True
    except Exception:  # noqa: BLE001 - an object's __getattr__ may raise anything
        return False


def cached_function_in(value: Any) -> Any:
    """The ``@cash.cache`` wrapper *value* is, or a partial wraps; else None."""
    value = unwrap_partials(value)
    return value if is_cash_wrapper(value) else None


def cash_wrapped(value: Any) -> Any:
    """The user function under a ``@cash.cache`` wrapper, else *value*.

    A wrapper's code is cash's, never part of a key: a code channel that
    meets one keys the function it wraps, and a data channel keys its state
    (`GlobalsFold.data_callable_identity`).
    """
    return getattr(value, "__wrapped__", value) if is_cash_wrapper(value) else value


def own_package(func: Any) -> str | None:
    """The top-level package of the module that defines *func*."""
    return top_package(getattr(func, "__module__", None))


def is_user_class(cls: Any, own_pkg: str | None = None) -> bool:
    """Is *cls* a class of the user's: inside *own_pkg*, or in a module
    `is_user_code_module` accepts (a notebook cell's class included)?

    Used to fold what a class holds (``ClassName.CONSTANT``, an instance's
    class): editing a class-level constant invalidates, while a library
    class's attributes do not churn the key.
    """
    name = getattr(cls, "__module__", None)
    if in_own_package(name, own_pkg):
        return True
    mod = sys.modules.get(name or "")
    return mod is not None and is_user_code_module(mod)


def is_immutable_capture(v: Any) -> bool:
    """True for values that are immutable and so define a closure's
    behaviour without drifting between calls. Mutable captures (dict/list/
    set/objects) are excluded: they are typically side-effect accumulators
    (e.g. a hit counter) whose value changes every call - folding those into
    the key would make every call miss. Tuples are looked through however deep
    they nest: one cannot hold itself."""
    if isinstance(v, (bool, int, float, complex, str, bytes, type(None))):
        return True
    if isinstance(v, (tuple, frozenset)):
        return all(is_immutable_capture(x) for x in v)
    return False


def _defaults_pin(functions: list[Any]) -> tuple | None:
    """The defaults of *functions*, when every one is immutable; else None."""
    pin = []
    for fn in functions:
        pos = getattr(fn, "__defaults__", None) or ()
        kwd = getattr(fn, "__kwdefaults__", None) or {}
        if not (all(is_immutable_capture(v) for v in pos) and all(is_immutable_capture(v) for v in kwd.values())):
            return None
        pin.append((pos, dict(kwd)))
    return tuple(pin)


#: Synchronization objects (``threading.Lock()``): no result is computed
#: from one, so a lock held as a class attribute, captured by a closure or
#: given as a default is left out of the key.
SYNC_TYPES: tuple[type, ...] = (
    type(threading.Lock()),
    type(threading.RLock()),
    threading.Condition,
    threading.Event,
    threading.Semaphore,
)

#: A memory address in a repr (``<object object at 0x7f...>``).
_ADDRESS_REPR = re.compile(r"\bat 0x[0-9a-fA-F]+")

_HEAPTYPE = 1 << 9
_IMMUTABLETYPE = 1 << 8

#: Callables implemented in C: builtin functions and bound methods, and the
#: slot and method descriptors of C types.
_C_CALLABLES = (
    types.BuiltinFunctionType,
    types.MethodWrapperType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
)


def _is_c_type(cls: type) -> bool:
    """Was *cls* made by C code rather than a ``class`` statement? A static
    type, or a heap type an extension created immutable; Python cannot make
    either."""
    flags = getattr(cls, "__flags__", 0)
    return not flags & _HEAPTYPE or bool(flags & _IMMUTABLETYPE)


def _c_code_verdict(obj: Any) -> bool | None:
    """`is_user_code_object` for C code whose ``__module__`` cannot be
    followed; ``None`` when *obj* is not C code.

    A C type or callable is never the exec'd or notebook code that the
    fallback is for. ``sys.stdout.write`` has no ``__module__`` and
    ``_thread.lock`` is not reachable as ``_thread.lock``, and both counted as
    user code whose code could not be hashed: a logger, a queue or a stream
    reached through an argument warned about ``lock.acquire``. A bound method
    is judged by the class that defines it, so one of an extension built in
    the project still counts and ``write`` of your ``io.StringIO`` subclass
    does not.
    """
    if isinstance(obj, type):
        return False if _is_c_type(obj) else None
    if not isinstance(obj, _C_CALLABLES):
        return None
    owner = getattr(obj, "__self__", None)
    if owner is None:
        owner = getattr(obj, "__objclass__", None)
    if isinstance(owner, types.ModuleType):
        return is_user_code_module(owner)
    if owner is None:
        return False
    cls = owner if isinstance(owner, type) else type(owner)
    name = getattr(obj, "__name__", None)
    try:
        cls = next((klass for klass in cls.__mro__ if name in vars(klass)), cls)
    except Exception:  # noqa: BLE001 - an odd class must not break the verdict
        pass
    return is_user_code_object(cls)


def is_user_code_object(obj: Any) -> bool:
    """True when *obj* -- a class OR a function -- is defined in code the user
    plausibly edits. Both carry ``__module__``, so one predicate serves both.

    ``__module__`` alone is not trustworthy. A class or function built by
    ``exec(body, ns)`` where *ns* lacks a ``__name__`` key (a bare ``{}``,
    unlike a real notebook's globals, which start with ``__name__ ==
    '__main__'``) gets a fallback ``__module__`` from CPython's implicit
    ``__module__ = __name__`` lookup at definition time: ``None`` for a
    function, and -- because that lookup falls all the way through to the
    REAL ``builtins`` module's own ``__name__`` attribute -- literally
    ``'builtins'`` for a class. Neither reflects where the code actually
    lives. Confirm *obj* is actually reachable through the module it
    claims before trusting that module's verdict; otherwise this is the
    exec()/notebook case the predicate exists to catch, so it counts as
    user code (mirroring ``is_user_code_module``'s fileless-module
    handling) -- unless it is C code (`_c_code_verdict`).
    """
    code = getattr(obj, "__code__", None)
    if isinstance(code, types.CodeType) and is_cash_path(code.co_filename):
        # cash's own code -- the wrapper `@cash.cache` returns, whose
        # ``__module__`` and ``__qualname__`` are the user function's.
        return False
    mod_name = getattr(obj, "__module__", None)
    mod = sys.modules.get(mod_name) if mod_name else None
    if mod is None or not qualname_resolves_in(mod, obj):
        verdict = _c_code_verdict(obj)
        return True if verdict is None else verdict
    return is_user_code_module(mod)


def qualname_resolves_in(mod: Any, obj: Any) -> bool:
    """True if *obj* is actually reachable by walking its ``__qualname__``
    from *mod*, not merely claiming *mod* via ``__module__``.

    ``getattr(x, name, default)`` only swallows ``AttributeError`` -- a
    module implementing PEP 562 ``__getattr__`` (a real pattern for
    deprecation shims: raise a custom error for an old name instead of
    just returning it) can make this walk raise something else entirely,
    and ``is_user_code_object`` must never raise.
    """
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if not qualname:
        return False
    cur = mod
    try:
        for part in qualname.split("."):
            if part == "<locals>":
                return False  # nested in a function body - not module-reachable
            cur = getattr(cur, part, None)
            if cur is None:
                return False
        return cur is obj
    except Exception:  # noqa: BLE001 - a module __getattr__ may raise anything
        return False  # could not confirm reachability - do not trust it


def wraps_code(value: Any) -> bool:
    """Is *value* a descriptor around a function (classmethod, staticmethod,
    property, cached_property, partialmethod...)?"""
    if isinstance(value, (classmethod, staticmethod, property, functools.cached_property, functools.partialmethod)):
        return True
    return hasattr(type(value), "__get__") and any(
        callable(getattr(value, name, None)) for name in ("__func__", "fget", "func")
    )


class CodeIdentity:
    """What identifies code for the key: a function's own pinned source, and
    the code surface of classes, instances and functions reached through
    arguments and globals."""

    #: Pins held at once. Not a memo: a pin is the text on disk when the
    #: decorator ran and cannot be taken again later, so a full table keeps
    #: the pins it has (see `pin_own_source`).
    OWN_PINS_MAX = 4096

    def __init__(self, args: ArgHasher) -> None:
        self._args = args
        # id(func) -> (reference to func, decoration-pinned own-source
        # identity). The reference is checked on every read: a redefined
        # function's id can go to a later definition once the old one dies.
        self._own_pins: dict[int, tuple[Callable[[], Any], str]] = {}
        # Pins taken at decoration whose file has not yet been compared with
        # the loaded code; the first call does it once (see `pin_own_source`).
        self._own_pins_unverified: set[int] = set()
        # (first_param, self_attrs, uses_super) per code object; see
        # `_analyze_method_self_deps`.
        self._method_self_dep_cache: LruMemo[Any, tuple[str | None, tuple[str, ...], bool]] = LruMemo(CODE_OBJECTS)
        # user class -> source hash. A class's source cannot change within a
        # running interpreter, so it is hashed once and reused; see
        # `user_class_source_hash` / `instance_class_source_parts`.
        self._user_class_src_cache: LruMemo[type, str] = LruMemo(CODE_OBJECTS)
        # user class or function -> code-surface digest (bytecode-based, class-
        # aware); see `code_surface_hash`. Keyed on the object itself, not
        # id(), so a redefinition (a new object) is a distinct memo entry.
        self._code_surface_cache: LruMemo[Any, tuple[tuple | None, list, str]] = LruMemo(CODE_OBJECTS)
        # object -> tuple of (code object, globals dict) it carries. Static for
        # as long as that object exists (a redefinition makes a new one), so it
        # is safe to memo; the NAMES those code objects reference are resolved
        # fresh per call, because what a name is bound to can change.
        self._code_refs_cache: LruMemo[Any, tuple] = LruMemo(CODE_OBJECTS)

    def _analyze_method_self_deps(self, func: Callable) -> tuple[str | None, tuple[str, ...], bool]:
        """Attributes a method reads on its first parameter, and whether it calls super().

        A ``@cash.cache`` method reaching class-level code -- ``self.helper()``,
        ``self.RATE``, ``super().m()`` -- had none of that in its key, because at
        decoration time the class does not exist yet and the analyzer sees only
        an attribute access on a parameter. Recorded here (source-derived,
        cached per code object) and resolved against the real class at call time
        by :meth:`CodeIdentity.fold_method_class_deps`.

        Returns ``(first_param_name, attr_names_accessed_on_it, uses_super)``.
        ``first_param_name`` is ``None`` when there is no source / no parameters.
        """
        code = getattr(func, "__code__", None)
        if code is not None:
            cached = self._method_self_dep_cache.get(code)
            if cached is not None:
                return cached
        result: tuple[str | None, tuple[str, ...], bool] = (None, (), False)
        try:
            src = textwrap.dedent(getsource(func))
            tree = ast.parse(src)
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError,):
            if code is not None:
                self._method_self_dep_cache[code] = result
            return result
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_def = node
                break
        if func_def is None or not func_def.args.args:
            if code is not None:
                self._method_self_dep_cache[code] = result
            return result
        self_name = func_def.args.args[0].arg
        attrs: set[str] = set()
        uses_super = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Attribute):
                v = node.value
                # ``self.<attr>`` in any position (read or call receiver).
                if isinstance(v, ast.Name) and v.id == self_name:
                    attrs.add(node.attr)
                # ``type(self).<attr>`` -- resolves to the same class member as
                # self.<attr> for class-level attributes; treat it the same.
                elif (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Name)
                    and v.func.id == "type"
                    and len(v.args) == 1
                    and isinstance(v.args[0], ast.Name)
                    and v.args[0].id == self_name
                ):
                    attrs.add(node.attr)
            # ``super()`` / ``super(...)`` anywhere.
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id == "super":
                    uses_super = True
                elif (
                    isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Call)
                    and isinstance(f.value.func, ast.Name)
                    and f.value.func.id == "super"
                ):
                    uses_super = True
        result = (self_name, tuple(sorted(attrs)), uses_super)
        if code is not None:
            self._method_self_dep_cache[code] = result
        return result

    def fold_method_class_deps(self, func: Callable, args: tuple, state_hash: str) -> str:
        """Fold class-level code a cached method reaches into its key.

        At call time the real class IS known (``args[0]`` is the instance, or the
        class for a ``classmethod``), so ``self.helper`` resolves to
        ``type(self).helper`` and its source can be folded; ``self.RATE`` folds
        the class constant's value; ``super()`` folds the user base classes.

        Only CLASS-level members are folded. An instance attribute (in
        ``self.__dict__``) is already covered by hashing ``self`` itself, so it
        is skipped here -- ``getattr(class, attr)`` simply misses it.
        """
        self_name, attrs, uses_super = self._analyze_method_self_deps(func)
        if self_name is None or (not attrs and not uses_super):
            return state_hash
        if not args:
            return state_hash
        owner = args[0]
        # Resolve the class this method was called against, and confirm ``owner``
        # really is its ``self``/``cls`` (guard against a plain function whose
        # first parameter merely happens to be named ``self``). ``owner`` is the
        # class itself for a classmethod, else an instance.
        owner_class = owner if isinstance(owner, type) else type(owner)
        try:
            raw = inspect.getattr_static(owner_class, getattr(func, "__name__", ""))
        except (AttributeError, Exception):  # noqa: BLE001 - never break a call
            return state_hash
        target = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
        target = getattr(target, "__wrapped__", target)
        if target is not func:
            # Not this class's method (unbound call, or a look-alike param).
            return state_hash

        parts: list[str] = []
        # Transitive, not one-hop: a method reached via self may itself read a
        # class constant or call another method, and editing THAT must also
        # invalidate. Walk the reachable self-members, folding each once. Keyed
        # by attribute name -- within one class hierarchy ``self.X`` always
        # resolves to the same member -- so a ``seen`` set both dedups and stops
        # a mutually-recursive method pair from looping. Every member reached
        # is folded: the names come from the class's source, so they end.
        seen: set[str] = set()
        worklist: list[str] = list(attrs)
        while worklist:
            attr = worklist.pop()
            if attr in seen:
                continue
            seen.add(attr)
            try:
                member = inspect.getattr_static(owner_class, attr)
            except (AttributeError, Exception):  # noqa: BLE001 - never break a call
                continue  # instance-only attr (already in self's hash) or unresolved
            if isinstance(member, property):
                # Fold the getter's source, and follow what the getter reads.
                getter = member.fget
                if getter is not None:
                    try:
                        parts.append(f"p:{attr}:{hash_callable_source(getter)}")
                    except (OSError, TypeError, ValueError):
                        pass
                    _, sub_attrs, _ = self._analyze_method_self_deps(getter)
                    worklist.extend(a for a in sub_attrs if a not in seen)
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if inspect.isfunction(member) or inspect.ismethod(member):
                try:
                    parts.append(f"m:{attr}:{hash_callable_source(member)}")
                except (OSError, TypeError, ValueError):
                    continue
                # Recurse into what this method itself reaches through self.
                _, sub_attrs, _ = self._analyze_method_self_deps(member)
                worklist.extend(a for a in sub_attrs if a not in seen)
            elif not isinstance(member, (types.ModuleType, type)) and not callable(member):
                # A class-level DATA attribute (a constant). Fold its value.
                try:
                    parts.append(f"c:{attr}:{self._args.hash_payload((member,), {})}")
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                    continue
        if uses_super:
            for base in owner_class.__mro__[1:]:
                if base is object:
                    continue
                try:
                    parts.append(f"b:{base.__qualname__}:{hash_callable_source(base)}")
                except (OSError, TypeError, ValueError):
                    continue
        if not parts:
            return state_hash
        payload = ":".join(sorted(parts))
        return hashlib.sha256(f"{state_hash}:selfdeps:{payload}".encode("utf-8")).hexdigest()

    def pin_own_source(self, func: Callable, source_hash: str | None = None) -> str:
        """Identity of *func* itself, pinned per function object.

        The state hash's root component must describe the function the
        wrapper EXECUTES, not the live ``source_hashes[qualname]`` registry
        slot — a redefinition (notebook cell re-run) or a second lambda
        sharing the ``<lambda>`` qualname overwrites that slot, letting a
        stale wrapper store its results under the new function's identity.

        For named functions the pin is the registration-time source hash.
        Lambdas additionally fold `code_fingerprint`: two lambdas defined on
        the SAME source line share their source text, and only their code,
        nested code included, tells them apart.

        Taken when the decorator runs (*source_hash* is the hash registration
        just computed), because that is when the text on disk is the text the
        import compiled. Taken at the first call instead, a deploy that lands
        new files between import and that call would key the old body's
        result by the new body's text, and every restarted process would
        serve it as an ordinary hit.
        The first call still compares the loaded code with the file once, to
        say so. A pin not taken at decoration (the table is full) falls back to
        the same loaded-vs-disk check helpers get.
        """
        key = id(func)
        owner = func
        # A partial has no code of its own, and the fallbacks below then keyed
        # on a repr carrying the wrapped function's ADDRESS -- a different pin
        # in every process, so a cached partial never hit across processes.
        # What it wraps is the code that runs; what it binds is already in the
        # namespace name (`get_func_key`).
        func = unwrap_partials(func)
        # An id outlives nothing: once a redefined function dies, a later
        # definition can get its address. So an entry counts only while it
        # still refers to this very object, and the decorator (which passes
        # the hash it just computed) always takes a fresh pin.
        entry = self._own_pins.get(key)
        pin = entry[1] if entry is not None and source_hash is None and entry[0]() is owner else None
        if pin is not None:
            if self._own_pins_unverified and key in self._own_pins_unverified:
                self._own_pins_unverified.discard(key)
                if not loaded_code_matches_disk(func):
                    warn_source_changed_since_load(func)
                    # The file changed before the decorator ran -- after the
                    # module was compiled, while it was still importing -- so
                    # the text the pin was read from is not the code that runs.
                    # Keyed by what runs instead: the result belongs to the old
                    # body, and a process running the new one keys by the new
                    # text and recomputes.
                    live = bytecode_identity(func)
                    if live is not None:
                        pin = live
                        self._own_pins[key] = (entry[0], live)
            return pin
        at_decoration = source_hash is not None
        keyed_stat = stat_code_file(func)
        if source_hash is None:
            if loaded_code_matches_disk(func):
                source_hash = callable_identity(func)
            else:
                source_hash = bytecode_identity(func) or callable_identity(func)
                warn_source_changed_since_load(func)
                keyed_stat = None  # keyed by what runs, not by the file
        if keyed_stat is not None and id(keyed_stat[0]) not in CODE_KEYED_STATS:
            disk_digest = own_source_digest(func)
            if disk_digest is not None:
                CODE_KEYED_STATS[id(keyed_stat[0])] = (*keyed_stat, disk_digest)
        pin = source_hash
        if getattr(func, "__name__", "") == "<lambda>":
            code = getattr(func, "__code__", None)
            if code is not None:
                pin = hashlib.sha256(f"{pin}:{code_fingerprint(code)}".encode("utf-8")).hexdigest()
        self._own_pins_unverified.discard(key)
        if key in self._own_pins or len(self._own_pins) < self.OWN_PINS_MAX:
            self._own_pins[key] = (self._pin_owner_ref(owner, key), pin)
            if at_decoration:
                self._own_pins_unverified.add(key)
        return pin

    def _pin_owner_ref(self, owner: Any, key: int) -> Callable[[], Any]:
        """A zero-argument callable returning *owner* while it lives.

        A weak reference drops the pin when the function dies, so the table
        does not keep every redefinition alive; the few callables that take
        no weak reference are held strongly, which also keeps their id theirs.
        """
        pins, unverified = self._own_pins, self._own_pins_unverified

        def _drop(ref: weakref.ref) -> None:
            entry = pins.get(key)
            if entry is not None and entry[0] is ref:
                pins.pop(key, None)
                unverified.discard(key)

        try:
            return weakref.ref(owner, _drop)
        except TypeError:
            return lambda: owner

    def user_layers(self, fn: Any) -> list[Any]:
        """The user-code functions *fn* runs besides its own code
        (`callable_layers`): what its decorators wrap, the functions its
        closure holds. Raises `KeyBuildFailed` when they cannot all be found."""
        try:
            layers = callable_layers(fn)
        except UnwalkableLayers as e:
            raise KeyBuildFailed(
                "KEY-HELPERS-UNWALKABLE",
                f"cash cannot key the code {getattr(fn, '__qualname__', type(fn).__qualname__)} runs: {e}, "
                f"so the call ran uncached.",
                "Name what the result depends on with depends_on=[...].",
            ) from e
        return [layer for layer in layers if is_user_code_object(layer)]

    def _code_identity(self, fn: Any) -> tuple:
        """The bytecode-level identity of a callable, or ``()`` if it has none.

        Bytecode rather than source because a class defined in a notebook cell
        has no retrievable source at all: ``inspect.getsource`` resolves a class
        through ``sys.modules[cls.__module__].__file__``, and a notebook
        ``__main__`` has none. A function escapes this via ``co_filename``,
        which is why ``hash_callable_source`` works for helpers and not here.

        Comments and formatting are absent from bytecode, so they do not
        invalidate -- strictly better than source hashing. Docstrings live in
        ``co_consts`` and are masked out, so they do not either.

        Instance method (not static) because defaults/kwdefaults go through
        ``_value_identity`` -> ``self._args.hash_payload``: a default like
        ``def m(self, x=_MISSING)`` reprs as ``<object object at 0x...>``,
        the same address leak as a nested code object, just one layer up.
        """
        code = getattr(fn, "__code__", None)
        if code is None:
            return ()
        return (
            self._code_object_identity(code),
            self._value_identity(getattr(fn, "__defaults__", None)),
            self._value_identity(getattr(fn, "__kwdefaults__", None)),
        )

    def _code_object_identity(self, code: types.CodeType) -> tuple:
        """Structural identity of one ``types.CodeType``, recursing into any
        nested code object in ``co_consts`` instead of ``repr()``-ing it, and
        folding every OTHER const through ``_value_identity`` instead of
        ``repr()`` too.

        A lambda, a generator expression, or -- pre-3.12, before PEP 709
        inlined them -- a plain comprehension compiles to a NESTED code
        object stored in the enclosing method's ``co_consts``. Its ``repr()``
        is ``<code object <genexpr> at 0x...>``: a live memory address, fresh
        every process, which would give a class a different digest in every
        process. A nested code object carries no defaults of
        its own (those belong to the FUNCTION eventually built from it, not to
        the raw code object), so only co_code/co_consts/co_names apply here.

        The same holds for a plain (non-code) const: ``x in {'alpha',
        'beta'}`` compiles a ``frozenset`` straight into ``co_consts``, and
        ``repr()`` of a set/frozenset follows the table's internal
        (hash-order-dependent) iteration, which Python's per-process
        string-hash randomization changes. ``_value_identity``
        folds CONTENT instead, which is order-independent for a set/frozenset.
        """
        return (
            code.co_code,
            tuple(
                self._code_object_identity(k) if isinstance(k, types.CodeType) else self._value_identity(k)
                for k in code_consts_without_docstring(code)
            ),
            tuple(code.co_names),
        )

    def _value_identity(self, v: Any) -> str:
        """Address-free identity for a value that is not itself a code object.

        ``ArgHasher.hash_payload`` folds CONTENT and is the established,
        address-free tool used throughout this file for exactly this. What it
        cannot pickle goes to `_unpicklable_identity`, not to ``repr()``:
        ``repr()`` is a memory ADDRESS for the most ordinary unpicklable
        defaults (``key=lambda r: r``, ``lock=threading.Lock()``), different
        in every process, so the entry would never hit after a restart.

        ``CodeIdentity.class_surface_parts`` already refuses a ``repr()`` fallback, on the
        grounds that it "would reintroduce the address leak this member-content
        fold exists to avoid". This makes the two agree.
        """
        try:
            return self._args.hash_payload((v,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            return self._unpicklable_identity(v)

    def _unpicklable_identity(self, v: Any, _path: frozenset[int] = frozenset()) -> str:
        """Process-stable stand-in for a value ``ArgHasher.hash_payload`` refused.

        Recursive because ``__defaults__`` is hashed as a WHOLE TUPLE: one
        unpicklable element poisons the entire tuple, so every element that
        CAN be folded still is, and only the residue is approximated.

        A lambda, function or class is approximated by its CODE SURFACE, which
        is stable across processes and moves with an edit of the lambda's
        body; an address does neither.

        Everything else keeps its ``repr()`` when that repr is value-based
        (``Config(n=1)``): deterministic across processes, and it moves with
        the value. A repr that is only an address carries neither, so the
        value cannot be keyed: `KeyBuildFailed`, and the call runs uncached,
        as for an unhashable default of the cached function itself.
        """
        # However deep the containers nest: a lambda five lists down is code
        # like any other, and editing it must move the key. *_path* (the
        # containers being walked) ends a container that holds itself.
        if isinstance(v, (list, tuple, set, frozenset, dict)):
            if id(v) in _path:
                return f"<cycle:{type(v).__qualname__}>"
            _path = _path | {id(v)}
        if isinstance(v, (list, tuple, set, frozenset)):
            inner = [self._unpicklable_identity(x, _path) for x in v]
            if isinstance(v, (set, frozenset)):
                # Set iteration order follows the hash table, and string
                # hashing is randomized per process -- sort or reintroduce the
                # very instability this method exists to remove.
                inner.sort()
            return f"{type(v).__qualname__}[{'|'.join(inner)}]"
        if isinstance(v, dict):
            return (
                "dict["
                + "|".join(
                    sorted(
                        f"{self._unpicklable_identity(k, _path)}={self._unpicklable_identity(val, _path)}"
                        for k, val in v.items()
                    )
                )
                + "]"
            )
        try:
            return self._args.hash_payload((v,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            pass
        surface = self.code_surface_hash(v)
        if surface is not None:
            return f"code:{surface}"
        try:
            text = repr(v)
        except Exception as e:  # noqa: BLE001 - a __repr__ may raise
            logger.debug("[CORE] repr() failed while identifying %s: %s", type(v), e)
            text = ""
        # A repr with an address (``<object object at 0x...>``) is different
        # in every process and does not move when the object's content
        # does: nothing about the value can be keyed. A value-based repr
        # stands for the value, a hex literal in it included.
        if text and not _ADDRESS_REPR.search(text):
            return text
        cls = type(v)
        if isinstance(v, SYNC_TYPES):
            return f"<{cls.__module__}.{cls.__qualname__}>"
        name = f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', '?')}"
        raise KeyBuildFailed(
            "KEY-UNHASHABLE-DEFAULT",
            f"cash cannot key a {name} that code reaching the call holds as a default or constant: "
            f"it cannot be hashed and has no value-based repr, so a change to it could not be "
            f"seen and the call ran uncached.",
            f"get the value out of the signature -- build it in the body or pass it as an argument -- "
            f"or register a hasher with cash.register_hasher({cls.__qualname__}, ...).",
        )

    def _code_surface_own(self, obj: Any) -> str | None:
        """A digest of the user code *obj* itself carries, or ``None``.

        Its OWN surface only -- code merely referenced by that code is folded
        by :meth:`CodeIdentity.code_surface_hash`, which combines these per-object digests.
        The split is what keeps the memo below honest: memoizing a digest that
        included a referenced class would serve a stale answer when only that
        OTHER class is redefined (a notebook cell re-run), because *obj* is
        still the same object and still hits its memo entry.

        ``None`` means "cannot determine" and the caller must fall back to
        today's by-reference key. A hashing failure is not "cannot determine":
        it propagates, and the key build that asked runs the call uncached.

        Memoized on the object itself, following ``_user_class_src_cache``:
        redefining a class produces a NEW object and therefore a distinct dict
        key, so the memo cannot serve a stale entry. Keying on ``id()`` would
        be a correctness bug, since CPython recycles addresses.
        """
        # A `functools.partial` is its function plus arguments. The arguments
        # already reach the key (a partial pickles them, by value); its code is
        # the wrapped function's, which pickle names only by reference -- so
        # the wrapped function's code is keyed here, or an edit to its body
        # would keep the key.
        obj = cash_wrapped(unwrap_partials(obj))
        # Dispatch FIRST, memo read second. Every argument to a cached
        # function passes through here, and most are not a
        # class or callable at all -- a list, dict, set, numpy array,
        # DataFrame. Checking the dispatch before touching the memo means
        # those return None from a plain isinstance()/callable() check
        # instead of reaching the memo at all.
        is_type = isinstance(obj, type)
        if not (is_type or callable(obj)):
            return None
        if not is_user_code_object(obj):
            return None
        # By this point *obj* is a class or a callable, and while both are
        # hashable in the overwhelming common case, neither is guaranteed to
        # be (a __call__-implementing instance can set __hash__ = None).
        try:
            cached = self._code_surface_cache.get(obj)
        except TypeError:
            cached = None
        if cached is not None and cached[0] == _defaults_pin(cached[1]):
            return cached[2]
        layers: list[Any] = []
        if is_type:
            parts = self.class_surface_parts(obj)
        else:
            ident = self._code_identity(obj)
            if not ident:
                return None
            parts = [(getattr(obj, "__qualname__", "?"), "", ident)]
            # And the user functions it runs besides its own code: what a
            # decorator wraps, a function a closure holds. A decorator's
            # wrapper code is shared by every function it wraps; the
            # function itself is what tells them apart.
            layers = [obj, *self.user_layers(obj)]
            for layer in layers[1:]:
                parts.append((getattr(layer, "__qualname__", "?"), "runs", self._code_identity(layer)))
        if not parts:
            return None
        digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
        # A function's defaults are rebound (`f.__defaults__ = ...`) or
        # mutated in place on the same object: the memo holds only while
        # every default is immutable and the containers are unchanged.
        pin = _defaults_pin(layers)
        if pin is not None:
            try:
                self._code_surface_cache[obj] = (pin, layers, digest)
            except TypeError:
                pass  # unhashable object - skip the memo, keep the answer
        return digest

    def _iter_code_and_globals(self, obj: Any):
        """Yield ``(code object, globals)`` for the code *obj* carries."""
        carriers: list[Any] = []
        if isinstance(obj, type):
            for base in obj.__mro__:
                if base is object or is_opaque(base):
                    continue
                if not is_user_code_object(base):
                    continue
                carriers.extend(vars(base).values())
                # Same blind spot as class_surface_parts: a field declaring
                # default_factory has no class attribute to find in vars().
                fields_map = getattr(base, "__dataclass_fields__", None)
                if isinstance(fields_map, dict):
                    for fld in fields_map.values():
                        factory = getattr(fld, "default_factory", None)
                        if factory is not None and factory is not dataclasses.MISSING:
                            carriers.append(factory)
        else:
            carriers.append(obj)

        for member in carriers:
            if isinstance(member, (classmethod, staticmethod)):
                member = member.__func__
            if isinstance(member, property):
                accessors = [a for a in (member.fget, member.fset, member.fdel) if a]
            else:
                accessors = [member]
            for accessor in accessors:
                # Every layer: a function under two decorators was read one
                # layer down, and what its body names was never reached.
                for layer in (accessor, *self.user_layers(accessor)):
                    code = getattr(layer, "__code__", None)
                    glb = getattr(layer, "__globals__", None)
                    if isinstance(code, types.CodeType) and isinstance(glb, dict):
                        yield from walk_nested_code(code, glb)

    def _code_ref_targets(self, obj: Any) -> list[Any]:
        """User-code objects that *obj*'s code references by global name.

        Resolution happens on every call, deliberately. Only the (code,
        globals) pairs are memoized -- those are fixed for as long as *obj*
        exists -- because what a NAME is bound to can change underneath us,
        and that change is precisely what must invalidate.

        Names come from ``co_names``, i.e. what the code actually LOADS, and
        from *obj*'s annotations. An annotation is not always a hint that never
        runs: pydantic runs ``B``'s validators for a field ``b: B``, and a
        ``typing.get_type_hints`` builder constructs ``B`` from ``A``'s hints
        (see ``cash._annotation_refs``). A hint that really is inert costs a
        recompute when its class is edited, never a stale value.
        """
        pairs = None
        try:
            pairs = self._code_refs_cache.get(obj)
        except TypeError:
            pass  # unhashable - recompute each time rather than fail
        if pairs is None:
            pairs = tuple(self._iter_code_and_globals(obj))
            try:
                self._code_refs_cache[obj] = pairs
            except TypeError:
                pass

        targets: list[Any] = []
        seen_names: set[str] = set()

        def consider(value: Any) -> None:
            value = cash_wrapped(value)
            if value is None or value is obj:
                return
            if not (isinstance(value, type) or callable(value)):
                return
            try:
                if is_opaque(value) or not is_user_code_object(value):
                    return
            except Exception:  # noqa: BLE001 - never break a call
                return
            targets.append(value)

        for code, glb in pairs:
            for name in code.co_names:
                if name in seen_names:
                    continue
                seen_names.add(name)
                consider(glb.get(name))
            # A name spelled as a string: `getattr(MOD, "fun1")()`,
            # `globals()["fun1"]`. The string is a constant, not a loaded name,
            # so `co_names` does not have it. Resolved in the code's
            # module and in the user modules it loads; a string that only
            # happens to match a function costs a needless recompute, never a
            # stale value.
            names = [c for c in code.co_consts if isinstance(c, str) and c.isidentifier() and c not in seen_names]
            if not names:
                continue
            modules = [glb.get(n) for n in code.co_names]
            modules = [m for m in modules if isinstance(m, types.ModuleType) and is_user_module(m)]
            for name in names:
                seen_names.add(name)
                consider(glb.get(name))
                for module in modules:
                    consider(getattr(module, name, None))
        if isinstance(obj, type) or callable(obj):
            seen_ids = {id(t) for t in targets}
            for value in annotation_referents(obj, is_user_code_object):
                if id(value) not in seen_ids:
                    seen_ids.add(id(value))
                    consider(value)
        return targets

    def code_surface_hash(self, obj: Any) -> str | None:
        """A digest of *obj*'s code AND the user code that code reaches.

        What an argument or global directly carries is not enough: a class
        whose field factory constructs another class changes behaviour when
        THAT class is edited (``A(value=B(value=10))`` becomes
        ``A(value=B(value=1000))``), and nothing in the first class's own
        surface moves.

        Reachability is STATIC: names the code loads from its globals,
        transitively, however deep. Code selected at runtime (a class picked out of
        a dict) still cannot be followed, so this narrows the gap rather than
        closing it.
        """
        own = self._code_surface_own(obj)
        if own is None:
            return None
        reached = self._code_ref_closure(obj)
        if not reached:
            return own
        return hashlib.sha256(":".join([own, *sorted(reached)]).encode("utf-8")).hexdigest()

    def _code_ref_closure(self, obj: Any) -> list[str]:
        """Own-digests of every user-code object reachable from *obj*'s code.

        Breadth-first with an identity ``seen`` set, so a mutually-referential
        pair (``A.make`` returns ``B``, ``B.make`` returns ``A``) terminates
        instead of recursing forever. Reached objects are held in *keep* for
        the duration: ``id()`` is only unique among LIVE objects, and a
        collected one could otherwise let a later object reuse its id and be
        skipped as already-seen.
        """
        seen: set[int] = {id(obj)}
        keep: list[Any] = [obj]
        digests: list[str] = []
        frontier: list[Any] = [obj]
        while frontier:
            following: list[Any] = []
            for source in frontier:
                for target in self._code_ref_targets(source):
                    if id(target) in seen:
                        continue
                    seen.add(id(target))
                    keep.append(target)
                    if len(keep) > MAX_CODE_REF_TARGETS:
                        name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
                        raise KeyBuildFailed(
                            "KEY-HELPERS-UNWALKABLE",
                            f"cash cannot key the code {name} reaches: it does not end (over "
                            f"{MAX_CODE_REF_TARGETS} functions and classes; code that makes a new "
                            f"function on every read can cause this), so the call ran uncached.",
                            "Name what the result depends on with depends_on=[...] instead of "
                            "creating it on every read.",
                        )
                    digest = self._code_surface_own(target)
                    if digest is not None:
                        digests.append(f"{getattr(target, '__qualname__', '?')}:{digest}")
                    following.append(target)
            frontier = following
        return digests

    def _dataclass_field_parts(self, base: type, field_map: dict) -> list[tuple]:
        """Fold a dataclass's field metadata, which nothing else reaches.

        ``@dataclass`` moves the per-field declaration off the class attribute
        and into ``__dataclass_fields__``. The attribute that remains is just
        the default value, so a field's TYPE and its ``metadata=`` never
        reached the digest -- and ``__dataclass_fields__`` itself cannot be
        pickled, because ``Field.metadata`` is a ``mappingproxy``, so the
        generic fold cannot take it.

        A field description (``field(metadata={"desc": ...})``) is prompt
        text in every structured-output library there is -- it is not
        decoration, it is the instruction.

        Only the parts nothing else covers are folded. The default value is
        already the class attribute and is hashed there; folding it again would
        change nothing and cost a hash.
        """
        parts: list[tuple] = []
        for fname, f in sorted(field_map.items()):
            try:
                meta = self._args.hash_payload((dict(getattr(f, "metadata", {}) or {}),), {})
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                # An unpicklable metadata VALUE. Skip the metadata rather than
                # repr() it: a repr here would leak an object address and make
                # the digest differ between processes, which is worse than not
                # tracking it.
                meta = None
            # `str(f.type)` and not the object: an annotation may be a string
            # (`from __future__ import annotations`) or a class, and both spell
            # the same thing deterministically this way.
            parts.append((base.__qualname__, f"__dataclass_field__:{fname}", (str(getattr(f, "type", "")), meta)))
        return parts

    def _pydantic_field_parts(self, cls: type) -> list[tuple]:
        """Fold a pydantic model's field declarations.

        The counterpart to :meth:`_dataclass_field_parts`. Pydantic keeps them
        in ``model_fields`` -- a property on the class, so ``vars(cls)`` never
        sees it -- and the only other place they appear is the compiled trio
        skipped above, whose digest is different in every process.

        Without these parts a pydantic spec passed as an argument would have
        no stable digest, and would never be cached across processes.

        `description` is the load-bearing one -- it is the instruction sent to
        the model in every structured-output library there is.
        """
        # Guarded: this runs for EVERY class the surface walk sees, and
        # `model_fields` on a non-pydantic class could be a property that
        # computes something, or raises. Hashing must never be the thing that
        # breaks a call.
        try:
            fields = getattr(cls, "model_fields", None)
        except Exception:  # noqa: BLE001 - a descriptor of someone else's
            return []
        if not isinstance(fields, dict):
            return []
        parts: list[tuple] = []
        for fname, info in sorted(fields.items()):
            try:
                default = self._args.hash_payload((getattr(info, "default", None),), {})
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                default = None
            parts.append(
                (
                    cls.__qualname__,
                    f"__pydantic_field__:{fname}",
                    (
                        str(getattr(info, "annotation", "")),
                        getattr(info, "description", None),
                        getattr(info, "alias", None),
                        default,
                    ),
                )
            )
        return parts

    def class_surface_parts(self, cls: type, _path: tuple[type, ...] = ()) -> list[tuple]:
        """Every user-code member of *cls* and its user base classes.

        Walked in reverse MRO so a subclass override lands after the base it
        replaces, and sorted within each class so dict ordering cannot change
        the digest. *_path* is the chain of classes whose nested members led
        here, which ends a cycle.
        """
        parts: list[tuple] = self._pydantic_field_parts(cls)
        for base in reversed(cls.__mro__):
            # An OPAQUE base contributes nothing, so `cash.opaque(VendorBase)`
            # also stops a `Derived(VendorBase)` digest moving when the vendor
            # edits its own base. Without this the escape hatch worked only
            # when the opaque class was the one passed, which is not how
            # `docs/decorator.md` advertises it. Per-base and exact-match, so
            # opacity still does not inherit: marking a base does not make
            # `Derived` opaque, it only drops that base's own members.
            if base is object or is_opaque(base):
                continue
            if not is_user_code_object(base):
                continue
            for name, member in sorted(vars(base).items(), key=lambda kv: kv[0]):
                # __firstlineno__ (class attribute since Python 3.13, absent on
                # 3.10/3.11) records the class's first source line, which shifts
                # when a comment or blank line is added above it -- with no
                # code change at all. Skipping it is what keeps "comments do
                # not invalidate" (see _code_identity) true on 3.13+ too.
                if name in ("__dict__", "__weakref__", "__module__", "__firstlineno__"):
                    continue
                # The class docstring is documentation, the same as a method's
                # (masked in `_code_object_identity`), so it is folded as if
                # there were none -- the member itself stays, because for a
                # type whose surface is nothing else (a C type like
                # `_thread.lock`) dropping it left no surface at all and the
                # type was reported as unhashable code. Except on a pydantic
                # model: its docstring is the schema's `description`, which
                # structured-output libraries send to the model as the prompt.
                if name == "__doc__" and not self._pydantic_field_parts(base):
                    member = None
                # Pydantic v2 compiles three Rust objects onto every model.
                # They are DERIVED from the field declarations, and their
                # digest differs in every process. `model_fields` below carries the same
                # declarations and is stable, so this loses nothing.
                if name in PYDANTIC_COMPILED:
                    continue
                if name == "__dataclass_fields__" and isinstance(member, dict):
                    parts.extend(self._dataclass_field_parts(base, member))
                    continue
                target = member
                if isinstance(member, (classmethod, staticmethod)):
                    target = member.__func__
                elif isinstance(member, property):
                    for tag, accessor in (("get", member.fget), ("set", member.fset)):
                        ident = self._code_identity(accessor)
                        if ident:
                            parts.append((base.__qualname__, f"{name}.{tag}", ident))
                    continue
                # Unwrap decoration to reach the function whose __code__
                # actually reflects a body edit (mirrors the single-level
                # __wrapped__ unwrap in _analyze_method_self_deps). Without
                # this, @functools.wraps and @functools.lru_cache both hash
                # the WRAPPER's own generic dispatch code -- fixed regardless
                # of what the wrapped body says -- and @functools.
                # singledispatchmethod has no __wrapped__ or __code__ at all
                # (it exposes the underlying function as .func instead), and
                # would reach the data-attribute branch below as an
                # unchanging descriptor repr.
                outer = target
                target = getattr(target, "__wrapped__", target)
                if not hasattr(target, "__code__"):
                    func_attr = getattr(target, "func", None)
                    if func_attr is not None and hasattr(func_attr, "__code__"):
                        target = func_attr
                ident = self._code_identity(target)
                if ident and callable(outer):
                    # Every other user function it runs: the decorator's own
                    # wrapper, and the layers below the first ``__wrapped__``.
                    # Under two decorators the method body itself was never
                    # reached.
                    extra = [
                        layer
                        for layer in (outer, *self.user_layers(outer))
                        if layer is not target and isinstance(layer, types.FunctionType) and is_user_code_object(layer)
                    ]
                    if extra:
                        ident = (ident, tuple(self._code_identity(layer) for layer in extra))
                if callable(member):
                    if ident:
                        # When `ident` was reached by UNWRAPPING (``.func`` /
                        # ``__wrapped__``), it describes the inner function and
                        # says nothing about the state the wrapper itself
                        # carries: ``functools.partial(scale, 3)`` and
                        # ``partial(scale, 4)`` unwrap to the same ``scale``
                        # and collided. Fold the wrapper's own content too,
                        # exactly as the non-callable branch does for a
                        # ``partialmethod``. Skipped when nothing was
                        # unwrapped, because a plain method's own pickle is
                        # its module path -- which would make every class's
                        # digest depend on the module it lives in.
                        if target is not member:
                            try:
                                own = self._args.hash_payload((member,), {})
                            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                                own = None
                            if own is not None:
                                parts.append((base.__qualname__, name, (ident, own)))
                                continue
                        parts.append((base.__qualname__, name, ident))
                        continue
                    # A callable member with NO reachable ``__code__``: a
                    # nested class (``class Outer: inner = Inner``), a
                    # ``functools.partial``, a callable instance. Its code
                    # (``Inner.f``) and its content (``partial(scale, 3)``
                    # against ``partial(scale, 4)``) are both folded.
                    #
                    # The nested walk recurses into ``CodeIdentity.class_surface_parts``
                    # DIRECTLY, not through the memoized ``CodeIdentity.code_surface_hash``,
                    # and a cycle ends on the PATH that led to it, never on a
                    # set shared across the walk. A shared set would make the
                    # digest depend on which class happened to be hashed first
                    # (the memo would hold a cut result for one order and a
                    # full one for the other) -- reintroducing exactly the
                    # cross-process instability ``_value_identity`` avoids.
                    # The path is fixed by *cls* alone, so every
                    # process gets the same answer regardless of order. No
                    # depth bound: a nested class is followed however deep.
                    nested = None
                    inner_cls = member if isinstance(member, type) else type(member)
                    if is_user_code_object(inner_cls):
                        if inner_cls is cls or inner_cls in _path:
                            nested = f"cycle:{inner_cls.__qualname__}"
                        else:
                            sub_parts = self.class_surface_parts(inner_cls, (*_path, cls))
                            if sub_parts:
                                nested = hashlib.sha256(
                                    repr(sub_parts).encode("utf-8"),
                                ).hexdigest()
                    try:
                        content = self._args.hash_payload((member,), {})
                    except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                        content = None
                    if nested is not None or content is not None:
                        parts.append((base.__qualname__, name, (nested, content)))
                else:
                    # A non-callable member. Fold its OWN content
                    # unconditionally -- a descriptor like
                    # functools.partialmethod carries bound state (.args)
                    # that lives on the descriptor ITSELF, not on the inner
                    # function `ident` above resolved through .func, and a
                    # plain object that happens to expose an unrelated `.func`
                    # attribute must not have its OTHER state go invisible
                    # just because that lookup succeeded (a partialmethod's
                    # bound arguments, an unrelated object's own attributes).
                    # Fold `ident` TOO
                    # when reachable, so a non-callable descriptor that ALSO
                    # wraps a real function body -- functools.
                    # singledispatchmethod, functools.cached_property, both
                    # confirmed to expose .func without __wrapped__ or
                    # __code__ of their own -- has that body participate as
                    # well. Folding only one half silently drops whichever
                    # state that particular member happens to carry.
                    #
                    # No repr() fallback here (unlike _value_identity):
                    # falling back to repr() on this specific path would
                    # reintroduce the address leak this member-content fold
                    # exists to avoid (a class attribute's repr is often an
                    # address). If content can't
                    # be folded and no `ident` was found either, dropping the
                    # member is strictly safer than a non-deterministic repr.
                    try:
                        content = self._args.hash_payload((member,), {})
                    except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                        content = None
                    if ident or content is not None:
                        parts.append((base.__qualname__, name, (ident, content)))

        # Dataclass field factories. ``dataclasses`` DELETES the class
        # attribute when a field declares ``default_factory``, so the
        # ``vars()`` walk above cannot see it -- ``getattr_static`` raises
        # AttributeError for that name. The factory is nonetheless code that
        # decides what every instance holds: editing
        # ``field(default_factory=lambda: B(0))`` to ``B(999)`` changes what
        # ``A()`` produces.
        #
        # Read from ``__dataclass_fields__`` rather than calling
        # ``dataclasses.fields()``: the latter raises on a non-dataclass and
        # skips pseudo-fields, and this must never raise.
        fields_map = getattr(cls, "__dataclass_fields__", None)
        if isinstance(fields_map, dict):
            for fname, fld in sorted(fields_map.items()):
                factory = getattr(fld, "default_factory", None)
                # ``MISSING`` is a sentinel INSTANCE, not None; compare by
                # identity against the one dataclasses hands out.
                if factory is None or factory is dataclasses.MISSING:
                    continue
                ident = self._code_identity(factory)
                if ident:
                    parts.append((cls.__qualname__, f"field:{fname}:factory", ident))
        return parts

    def user_class_source_hash(self, cls: type) -> str:
        """Memoized source hash of a USER class.

        A class's source cannot change within a running interpreter: editing the
        file and re-importing produces a NEW class object (a distinct dict key),
        so the hash is computed once per class object and reused on every
        subsequent call. The per-call cost of the instance channel below is then
        a cheap object-graph walk plus dict lookups -- never source I/O.

        Source-first, surface-as-fallback. Both of this method's callers
        (``CodeIdentity.instance_class_source_parts``, directly and via
        ``GlobalsFold.fold_read_globals``) gate on ``is_user_class`` -> ``is_user_module``,
        which requires ``__file__`` -- so every class actually reachable here
        already has retrievable source, and ``inspect.getsource`` succeeds. The
        class-aware surface (``CodeIdentity.code_surface_hash``) only engages on
        ``SOURCE_RETRIEVAL_ERRORS`` -- a class truly without source, e.g. a
        notebook cell's ``__main__`` has no ``__file__`` -- or when this method
        is reached some other way. Whole-class source is preferred because it
        sees a body edit under ``@functools.wraps``, ``@lru_cache`` or
        ``@singledispatchmethod`` (source is just text), where
        ``CodeIdentity.class_surface_parts`` walks the WRAPPER.
        """
        cached = self._user_class_src_cache.get(cls)
        if cached is not None:
            return cached
        h = source_digest(cls)
        if h is None:
            # No source to hash (or it doesn't parse). A class has no
            # __code__, so the callable fallback would key it on its name
            # alone; the class-aware surface sees its members.
            h = self.code_surface_hash(cls) or hash_callable_source(cls)
        self._user_class_src_cache[cls] = h
        return h

    def instance_class_source_parts(
        self,
        value: Any,
        _seen: set | None = None,
        own_pkg: str | None = None,
    ) -> list[tuple[str, str]]:
        """``(qualname, source-hash)`` for the user classes behind an INSTANCE.

        A cached function that reads a pre-built module-level object -- ``pre =
        MyTransformer()`` imported and dropped into a pipeline -- cannot be only
        VALUE-hashed: its ``__dict__`` pickle carries no method source, so an
        edit to ``MyTransformer.transform`` would leave the key unchanged and
        serve a stale result. Fold the source
        of the instance's class -- and of the user-class instances it holds,
        however deep -- so a method-body edit invalidates.

        The walk recurses only into user-class instances: a third-party object
        (a fitted sklearn estimator, a numpy array) is not user-editable and its
        internals must not churn the key, and stopping there also bounds the cost
        on real pipelines. A user class reachable only through a third-party
        container is not folded here; `CodeArgs.carrier_parts`, which a data
        global also goes through, searches library objects for it.
        """
        if _seen is None:
            _seen = set()
        parts: list[tuple[str, str]] = []
        # Depth-first with an explicit stack, however deep the instances nest
        # (a linked list of user objects is as deep as it is long).
        # ``_seen`` ends cycles.
        stack = [value]
        while stack:
            item = stack.pop()
            if id(item) in _seen:
                continue
            _seen.add(id(item))
            cls = type(item)
            if is_user_class(cls, own_pkg):
                try:
                    parts.append((cls.__qualname__, self.user_class_source_hash(cls)))
                except SOURCE_RETRIEVAL_ERRORS:
                    pass
            held = getattr(item, "__dict__", None)
            if isinstance(held, dict):
                found = [
                    inner
                    for attr_val in held.values()
                    for inner in iter_contained(attr_val)
                    if is_user_class(type(inner), own_pkg)
                ]
                stack.extend(reversed(found))
        return parts
