"""Main Cash class - decorator-based caching with automatic dependency tracking.

Provides the `Cash` entry point for ``@cash.cache`` function-level
caching and the `Cash.notebook` bridge for Jupyter integration.
"""

from __future__ import annotations

import ast
import atexit
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import os
import pickle
import sys
import textwrap
import threading
import time
import types
import weakref
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from .backends import CacheBackend, CacheMetadata, CascadingBackend
from .backends.factory import build_backend_from_config

if TYPE_CHECKING:
    from .ui.explorer import CacheExplorer
from .backends.serialization import get_serializer
from .config import CashConfig, get_config
from .data_source import DataSource
from .dependency_state import DependencyStateHasher, SysModulesHelperResolver
from .diagnostics import (
    format_diagnostic,
    log_diagnostic,
    warn_diagnostic,
    warn_diagnostic_message,
)
from .exceptions import (
    SOURCE_RETRIEVAL_ERRORS,
    CacheExpiredError,
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
    CashImpureFunctionError,
    CashImpurityWarning,
)
from .graph import DependencyGraph
from .notebook.file_dep_snapshot import ACTIVE_CONFIG
from .notebook.analysis import CodeAnalyzer

# The decorator path reuses the notebook path's randomness detector verbatim
# so the two cannot diverge on what counts as an unseeded draw.
# Imported from the submodule directly, like CodeAnalyzer above, to sidestep
# ``notebook/__init__``'s lazy circular-import chain; ``randomness`` itself only
# depends on ``..exceptions``, so there is no cycle.
from .notebook.annotations import parse_annotation_line
from .notebook.randomness import (
    CashRandomnessWarning,
    RandomnessDetector,
    describe_random_call,
)
from .purity_analyzer import (
    PurityReport,
    bindings_changed,
    get_analyzer,
    is_mock,
    resolve_binding,
)
from .source_norm import (
    bytecode_identity,
    loaded_class_identity,
    loaded_code_matches_disk,
    source_identity_digest,
)
from .utils import resolve_main_module

# Configure Logging
logger = logging.getLogger(__name__)

# Two fix lines are shared by more than one emit site, because more than one
# site tells the same story: a global whose value cannot be hashed is one
# problem reached through two channels (a function's own globals and a
# helper's), and a refused write is one problem whether the value went whole or
# as a chunked manifest. Sharing the text is what keeps the two halves of each
# pair from drifting into two different pieces of advice for one doc section.
_UNHASHABLE_GLOBAL_FIX = (
    "register a hasher for its type with cash.register_hasher, or read the "
    "part the result actually depends on -- a URL, a connection string -- "
    "instead of the live object."
)
_STORE_FAILED_FIX = (
    "read the exception: a full disk, a cache_dir you cannot write to, or a "
    "value that cannot be pickled -- return the data, not the handle that "
    "produced it."
)

def get_ipython():
    """Return the live IPython shell, or ``None``.

    Resolved on FIRST CALL rather than at import. ``from IPython import
    get_ipython`` looks cheap but pulls the whole package -- measured at ~4s of
    the ~10s ``import cash``, most of it ``IPython.terminal.embed``. That cost
    sits in front of every kernel start and every subprocess a test spawns, and
    a test running three subprocesses tripped the 30s per-test timeout on
    imports alone.

    Outside IPython this is the common case and stays cheap: ``sys.modules`` is
    consulted first, so a plain script never imports IPython at all. Inside a
    notebook IPython is already imported, so the lookup is free.
    """
    ipython_module = sys.modules.get("IPython")
    if ipython_module is None:
        # Not already imported. In a notebook it always is, so reaching here
        # means we are not in one -- do not pay the import to find that out.
        return None
    try:
        return ipython_module.get_ipython()
    except Exception:  # noqa: BLE001 - never break a call over shell detection
        return None

# Sentinel object used by wrapper helpers to signal a cache miss without
# conflicting with any legitimate cached value (including None).
_CACHE_MISS = object()

# Types that can carry no user code. `_iter_code_carriers` walks EVERY argument
# of every cached call, so this is checked once per element of a container and
# is deliberately a module-level global (one LOAD_GLOBAL) rather than a class
# attribute (an extra attribute lookup per element). Ordered by how often each
# actually shows up in an argument, because `in` on a tuple scans in order.
#
# Tested as `type(v) in _CODELESS_PRIMS`, NOT `isinstance(v, _CODELESS_PRIMS)`:
# isinstance is true for SUBCLASSES, so a user class deriving from str/int/
# float/bytes -- and every IntEnum member, whose type is the user's own enum
# class -- was skipped here and never reached the fold. That is a missed
# invalidation on exactly the bug class this feature exists to fix.
#
# And a TUPLE, not a frozenset: `x in frozenset` hashes `x`, and a class whose
# metaclass defines __eq__ without __hash__ is itself unhashable (see
# `_is_opaque`, which carries a test for that shape), so a frozenset raises
# TypeError on an instance of one and fails the whole fold open. Tuple `in`
# compares with `is`/`==` and never hashes. Measured over 200k elements it is
# also FASTER than the isinstance form it replaces: 4.0ms vs 5.1ms (ints),
# 3.2ms vs 3.4ms (strs), 3.6ms vs 4.3ms (mixed).
_CODELESS_PRIMS = (str, int, float, bool, type(None), bytes, complex, bytearray)

#: Exact types of the builtin containers walked below. An instance of a SUBCLASS
#: of one of these is still walked for its contents, but it is also a user
#: object whose class carries code, so it additionally contributes that class.
_BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset)

P = ParamSpec("P")
T = TypeVar("T")


def _object_state(value: Any) -> dict:
    """Return an object's instance state as a name -> value dict, covering both
    ``__dict__`` and ``__slots__`` (collected across the MRO so slots declared
    on base classes are included). Builtins and leaf values yield ``{}``. Used
    so set-canonicalisation reaches a set buried inside a ``__slots__`` object,
    not just a ``__dict__``-backed one.
    """
    state: dict = {}
    obj_dict = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        state.update(obj_dict)
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__") or name in state:
                continue
            try:
                state[name] = getattr(value, name)
            except AttributeError:
                pass  # slot declared but never assigned
    return state


def _tag_subtype(value: Any, base: type, canon: Any) -> Any:
    """Wrap *canon* with the concrete type when *value* is a SUBCLASS of *base*.

    tuple/dict/list subclasses (namedtuple, OrderedDict, defaultdict, ...) were
    canonicalised into their base type, so ``f(P(1,2))`` and ``f(Q(1,2))`` --
    two distinct namedtuple types with equal values -- collided onto one cache
    key and the second call was served the first's result (a silent wrong-HIT).
    The ``__cash_obj__`` path already tags arbitrary objects with their type for
    exactly this reason; the container branches did not.

    An EXACT base instance is returned untouched, so ordinary tuple/dict/list
    arguments keep byte-identical keys and no cache is invalidated.
    """
    if type(value) is base:
        return canon
    t = type(value)
    return ("__cash_subtype__", f"{t.__module__}.{t.__qualname__}", canon)


def _stable_key_repr(value: Any, _depth: int = 0) -> Any:
    """Rewrite *value* into a form whose pickled bytes are independent of
    set/dict iteration order (which depends on PYTHONHASHSEED for str/bytes
    elements). Sets/frozensets and dict items are sorted by their pickled
    element bytes; lists/tuples keep order. Recurses into arbitrary objects via
    their ``__dict__`` so a set buried inside a dataclass is canonicalised too.
    Leaf values pass through unchanged.
    """
    if _depth > 50:
        return value
    if isinstance(value, (set, frozenset)):
        items = [_stable_key_repr(v, _depth + 1) for v in value]
        items.sort(key=lambda x: pickle.dumps(x, protocol=4))
        tag = "__cash_frozenset__" if isinstance(value, frozenset) else "__cash_set__"
        return (tag, tuple(items))
    if isinstance(value, dict):
        # A dict SUBCLASS (OrderedDict, defaultdict) may be order-significant,
        # so preserve item order and tag the type; a plain dict is order-
        # insensitive by ``==`` and keeps the sorted, untagged form so its key
        # is byte-identical to before this change.
        subclass = type(value) is not dict
        items = [
            (_stable_key_repr(k, _depth + 1), _stable_key_repr(v, _depth + 1))
            for k, v in value.items()
        ]
        if not subclass:
            items.sort(key=lambda kv: pickle.dumps(kv[0], protocol=4))
        canon = ("__cash_dict__", tuple(items))
        return _tag_subtype(value, dict, canon)
    if isinstance(value, list):
        canon = ("__cash_list__", tuple(_stable_key_repr(v, _depth + 1) for v in value))
        return _tag_subtype(value, list, canon)
    if isinstance(value, tuple):
        canon = tuple(_stable_key_repr(v, _depth + 1) for v in value)
        return _tag_subtype(value, tuple, canon)
    obj_state = _object_state(value)
    if obj_state:
        # Arbitrary object (dataclass, __slots__ class, ...) - canonicalise its
        # instance state, tagged with the type so two types don't collide.
        return ("__cash_obj__", type(value).__qualname__,
                _stable_key_repr(obj_state, _depth + 1))
    return value


def _canonicalize_dict_order(value: Any, _depth: int = 0) -> Any:
    """Rebuild every ``dict`` in *value* in canonical (sorted-key) order so that
    two dicts that are equal but for insertion order pickle to identical bytes
   . Recurses through ``dict``/``list``/``tuple``; other types pass
    through unchanged. ``list``/``tuple`` order is preserved (semantic), and the
    dict TYPE is kept, so a payload whose dicts are already sorted (e.g. the
    top-level kwargs canonicalised by ``_normalize_call_args``) is byte-identical
    to before — only out-of-order dict *values* change.

    Keys are ordered by their pickled bytes (a total order that never raises on
    mixed key types); on an unpicklable key it falls back to ``repr(key)``, then
    to insertion order — it never crashes. Sets are intentionally NOT handled
    here: a payload containing a set is routed through ``_stable_key_repr``
    instead, which canonicalises sets (including frozenset dict keys, whose
    pickle bytes are PYTHONHASHSEED-dependent) deterministically.
    """
    if _depth > 50:
        return value
    # Same short-circuit as ``_contains_set``: a primitive has no dict inside
    # to reorder, and this runs once per element of every container argument.
    if type(value) in _CODELESS_PRIMS:
        return value
    if isinstance(value, dict):
        # A dict SUBCLASS keeps insertion order (it may be semantic) and is
        # tagged with its type; a plain dict is sorted (order-insensitive) and
        # untagged, so its key is byte-identical to before this change.
        subclass = type(value) is not dict
        items = [
            (k, _canonicalize_dict_order(v, _depth + 1)) for k, v in value.items()
        ]
        if not subclass:
            try:
                items.sort(key=lambda kv: pickle.dumps(kv[0], protocol=4))
            except Exception:  # noqa: BLE001 - unpicklable key: degrade, never crash
                try:
                    items.sort(key=lambda kv: repr(kv[0]))
                except Exception:  # noqa: BLE001 - unsortable even by repr: keep order
                    pass
        canon = dict(items)
        return _tag_subtype(value, dict, canon)
    if isinstance(value, list):
        canon = [_canonicalize_dict_order(v, _depth + 1) for v in value]
        return _tag_subtype(value, list, canon)
    if isinstance(value, tuple):
        canon = tuple(_canonicalize_dict_order(v, _depth + 1) for v in value)
        return _tag_subtype(value, tuple, canon)
    return value


def _contains_set(value: Any, _depth: int = 0) -> bool:
    """True if *value* contains a set/frozenset anywhere (recursively, including
    inside objects). Gates the canonicalisation so ordinary args are untouched."""
    if _depth > 50:
        return False
    # An exact builtin primitive cannot contain anything, so it cannot contain
    # a set. Without this the fall-through below called ``_object_state`` on
    # EVERY element -- which walks ``type(value).__mro__`` looking for
    # ``__slots__`` -- so hashing a 10k-element list of ints made 10k such
    # walks per cache hit. Exact-type test, matching ``_CODELESS_PRIMS``'s own
    # contract: a str/int SUBCLASS can carry a ``__dict__`` holding a set and
    # must still be walked.
    if type(value) in _CODELESS_PRIMS:
        return False
    if isinstance(value, (set, frozenset)):
        return True
    if isinstance(value, dict):
        return any(
            _contains_set(k, _depth + 1) or _contains_set(v, _depth + 1)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_set(v, _depth + 1) for v in value)
    obj_state = _object_state(value)
    if obj_state:
        return any(_contains_set(v, _depth + 1) for v in obj_state.values())
    return False

__all__ = ["Cash", "CacheExplanation"]


# Reason codes returned by `Cash._explain_call` / ``f.explain(...)``.
# Kept as module-level constants so external code can match against them
# without string-typo risk: ``if e.reason == EXPLAIN_HIT: ...``.
EXPLAIN_HIT = "hit"
EXPLAIN_KEY_UNCOMPUTABLE = "key_uncomputable"
EXPLAIN_NO_ENTRY = "no_entry"
EXPLAIN_TTL_EXPIRED = "ttl_expired"
EXPLAIN_FILE_CHANGED = "file_changed"
EXPLAIN_DISABLED = "disabled"


@dataclass(frozen=True)
class CacheExplanation:
    """Why a specific call would hit or miss the cache *right now*.

    Returned by ``f.explain(*args, **kwargs)`` on any ``@cash.cache``-wrapped
    function. Inspecting an explanation does NOT mutate stats, call the
    underlying function, or write to the backend - it only reads what the
    cache already knows.

    Attributes:
        would_hit: True if the next call with these args would return a
            cached value (without recomputing).
        reason: Short stable string identifying the outcome. One of:
            ``"hit"``, ``"key_uncomputable"``, ``"no_entry"``,
            ``"ttl_expired"``, ``"file_changed"``, ``"disabled"``.
        func_name: Module-qualified name of the cached function.
        cache_key: The cache key computed for these args, or ``None``
            when key generation failed (``reason == "key_uncomputable"``).
        details: Reason-specific extras. Common keys:

            * ``hit``: ``cached_at`` (unix ts), ``execution_time_saved`` (s),
              ``cache_age_seconds``.
            * ``key_uncomputable``: ``arg_type`` (qualname or ``"<unknown>"``),
              ``error`` (exception type+message), ``hint``.
            * ``no_entry``: ``hint``, and ``why`` -- what this process
              knows about the key: never stored and why, stored and since
              evicted, or which part of the key moved since the last call
              (``new arguments``, ``code or state changed``, ...).
            * ``ttl_expired``: ``ttl_seconds``, ``age_seconds``, ``cached_at``
              when the decorator's ttl ran out; ``why`` when the entry
              expired under the ttl it was written with.
            * ``file_changed``: ``changed_files`` (dict of path -> reason),
              ``file_deps``.
            * ``file_deps`` (on ``hit`` and ``file_changed``): every file the
              entry recorded, with the fingerprint it was checked against.

    ``entry_id`` is the id ``cash inspect --function`` lists and
    ``cash clear --entry`` accepts.
    """

    would_hit: bool
    reason: str
    func_name: str
    cache_key: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_id(self) -> str | None:
        """The id `cash inspect` lists and `cash clear --entry` takes."""
        return entry_id_of(self.cache_key) if self.cache_key else None

    def __str__(self) -> str:
        verdict = "HIT" if self.would_hit else "MISS"
        lines = [f"[{verdict}] {self.func_name} - {self.reason}"]
        if self.cache_key:
            lines.append(f"  cache_key: {self.cache_key}")
            lines.append(f"  entry_id: {self.entry_id}")
        for k, v in self.details.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class _StreamingCachedIterator:
    """Passes the producer's items through as they arrive, caching at the end.

    Returned on a MISS. `@cash.cache` should not change how a function
    behaves, and for a generator it used to: cash drained the whole thing
    before returning anything, so a streamed response arrived all at once
    after the full latency. Measured on a token stream -- 494ms to first item
    uncached, 2444ms cached, the entire completion in one go.

    Same surface as the replay iterator, deliberately: no `send`/`throw`,
    because a cached generator cannot support them on the hit either.
    """

    __slots__ = ("_gen",)

    def __init__(self, gen):
        self._gen = gen

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def close(self):
        """Abandon the stream. Nothing is cached -- see `_stream_and_store`."""
        self._gen.close()

    def send(self, value):
        raise AttributeError(
            "cached generator: .send() is not supported. If you "
            "need send() semantics, the function cannot be cached."
        )

    def throw(self, *args, **kwargs):
        raise AttributeError(
            "cached generator: .throw() is not supported. If you "
            "need throw() semantics, the function cannot be cached."
        )


class _ChunkedCachedIterator:
    """Lazy iterator that reads cached chunks from the backend on demand.

    Used by `Cash.cache` for iterator-returning functions whose
    output spans multiple backend keys. Each chunk is fetched only
    when the user iterates into it; chunks the user never reaches are
    never read. The retrieval is RAM-bounded by chunk size.

    The class satisfies the iterator protocol (``iter(x) is x``,
    ``__next__``, ``close``); generator-specific methods (``send``,
    ``throw``) raise ``AttributeError`` - the cached iterator is a
    replay of stored values, not a coroutine.

    Args:
        cash: The owning `Cash` instance (used for backend access).
        cache_key: The canonical key under which the manifest is stored.
            Chunk keys are derived as ``f"{cache_key}:chunk_{i}"``.
        n_chunks: Total chunk count, taken from the manifest at construction.

    The iterator is robust to chunk loss: if ``backend.get`` returns
    ``(None, None)`` for any chunk (e.g. RAM-only eviction), iteration
    terminates cleanly via ``StopIteration``. The next call to the
    decorated function will see a cache miss and recompute.
    """

    __slots__ = ("_cash", "_cache_key", "_n_chunks",
                 "_chunk_index", "_current_chunk_iter", "_closed")

    def __init__(self, cash: Any, cache_key: str, n_chunks: int):
        self._cash = cash
        self._cache_key = cache_key
        self._n_chunks = n_chunks
        self._chunk_index = 0
        self._current_chunk_iter = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        while True:
            if self._current_chunk_iter is not None:
                try:
                    return next(self._current_chunk_iter)
                except StopIteration:
                    self._current_chunk_iter = None
                    # Fall through to load the next chunk.
            if self._chunk_index >= self._n_chunks:
                raise StopIteration
            chunk_key = f"{self._cache_key}:chunk_{self._chunk_index}"
            _, chunk = self._cash.backend.get(chunk_key)
            self._chunk_index += 1
            if chunk is None:
                # Chunk lost (eviction, partial cleanup). Safe termination -
                # the next call to the decorated function will see a cache
                # miss on the manifest and recompute from scratch.
                raise StopIteration
            self._current_chunk_iter = iter(chunk)

    def close(self):
        """Stop iteration. Subsequent ``next()`` raises ``StopIteration``."""
        self._closed = True
        self._current_chunk_iter = None

    def send(self, value):
        raise AttributeError(
            "cached generator: .send() is not supported on chunked "
            "iterators. The cached iterator replays values from the "
            "backend. If you need send() semantics, the function "
            "cannot be cached."
        )

    def throw(self, *args, **kwargs):
        raise AttributeError(
            "cached generator: .throw() is not supported on chunked "
            "iterators. If you need throw() semantics, the function "
            "cannot be cached."
        )


def _make_opaque_issue(func_name: str, opaque_list: str) -> Any:
    """Build a synthetic `PurityIssue` for opaque callees
    encountered in ``strict`` mode. Defined at module scope so the
    ``_surface_purity`` import stays local."""
    from .purity_analyzer import ISSUE_IMPURE_CALL, PurityIssue
    return PurityIssue(
        kind=ISSUE_IMPURE_CALL,
        description=f"opaque callees (strict): {opaque_list}",
        where=func_name,
        line=0,
    )


def _format_issues_summary(func_name: str, issues: list[Any]) -> str:
    """Pretty-print a list of `PurityIssue` records, grouped
    by their ``where`` field. Used by both the warning body and the
    strict-mode exception body so users get the same diagnostic.
    """
    by_where: dict[str, list[Any]] = {}
    for i in issues:
        by_where.setdefault(i.where, []).append(i)
    lines = []
    for where in sorted(by_where):
        # The defining file, so a finding in a helper names the helper's
        # module -- the warning's own header names the CALL site's file.
        filename = next((getattr(i, "filename", "") for i in by_where[where]
                         if getattr(i, "filename", "")), "")
        lines.append(f"  in {where} ({filename}):" if filename else f"  in {where}:")
        for issue in by_where[where]:
            line_part = f"line {issue.line}: " if issue.line else ""
            lines.append(f"    {line_part}[{issue.kind}] {issue.description}")
    return "\n".join(lines)


def _is_one_shot_iterator(value: Any) -> bool:
    """Return True if *value* is its own iterator (a one-shot consumable).

    Matches Python generators, ``map``/``filter``/``zip`` results, and
    custom iterators that return ``self`` from ``__iter__``. Returns
    False for collections (``list``/``dict``/``set``/``tuple``/``str``/
    ``range``) which are iterable but return fresh iterators on
    ``iter()`` - those are safely cacheable as-is.
    """
    try:
        return iter(value) is value
    except TypeError:
        return False

# id(code object) -> (the object itself, its source digest). Keyed by IDENTITY,
# and the object is retained so the id cannot be recycled under us -- the same
# guard `function_tracker._source_cache` uses.
#
# NOT keyed on the code object directly, which was the first attempt: CodeType
# implements __eq__/__hash__ BY VALUE, and co_filename is not part of that
# equality, so two helpers with the same body in different modules share one
# dict slot. That made `test_real_helper_change_still_recomputes` fail
# reproducibly under xdist while passing alone -- a stale digest served across
# tests through a module-level memo.
#
# A redefinition (reloaded module, re-run cell) compiles a NEW code object, so
# identity keying still cannot serve a digest for code that is no longer
# running. Editing a .py file WITHOUT reloading leaves the old code object
# live, and the old digest is then the correct answer.
#
# Load-bearing, not a micro-optimisation. `_hash_callable_source` is the live
# per-call identity of every transitive helper, and it calls
# `inspect.getsource`, which re-reads and RE-TOKENISES the source block on
# every call. Measured on a 2-helper function: 8700 tokenizer calls per 300
# cache hits, and 37ms of a 65ms key computation.
#
# Module-level rather than per-instance: the digest depends only on the code
# object, so two Cash instances cannot legitimately disagree about it.
_SOURCE_HASH_MEMO: dict = {}
_SOURCE_HASH_MEMO_MAX = 4096

#: Seeding calls, by the last segment of their dotted name. ``seed`` alone is
#: too common a method name, so it only counts under a ``random`` prefix.
_SEEDING_CALLS = frozenset({
    "default_rng", "RandomState", "Random", "manual_seed", "SeedSequence",
    "PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64",
})


def _seed_access_path(node: ast.AST) -> tuple[str, tuple[tuple[str, Any], ...]] | None:
    """``settings.sim.seed`` -> ``("settings", (("attr", "sim"), ("attr", "seed")))``;
    ``opts["seed"]`` -> ``("opts", (("item", "seed"),))``; None for anything
    else (a call, a computed key, an expression)."""
    path: list[tuple[str, Any]] = []
    while True:
        if isinstance(node, ast.Attribute):
            path.append(("attr", node.attr))
            node = node.value
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            path.append(("item", node.slice.value))
            node = node.value
        else:
            break
    if not isinstance(node, ast.Name):
        return None
    return node.id, tuple(reversed(path))


def _seed_parameters(src: str) -> dict[str, tuple[str, str, bool, tuple]]:
    """``{seed expression: (call, root name, root is a parameter, path)}``.

    For seeding calls whose seed is a parameter, or an attribute or
    constant-key item reached from a parameter or a module global:
    ``default_rng(seed)``, ``default_rng(settings.seed)``,
    ``default_rng(opts["seed"])``, ``default_rng(CONFIG.seed)``. A root the
    function assigns itself is a local, which cannot be read before the call,
    and is skipped.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    fn = next((n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return {}
    a = fn.args
    params = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    assigned = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    found: dict[str, tuple[str, str, bool, tuple]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        dotted = ast.unparse(node.func)
        last = dotted.rsplit(".", 1)[-1]
        if not (last in _SEEDING_CALLS or (last == "seed" and "random" in dotted)):
            continue
        seed_arg = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg in ("seed", "x", "a")), None)
        if seed_arg is None:
            continue
        access = _seed_access_path(seed_arg)
        if access is None:
            continue
        root, path = access
        is_param = root in params
        if not is_param and root in assigned:
            continue
        expr = ast.unparse(seed_arg)
        found.setdefault(expr, (f"{dotted}({expr})", root, is_param, path))
    return found


_SEED_UNREADABLE = object()


def _read_seed(value: Any, path: tuple) -> Any:
    """Follow *path* from *value* WITHOUT running user code, or `_SEED_UNREADABLE`.

    Attributes through ``inspect.getattr_static``: a plain instance or class
    attribute is read, a property or other descriptor is not evaluated. Items
    only from a plain mapping. This runs on every call, so it may not call
    anything the user wrote.
    """
    for kind, key in path:
        if kind == "attr":
            try:
                value = inspect.getattr_static(value, key)
            except AttributeError:
                return _SEED_UNREADABLE
            if hasattr(type(value), "__get__") and not isinstance(
                    value, (types.FunctionType, types.BuiltinFunctionType)):
                return _SEED_UNREADABLE          # a property or descriptor
        elif isinstance(value, (dict, types.MappingProxyType)):
            value = value.get(key, _SEED_UNREADABLE)
            if value is _SEED_UNREADABLE:
                return value
        else:
            return _SEED_UNREADABLE
    return value


#: Source files already reported as edited-since-load, one notice per file.
_SOURCE_CHANGED_WARNED: set[str] = set()

#: One line per decorated call -- hit or miss, and why -- when `debug=True` /
#: `CASH_DEBUG=1` or `verbose=True` asks for it.
_calls_logger = logging.getLogger("cash.calls")

#: The stderr handler `_enable_cash_logging` installed, if it installed one.
_CASH_STDERR_HANDLER: logging.Handler | None = None


def _enable_cash_logging(level: int) -> None:
    """Make `cash` log records at *level* reach the user.

    Lowers the `cash` logger's level, and -- only when nothing anywhere would
    print a record, which is a script's default -- attaches one stderr handler.
    An application that configured logging keeps its own handlers and format;
    it just starts receiving cash's records. stderr, not stdout: stdout is
    often the program's output (a report, a pipe, a JSON response).
    """
    global _CASH_STDERR_HANDLER
    cash_logger = logging.getLogger("cash")
    if cash_logger.level == logging.NOTSET or cash_logger.level > level:
        cash_logger.setLevel(level)
    if _CASH_STDERR_HANDLER is None and not cash_logger.hasHandlers():
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        cash_logger.addHandler(handler)
        _CASH_STDERR_HANDLER = handler


# Why a call missed. The KIND is what the summary counts; the detail goes to
# the per-call debug line and to explain().
MISS_FIRST = "no entry yet"
MISS_ARGS = "new arguments"
MISS_CODE = "code or state changed"
MISS_DYNAMIC = "dynamic dependency changed"
MISS_FILE = "file changed"
MISS_TTL = "ttl expired"
MISS_NOT_STORED = "not stored last time"
MISS_GONE = "entry gone"
MISS_INCOMPLETE = "entry incomplete"
MISS_UNHASHABLE = "unhashable argument"
MISS_KEY_FAILED = "key could not be built"
MISS_MOCKED = "a helper is a mock"

#: `_store_refusal` was not handed a capture watch (the streaming path).
_NO_WATCH = object()

#: `file_dep_is_fresh` reason codes, as the miss reason and explain() say them.
_STALE_REASON_TEXT = {
    'unreadable': 'file missing',
    'size': 'size changed',
    'content': 'content changed',
    'mtime': 'mtime changed',
    'mtime-sampled': 'mtime changed (sampled file)',
    'ctime-sampled': 'the file was written (sampled file)',
    'appeared': 'a file the call looked for and did not find now exists',
    'remote-changed': 'remote object changed',
    'remote-unresolved': 'remote object could not be checked',
}

#: How many keys' store outcomes to remember. It explains the recent past;
#: a long-running service does not need the whole history to do that.
_STORE_OUTCOMES_MAX = 4096


def _describe_file_deps(deps: dict[str, Any] | None) -> dict[str, str]:
    """``{path: fingerprint}`` for the files an entry recorded, readably."""
    out: dict[str, str] = {}
    for path, rec in (deps or {}).items():
        if not isinstance(rec, dict):
            out[path] = str(rec)
            continue
        if rec.get("absent"):
            out[path] = "absent when read"
            continue
        parts = ["remote"] if rec.get("remote") else []
        if rec.get("size") is not None:
            parts.append(f"{rec['size']} bytes")
        if rec.get("hash"):
            parts.append(f"hash {str(rec['hash'])[:12]}")
        out[path] = ", ".join(parts) or "recorded"
    return out


def entry_id_of(cache_key: str) -> str:
    """The id `cash inspect` and `cash clear --entry` use for *cache_key*."""
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]

#: Values whose identity is code plus what it captures. A hasher registered for
#: one of these types covers every such value in the process, and the obvious
#: one -- by name -- gives every closure one factory makes the same identity.
#: `Cash._first_unhashable_arg` found only built-in-typed arguments.
_NO_SUSPECT = object()

_CODE_VALUE_TYPES = (types.FunctionType, types.MethodType, functools.partial)

#: The fix for an unhashable code value. It must NOT suggest
#: `register_hasher(function, ...)`: following that advice is how a second
#: closure got the first one's result.
_CODE_ARG_FIX = (
    "pass a module-level function in its place, and give the values it "
    "captures to the cached function as plain arguments, where they reach the "
    "key. Do not register a hasher for function: every closure one factory "
    "makes shares a name, so a hasher keyed on it hands one closure's result "
    "to another. See known-limitations.md, 'A closure or lambda passed as an "
    "argument'."
)


def _unhashable_arg_fix(value: Any, type_name: str) -> str:
    """The fix line for an argument of *type_name* that could not be hashed."""
    if isinstance(value, _CODE_VALUE_TYPES):
        return _CODE_ARG_FIX
    return (
        f"register a hasher with cash.register_hasher({type_name}, ...), or "
        f"pass the argument by a hashable value."
    )


#: Who wrote a value's ``_cash_lineage_hash``, in ``_cash_lineage_src``. Only the
#: notebook's statement layer keeps the tag current as the value changes, so
#: only its tag stands in for the value's content (see `_hash_arg_payload`).
LINEAGE_SRC_STATEMENT = "statement"
LINEAGE_SRC_DECORATOR = "decorator"
#: Written for a function decorated ``frozen=True``: the user's promise that the
#: result is not modified afterwards, trusted like the statement layer's tag and
#: audited now and then (`_audit_frozen`).
LINEAGE_SRC_FROZEN = "frozen"

#: A frozen object is re-hashed at its 8th use as an argument and every 64th
#: after that (every use under CASH_DEBUG), and compared with the first audit.
_FROZEN_AUDIT_FIRST = 8
_FROZEN_AUDIT_EVERY = 64

_COW_PANDAS: bool | None = None

#: The costliest argument of the key most recently hashed on this thread:
#: ``(label, seconds, type name, producer, pandas without copy-on-write)``.
#: A description, never the value: a reference here would keep a large
#: argument alive after its caller dropped it.
_ARG_COST = threading.local()


def _is_cow_pandas(value: Any) -> bool:
    """Is *value* a pandas DataFrame/Series under copy-on-write?

    Copy-on-write is the only mode in pandas 3 and opt-in before. Checked
    without importing pandas: a pandas object means it is already loaded.
    """
    global _COW_PANDAS
    t = type(value)
    if t.__name__ not in ("DataFrame", "Series") or not (t.__module__ or "").startswith("pandas"):
        return False
    if _COW_PANDAS is None:
        try:
            import pandas as pd
            major = int(pd.__version__.split(".", 1)[0])
            _COW_PANDAS = major >= 3 or pd.options.mode.copy_on_write is True
        except Exception:  # noqa: BLE001 - unknown pandas: no memo, hash every time
            _COW_PANDAS = False
    return _COW_PANDAS


def _warn_source_changed_since_load(fn: Callable) -> None:
    """Say, once per file, that a helper is keyed by its loaded code."""
    code = getattr(fn, "__code__", None)
    path = getattr(code, "co_filename", "") or ""
    if path in _SOURCE_CHANGED_WARNED:
        return
    _SOURCE_CHANGED_WARNED.add(path)
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "a function")
    try:
        from .diagnostics import warn_diagnostic
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
    except Exception:  # noqa: BLE001 - a notice must never break a call
        logger.debug("Could not emit the source-changed notice", exc_info=True)


def _builtin_hash_family(type_name: str, module: str) -> str | None:
    """Name the built-in content hasher that claims ``module.type_name``.

    Split out of `Cash._try_builtin_type_hash` so the same question can be
    asked of a bare TYPE, with no value in hand: ``register_hasher`` needs it
    to tell a user that the hasher they are registering would never run.

    Matched on module PREFIX, so a user's own subclass defined in their own
    module is deliberately not claimed -- registering a hasher for it has
    always worked and still does.
    """
    if module.startswith('pandas') and type_name in ('DataFrame', 'Series'):
        return 'pandas'
    if type_name == 'ndarray' and module.startswith('numpy'):
        return 'numpy'
    if module.startswith('polars'):
        return 'polars'
    if module.startswith('pyarrow'):
        return 'pyarrow'
    if module.startswith('modin'):
        return 'modin'
    if module.startswith('dask'):
        return 'dask'
    return None


#: Pydantic v2 compiles these onto every model class. They are derived from the
#: field declarations and their digest differs in every process, so folding them
#: made a pydantic spec un-cacheable across runs. `Cash._pydantic_field_parts`
#: folds the declarations they were standing in for.
_PYDANTIC_COMPILED = frozenset({
    "__pydantic_core_schema__",
    "__pydantic_serializer__",
    "__pydantic_validator__",
})


class Cash:
    """Smart caching framework for Python functions and Jupyter notebooks.

    Provides decorator-based caching with automatic dependency tracking,
    file dependency monitoring, and pluggable storage backends.

    By default (no ``backend`` or ``cache_dir`` specified), uses a
    TieredBackend with L1 in-memory + L2 file-based storage in a local
    ``.cash`` directory.  Pass ``backend=`` to override.

    Args:
        backend: A specific cache backend instance to use.
        cache_dir: Path to cache directory (creates TieredBackend automatically).
        backends: List of backends for cascading cache (L1/L2/L3).
        compress: Enable gzip compression for file-based caching.
        register_magic: Register IPython magic commands (default True).
        debug: Enable debug logging output.
        use_locking: Enable double-checked locking for thread-safe caching.
        config_path: Path to custom config TOML file.

    Example:

        from cash import Cash
        c = Cash()

        @c.cache
        def expensive(x):
            return x ** 2
    """

    graph: DependencyGraph
    functions: dict[str, Callable[..., Any]]
    data_sources: dict[str, DataSource]
    source_hashes: dict[str, str]
    debug: bool | None
    use_locking: bool
    config: CashConfig

    def __init__(
        self,
        backend: CacheBackend | None = None,
        cache_dir: str | None = None,
        backends: list[CacheBackend] | None = None,
        compress: bool | None = None,
        register_magic: bool | None = None,
        debug: bool | None = None,
        use_locking: bool = False,
        config_path: str | None = None,
        verbose: bool = False,
        **config_overrides: Any,
    ) -> None:
        # Map the explicit convenience kwargs (cache_dir, compress, debug)
        # into the overrides dict so the config layer treats them with the
        # same priority as any other constructor-supplied override (highest).
        for key, val in (("cache_dir", cache_dir), ("compress", compress), ("debug", debug)):
            if val is not None:
                config_overrides.setdefault(key, val)
        self.config = get_config(config_path=config_path, overrides=config_overrides or None)

        debug = self.config.debug

        # Store params for lazy backend construction. If an explicit
        # backend (or list of backends) was provided, that wins - those
        # are concrete objects, not config - and we skip the factory.
        self._backend: CacheBackend | None = None
        self._explicit_backends = backends  # remembered for repr / debugging only
        if backend is not None:
            self._backend = backend
        elif backends:
            if len(backends) > 1:
                self._backend = CascadingBackend(backends)
            else:
                self._backend = backends[0]

        self._backend_lock = threading.Lock()

        self.graph = DependencyGraph()
        self.functions: dict[str, Callable[..., Any]] = {} # Registry of cached functions
        self.data_sources: dict[str, DataSource] = {} # Registry of data sources
        self.source_hashes: dict[str, str] = {} # Current source hashes
        # Session-scoped memo: id(arg) -> (weakref, lineage_hash, content_hash).
        # Lets a repeated ``@cash.cache`` call with the SAME unmutated argument
        # skip re-hashing a possibly-huge input. See ``_hash_arg_payload`` for
        # the read-side validation (weakref identity + lineage). Bounded below.
        self._arg_hash_memo: dict[int, tuple] = {}
        # id(frame) -> (weakref, shallow copy, signature, content hash): the
        # pandas copy-on-write memo, see `_frame_memo_store`.
        self._frame_memo: dict[int, tuple] = {}
        # Functions decorated frozen=True, by key.
        self._frozen_funcs: set[str] = set()
        # id(ndarray) -> [weakref, producer, content hash or None]: numpy
        # results of frozen functions, which cannot carry a tag.
        self._frozen_arrays: dict[int, list] = {}
        # id(obj) -> [weakref, uses, audit baseline or None], see `_audit_frozen`.
        self._frozen_uses: dict[int, list] = {}
        # func_name -> (parameter, type, seconds, producer, pandas without
        # copy-on-write): the costliest argument to hash, for CACHE-NET-LOSS.
        self._arg_costs: dict[str, tuple] = {}
        # Running account of what caching cost vs what it saved, per function.
        # The decorator always caches by design -- this only ever informs.
        from cash.effectiveness import EffectivenessLedger
        self._effectiveness = EffectivenessLedger()
        if self.config.summary:
            # Registered per instance rather than once per process: two Cash
            # instances are two independent caches, and each should account for
            # itself. ``run_summary`` returns "" when nothing was called, so an
            # unused instance prints nothing. (``atexit`` is imported at module
            # level; a local import here would shadow it for the whole method,
            # including the ``atexit.register(self.shutdown)`` further down.)
            atexit.register(self._print_run_summary)
        self._analyzed = set() # Track which functions we've *surfaced* purity for
        # ONE lock for the one-time analysis, whatever function triggers it.
        #
        # The check-and-analyze below is not atomic, and the CACHE KEY depends
        # on what the analysis populates (helper source hashes, graph edges).
        # Concurrent first calls therefore resolved two different keys for one
        # call -- the threads that got there before the analysis finished, and
        # the one that did it -- so `use_locking=True` looked like it admitted
        # exactly two threads into the compute at every thread count. It was
        # not the lock: each key was single-flighted correctly, there were just
        # two of them, and the pre-analysis one is an entry no later run will
        # ever look up. Measured: warming the analysis in the main thread first
        # collapsed 6 threads to one key and one execution.
        #
        # RLock, not Lock: analysis walks the dependency graph and re-enters
        # this same guard for the callees it populates on the way.
        #
        # One lock rather than one per function, deliberately. Analysis of f
        # populates f's whole callee closure, so per-function locks could be
        # taken in two orders by two threads and deadlock. It is a one-time,
        # source-reading step measured in milliseconds; serialising unrelated
        # first calls behind it costs nothing worth a lock-ordering rule.
        self._analysis_lock = threading.RLock()
        # Track which functions have had their graph edges + purity report
        # populated (separate from _analyzed: a dependency can be populated to
        # complete a parent's state hash long before it is called directly and
        # surfaced). Keeps the cache key stable from the first call (finding #7).
        self._populated: set[str] = set()
        self._func_ttls: dict[str, int | None] = {}  # func_name -> declared ttl
        self._effective_ttl_cache: dict[str, int | None] = {}
        self._deref_writes: dict = {}  # code object -> frozenset of reassigned freevars
        self._func_key_cache: dict[int, str] = {}  # id(func) -> module-qualified key
        # id(func) -> decoration-pinned own-source identity. The
        # wrapper closure keeps *func* alive, so the id stays valid for the
        # wrapper's lifetime (same contract as _func_key_cache).
        self._own_pins: dict[int, str] = {}
        # Pins taken at decoration whose file has not yet been compared with
        # the loaded code; the first call does it once (see _pin_own_source).
        self._own_pins_unverified: set[int] = set()
        # code object -> global names its decorator expressions read
        self._decorator_names_cache: dict = {}
        # code object -> frozenset of free vars with capture-unsafe uses
        self._capture_use_cache: dict = {}
        # code object -> tuple of global names it reads (global folding)
        self._global_read_cache: dict = {}
        # code object -> names folded only provisionally (CAS-270). See
        # `_read_global_data_names`. A missing entry means "unknown", which
        # `_fold_read_globals` treats as "watch everything".
        self._provisional_global_cache: dict = {}
        # (code object, scope) -> names a call was OBSERVED to mutate. Learned
        # once, then those names stop being folded (see `_learn_mutating_captures`).
        self._mutating_globals: dict = {}
        # code object -> closure free vars folded only provisionally (CAS-270).
        self._provisional_capture_cache: dict = {}
        # Per-call scratch: {name: (pre-call hash, scope, owner_globals)} for
        # every provisional capture folded into the key being built. Cleared
        # once per `_resolve_cache_key`, because BOTH `_fold_closure` and
        # `_fold_read_globals` contribute and either clearing it would wipe the
        # other's entries.
        #
        # `owner_globals` is the mapping the pre-call hash was taken FROM, and
        # it is not always the decorated function's own. `_fold_read_globals`
        # also runs on behalf of module-bounded HELPERS, so a global read by a
        # helper in another module lands here under a bare name that does not
        # exist in `func.__globals__` at all. Re-reading it there found None,
        # hashed that, and reported every such global as mutated by the call --
        # a provider registry read by a client helper warned on every first
        # call. Carrying the owning mapping is what makes the after-hash look
        # at the same variable the before-hash did.
        self._pending_capture_watch: dict[str, tuple[str, str, Any]] = {}
        # func_name -> RNG modules that function was OBSERVED drawing from.
        # Learned on a miss; only these functions get a seed-epoch in their key.
        self._rng_drawing_funcs: dict[str, set[str]] = {}
        # (module_global, attribute) read pairs per code object; see
        # _read_module_attr_pairs.
        self._module_attr_cache: dict = {}
        # (first_param, self_attrs, uses_super) per code object; see
        # _analyze_method_self_deps.
        self._method_self_dep_cache: dict = {}
        # user class -> source hash. A class's source cannot change within a
        # running interpreter, so it is hashed once and reused; see
        # _user_class_source_hash / _instance_class_source_parts.
        self._user_class_src_cache: dict = {}
        # user class or function -> code-surface digest (bytecode-based, class-
        # aware); see _code_surface_hash. Keyed on the object itself, not
        # id(), so a redefinition (a new object) is a distinct memo entry.
        self._code_surface_cache: dict = {}
        # object -> tuple of (code object, globals dict) it carries. Static for
        # as long as that object exists (a redefinition makes a new one), so it
        # is safe to memo; the NAMES those code objects reference are resolved
        # fresh per call, because what a name is bound to can change.
        self._code_refs_cache: dict = {}
        # func_name -> (function the signature was read from, inspect.Signature
        # or None if introspection failed). Used to bind call arguments to a
        # canonical form so that logically-identical calls written differently
        # (positional vs keyword, omitted vs explicit default, kwargs in
        # different orders) share one cache key.
        #
        # The function is stored alongside so the memo can be invalidated when
        # the name is REBOUND to a new function object (a notebook cell re-run).
        # Keying by name alone pinned the first signature forever, so
        # `apply_defaults()` kept folding a default the callee no longer has
        #.
        self._signatures: dict[str, tuple[Callable | None, inspect.Signature | None]] = {}
        # function object -> digest of its parameter defaults, for defaults that
        # are immutable and therefore cannot drift between calls.
        # Weak so the memo dies with the function instead of pinning it (and so
        # a later function object can never inherit a dead one's entry by
        # id-reuse). Mutable defaults are deliberately absent: they must be
        # re-hashed per call to stay correct.
        self._defaults_pins: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # id(helper) -> (helper, __defaults__, __kwdefaults__, identity); see
        # `_hash_helper_identity`. Holding the helper keeps its id from being
        # recycled while the entry lives.
        self._helper_defaults_memo: dict[int, tuple[Any, Any, Any, str]] = {}
        # func_name -> {parameter: seeding call} for seeding calls fed by a
        # parameter; checked per call by `_warn_if_seed_is_none`.
        self._seed_params: dict[str, dict[str, tuple]] = {}
        # In-process async single-flight registry: cache_key -> (event_loop,
        # asyncio.Event). When use_locking is set, concurrent awaits of the
        # same key coalesce - one coroutine computes, the rest wait on the
        # event and then read the stored result.
        self._async_inflight: dict[str, tuple[Any, Any]] = {}
        self.debug = debug  # Debug mode flag
        self.use_locking = use_locking
        self.verbose = verbose
        # Asking for debug output has to produce some. The flag used to set
        # nothing but this attribute, and a script has no logging configured,
        # so `CASH_DEBUG=1` printed not one line (round 17, three testers).
        if debug or verbose:
            _enable_cash_logging(logging.DEBUG if debug else logging.INFO)

        # What a miss was, for the people asking "why did that recompute?".
        # All three are in-process memory only, and bounded: they explain,
        # they never decide anything.
        #: func_name -> the last cache key it looked up.
        self._last_key: dict[str, str] = {}
        #: cache_key -> what happened when it was last computed here.
        self._store_outcomes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        #: cache_key -> (kind, detail) for a lookup that just missed, taken
        #: by the `_log_decorator_call` that reports it.
        self._pending_miss: dict[str, tuple[str, str]] = {}

        # Decorator call log for notebook integration.
        # Each entry is a dict with: func_name, cache_hit (bool), execution_time,
        # args_hash, cache_key, timestamp.  The notebook statement processor
        # drains this list after executing each statement so it can include
        # decorator metrics in the badge.
        self._decorator_call_log: list[dict[str, Any]] = []
        self._decorator_call_log_lock = threading.Lock()
        # Custom type hasher registry: maps type -> (callable(value) -> str, source hash).
        # The source hash is embedded in the args_hash composition so that
        # changing a hasher's body invalidates dependent cache entries.
        # func_name -> the live stats dict of its wrapper. Populated by
        # ``_wrap_with_stats``; read by the end-of-run summary.
        self._function_stats: dict[str, dict[str, Any]] = {}
        self._type_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}
        # Same shape, but consulted BEFORE cash's own content hashers rather
        # than after them. Separate registry rather than a flag in the tuple
        # above so the hot path can skip the whole question with one empty
        # check -- overriding is rare, and every cached call pays for this.
        self._override_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}

        # Dedup keys for _warn_once: (category, func_name, arg_type_name).
        # Guarded by _decorator_call_log_lock (already exists for thread safety).
        self._warning_keys_seen: set[tuple[type[Warning], str, str]] = set()

        # Per-function rolling log of recent warning emissions, surfaced
        # via ``f.cache_info()['warnings']``. Each entry is
        # ``{'category': str, 'message': str, 'timestamp': float}``.
        # Capped at ``_func_warnings_max`` entries per function so a
        # noisy function can't grow this dict unboundedly. Guarded by
        # ``_decorator_call_log_lock``.
        self._func_warnings: dict[str, list[dict[str, Any]]] = {}
        self._func_warnings_max = 20

        # Registry of wrapped (stats) functions for clear_all() support.
        # Maps func_name -> stats_wrapper (the object returned to the user).
        self._wrapped_funcs: dict[str, Any] = {}

        # Per-function purity mode set at decoration time:
        # ``"warn"`` (default) | ``"silent"`` (assume_safe=True) |
        # ``"strict"`` (raises). Read on first call.
        self._purity_modes: dict[str, str] = {}
        # Functions the STATIC pass already reported on. The runtime effect
        # observer stays quiet for these: it would be a second warning about
        # the same function, and the user has already been told.
        self._purity_static_flagged: set[str] = set()
        # Functions whose arguments cost too much to re-hash for the
        # mutation check. Checked once, then left alone.
        self._mutation_check_too_costly: set[str] = set()
        # Per-function purity report cache. Populated on first call.
        # Helper source hashes from this report fold into the cache
        # key state hash so cross-process helper edits invalidate.
        self._purity_reports: dict[str, PurityReport] = {}

        # Declared plain-callable dependencies (``depends_on=[proxy_fn]`` where
        # proxy_fn is NOT a decorated cached function). Snapshot source hash at
        # registration + a ``(module, attr_chain)`` path for live re-resolution,
        # so editing the dep on disk + reload invalidates the parent key.
        self._declared_dep_snapshots: dict[str, str] = {}
        self._declared_dep_paths: dict[str, tuple[str, tuple[str, ...]]] = {}

        # Deep seam over the registries above: folds source/dependency/
        # helper state into the cache key's ``state_hash`` segment. Borrows
        # the registry dicts by reference so later registrations are seen.
        self._state_hasher = DependencyStateHasher(
            functions=self.functions,
            data_sources=self.data_sources,
            source_hashes=self.source_hashes,
            purity_reports=self._purity_reports,
            graph=self.graph,
            helper_resolver=SysModulesHelperResolver(self._hash_helper_identity),
            declared_dep_snapshots=self._declared_dep_snapshots,
            declared_dep_resolver=self._resolve_declared_dep_hash,
        )

        atexit.register(self.shutdown)

        # register_magic=None (default) auto-detects: only register when an
        # active IPython session exists.  True forces registration; False skips.
        if register_magic is True or (register_magic is None and get_ipython() is not None):
            self.register_magic()

    @property
    def backend(self) -> CacheBackend:
        """Lazily build the cache backend from ``self.config`` on first access.

        This avoids filesystem I/O, thread creation, and directory
        scanning at ``Cash()`` construction time. The heavy lifting
        happens only when the cache is actually used.
        """
        if self._backend is not None:
            return self._backend
        with self._backend_lock:
            if self._backend is not None:
                return self._backend
            self._backend = build_backend_from_config(self.config)
            return self._backend

    @backend.setter
    def backend(self, value: CacheBackend) -> None:
        """Allow direct assignment (e.g. ``c.backend = MyBackend()``)."""
        self._backend = value

    def __repr__(self) -> str:
        backend_name = type(self._backend).__name__ if self._backend is not None else '<deferred>'
        n_funcs = len(self.functions)
        return f"Cash(backend={backend_name}, functions={n_funcs}, debug={self.debug})"

    @staticmethod
    def _get_func_key(func: Callable) -> str:
        """Return a module-qualified key for a function.

        Uses ``func.__module__ + '.' + func.__qualname__`` to avoid collisions
        when different modules define functions with the same ``__qualname__``
        (e.g. a notebook's ``dep()`` vs a library module's ``dep()``).

        ``__main__`` is resolved to the name the module would have when
        imported — see `_main_module_name`.

        Opaque callables such as ``functools.partial`` lack both ``__qualname__``
        and ``__name__``; fall back to ``repr`` so keying them never crashes.
        """
        module = getattr(func, '__module__', None) or '__unknown__'
        if module == '__main__':
            module = resolve_main_module(func)
        qualname = (
            getattr(func, '__qualname__', None)
            or getattr(func, '__name__', None)
            or repr(func)
        )
        return f"{module}.{qualname}"

    def _analyze_method_self_deps(self, func: Callable) -> tuple[str | None, tuple[str, ...], bool]:
        """Attributes a method reads on its first parameter, and whether it calls super().

        A ``@cash.cache`` method reaching class-level code -- ``self.helper()``,
        ``self.RATE``, ``super().m()`` -- had none of that in its key, because at
        decoration time the class does not exist yet and the analyzer sees only
        an attribute access on a parameter. Recorded here (source-derived,
        cached per code object) and resolved against the real class at call time
        by :meth:`_fold_method_class_deps`.

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
            src = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(src)
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError,):
            if code is not None and len(self._method_self_dep_cache) < 4096:
                self._method_self_dep_cache[code] = result
            return result
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_def = node
                break
        if func_def is None or not func_def.args.args:
            if code is not None and len(self._method_self_dep_cache) < 4096:
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
                elif (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                      and v.func.id == "type" and len(v.args) == 1
                      and isinstance(v.args[0], ast.Name) and v.args[0].id == self_name):
                    attrs.add(node.attr)
            # ``super()`` / ``super(...)`` anywhere.
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id == "super":
                    uses_super = True
                elif (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call)
                      and isinstance(f.value.func, ast.Name) and f.value.func.id == "super"):
                    uses_super = True
        result = (self_name, tuple(sorted(attrs)), uses_super)
        if code is not None and len(self._method_self_dep_cache) < 4096:
            self._method_self_dep_cache[code] = result
        return result

    def _fold_method_class_deps(self, func: Callable, args: tuple, state_hash: str) -> str:
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
        # a mutually-recursive method pair from looping. Bounded for safety.
        seen: set[str] = set()
        worklist: list[str] = list(attrs)
        while worklist and len(seen) < 512:
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
                        parts.append(f"p:{attr}:{self._hash_callable_source(getter)}")
                    except (OSError, TypeError, ValueError):
                        pass
                    _, sub_attrs, _ = self._analyze_method_self_deps(getter)
                    worklist.extend(a for a in sub_attrs if a not in seen)
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if inspect.isfunction(member) or inspect.ismethod(member):
                try:
                    parts.append(f"m:{attr}:{self._hash_callable_source(member)}")
                except (OSError, TypeError, ValueError):
                    continue
                # Recurse into what this method itself reaches through self.
                _, sub_attrs, _ = self._analyze_method_self_deps(member)
                worklist.extend(a for a in sub_attrs if a not in seen)
            elif not isinstance(member, (types.ModuleType, type)) and not callable(member):
                # A class-level DATA attribute (a constant). Fold its value.
                try:
                    parts.append(f"c:{attr}:{self._hash_arg_payload((member,), {})}")
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                    continue
        if uses_super:
            for base in owner_class.__mro__[1:]:
                if base is object:
                    continue
                try:
                    parts.append(f"b:{base.__qualname__}:{self._hash_callable_source(base)}")
                except (OSError, TypeError, ValueError):
                    continue
        if not parts:
            return state_hash
        payload = ":".join(sorted(parts))
        return hashlib.sha256(f"{state_hash}:selfdeps:{payload}".encode('utf-8')).hexdigest()

    #: Types already reported as unhashable, so the advisory stays once-per-type
    #: per session rather than once per call.
    _WARNED_UNHASHABLE: set = set()

    @staticmethod
    def _carrier_name(carrier: Any) -> str:
        """A stable, address-free name for a code carrier.

        ``__qualname__`` for anything that has one -- a class, a function.
        Otherwise the carrier's TYPE, deliberately not ``repr()``: a
        ``functools.partial`` reprs as ``functools.partial(<function f at
        0x...>, 3)``, and that address is unique per object, so a
        ``repr()``-derived key would make ``_WARNED_UNHASHABLE`` dedup
        nothing (one warning per partial ever constructed, plus a global set
        that grows without bound) and would make a folded part label differ
        between two processes holding equal arguments.
        """
        name = getattr(carrier, "__qualname__", None) or getattr(carrier, "__name__", None)
        if isinstance(name, str) and name:
            return name
        t = type(carrier)
        return getattr(t, "__qualname__", None) or getattr(t, "__name__", None) or "?"

    @staticmethod
    def _is_user_code_carrier(carrier: Any) -> bool:
        """``_is_user_code_object`` for the ADVISORY rather than for hashing.

        ``_is_user_code_object`` answers "could not confirm reachability ->
        treat as user code". That is the safe direction when deciding whether
        to HASH something and the wrong one when deciding whether to WARN about
        it: an object with no ``__qualname__`` of its own -- a
        ``functools.partial``, a ``weakref.ref`` -- can never be confirmed, so
        every single one was reported as un-hashable user code.

        Judge such an object by what it WRAPS (``.func``, the same attribute
        ``_class_surface_parts`` already follows for ``singledispatchmethod``
        and ``cached_property``), else by its TYPE. Measured:
        ``functools.partial(json.dumps)`` and ``weakref.ref(x)`` stop warning,
        while ``functools.partial(<a user function>)`` still warns -- and it
        must, because the wrapped body genuinely is absent from the key.
        """
        if getattr(carrier, "__qualname__", None) or getattr(carrier, "__name__", None):
            return Cash._is_user_code_object(carrier)
        wrapped = getattr(carrier, "func", None)
        if wrapped is not None:
            return Cash._is_user_code_object(wrapped)
        return Cash._is_user_code_object(type(carrier))

    def _warn_unhashable_code_once(self, carrier: Any) -> None:
        """Tell the user once that a reached type's code is NOT in the key.

        "reached", not "passed": the code channel keys off the BOUND arguments,
        so a carrier arriving as a parameter default the caller never typed
        gets here too.
        """
        name = self._carrier_name(carrier)
        if name in Cash._WARNED_UNHASHABLE:
            return
        Cash._WARNED_UNHASHABLE.add(name)
        what = (
            f"{name} reached a cached call as an argument or a parameter "
            f"default, but its code could not be hashed, so editing it will NOT "
            f"invalidate the cache."
        )
        fix = (
            f"name it with @cash.cache(depends_on=[...]) if the result depends "
            f"on its implementation, or record that it does not with "
            f"cash.mark_opaque({name})."
        )
        # The log carries the same rendered text as the warning, code and all,
        # so a log-only reader is not the one person without a handle to search.
        log_diagnostic(logger, "KEY-OPAQUE-CALLABLE", what, fix)
        warn_diagnostic(
            CashImpurityWarning, "KEY-OPAQUE-CALLABLE", what, fix,
        )

    def _iter_code_carriers(self, value: Any, _depth: int = 0, _seen: set | None = None):
        """Yield objects in *value* that carry user code.

        Depth-bounded at 8, matching ``_stabilize_for_global_hash``. ``_seen``
        guards self-referential containers, and doubles as a once-per-argument
        dedup for the classes yielded on behalf of instances: a list of 50k
        objects of one class must evaluate the user-code gate once, not 50k
        times. Dedup by identity is safe in both roles -- a class already
        yielded does not need yielding again, and the fold takes a ``set`` of
        the parts anyway.
        """
        if _depth > 8:
            return
        # Primitives carry no user code, and in a large argument they ARE the
        # argument. Returning before ``_seen`` is touched keeps a list of a
        # million numbers allocation-free; otherwise the id-set below would grow
        # to the container's length on every cached call. Mirrors
        # ``_iter_contained``'s first line. See `_CODELESS_PRIMS` for why this
        # is an exact-type test against a tuple rather than an isinstance.
        if type(value) in _CODELESS_PRIMS:
            return
        if _seen is None:
            _seen = set()
        # ``_seen`` is recorded on the paths that need it -- containers, for
        # cycle safety, and yielded carriers, to yield each once -- and NOT for
        # a leaf instance. A leaf cannot contain itself, and its class is
        # deduped by `_instance_class_carrier` anyway, so an entry per element
        # bought nothing and cost a set insert per element: measured 2000
        # ns/element at 200k against 470 ns/element at 10k, i.e. the set itself
        # had become the superlinear term.
        if isinstance(value, type):
            if id(value) not in _seen:
                _seen.add(id(value))
                yield value
            return
        if callable(value):
            # Distinguish a callable INSTANCE (whose class defines ``__call__``
            # in Python) from a function/method/C-callable. An instance has no
            # ``__code__`` of its own -- its code lives on its class -- so
            # yielding the instance would fold NOTHING, while the identical
            # object WITHOUT ``__call__`` takes the instance branch below and
            # folds its class. Measured before this branch existed: adding
            # ``__call__`` to a class silently removed that class's code from
            # the key, and the instance then also tripped the unhashable
            # advisory. ``_is_opaque`` returns the same verdict for a class as
            # for one of its instances, so routing the class here rather than
            # the instance leaves opacity unchanged.
            call = getattr(type(value), "__call__", None)
            if getattr(call, "__code__", None) is not None:
                cls = self._instance_class_carrier(value, _seen)
                if cls is not None:
                    yield cls
                return
            if id(value) not in _seen:
                _seen.add(id(value))
                yield value
            return
        # The primitive test is repeated INLINE in each loop below rather than
        # left to the recursive call's own first line. It is the same test and
        # the same result, but it skips building a generator frame per element,
        # and a container of primitives is the overwhelmingly common argument:
        # measured 25.8ms -> 7.2ms for a 200k-int list.
        if isinstance(value, dict):
            if id(value) in _seen:
                return
            _seen.add(id(value))
            # A dict SUBCLASS is both a container to walk and a user object
            # whose class carries code. Handling only the first is the same
            # "invisible because of what it inherits from" hole that the
            # exact-type test above closes for str/int subclasses.
            if type(value) is not dict:
                cls = self._instance_class_carrier(value, _seen)
                if cls is not None:
                    yield cls
            for k, v in value.items():
                if type(k) not in _CODELESS_PRIMS:
                    yield from self._iter_code_carriers(k, _depth + 1, _seen)
                if type(v) not in _CODELESS_PRIMS:
                    yield from self._iter_code_carriers(v, _depth + 1, _seen)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if id(value) in _seen:
                return
            _seen.add(id(value))
            if type(value) not in _BUILTIN_CONTAINERS:
                cls = self._instance_class_carrier(value, _seen)
                if cls is not None:
                    yield cls
            for v in value:
                if type(v) not in _CODELESS_PRIMS:
                    yield from self._iter_code_carriers(v, _depth + 1, _seen)
        else:
            # An instance contributes its class's code. Deliberately NOT gated
            # on ``hasattr(value, "__dict__")``: a class using ``__slots__``
            # gives its instances no ``__dict__``, and that has nothing to do
            # with whether the user edits the class. Same family as the two
            # holes above -- a user class invisible because of how it is
            # declared rather than because of what it is.
            cls = self._instance_class_carrier(value, _seen)
            if cls is not None:
                yield cls

    def _instance_class_carrier(self, value: Any, _seen: set) -> type | None:
        """``type(value)`` if it is user code and not already seen this walk.

        Split out because three branches need it, and because the dedup is the
        difference between one user-code gate evaluation per ARGUMENT and one
        per ELEMENT -- ``_is_user_code_object`` is a ``sys.modules`` lookup plus
        a ``__qualname__`` walk, and a list of 50k instances of one class was
        paying it 50k times (measured: 50000 calls -> 1).

        A plain function rather than a generator on purpose: the callers are in
        the per-element path, and `yield from` on a fresh generator costs more
        per element than the call plus the `is not None` test it replaces.
        """
        cls = type(value)
        if id(cls) in _seen:
            return None
        _seen.add(id(cls))
        return cls if self._is_user_code_object(cls) else None

    def _fold_code_args(self, args: tuple, kwargs: dict, state_hash: str,
                        warn: bool = True, func_name: str = "?") -> str:
        """Fold user code reached through the arguments into the key.

        ``args_hash`` is a digest of the PICKLED arguments, and pickle
        serializes a class or function by reference -- module plus qualname,
        never its code. So editing a passed class produced an identical key and
        a hit, and cash handed back the stale class object it had cached.

        Folded into ``state_hash`` rather than ``args_hash`` because this is
        code, and ``state_hash`` is already where code lives: function source,
        ``depends_on``, transitive helpers, module globals read.

        ``warn=False`` for ``_explain_call``, whose contract is that inspecting
        an explanation emits no warnings.
        """
        parts: list[str] = []
        seen_carriers: set[int] = set()
        try:
            for value in (*args, *kwargs.values()):
                for carrier in self._iter_code_carriers(value):
                    # Dedup ACROSS arguments too, not just within one walk:
                    # `f(a, b, c)` with three instances of one class reaches
                    # `_is_opaque` + `_code_surface_hash` once instead of three
                    # times. Safe by identity because every carrier is
                    # reachable from `args`/`kwargs` for this whole loop, so no
                    # id can be recycled underneath us.
                    if id(carrier) in seen_carriers:
                        continue
                    seen_carriers.add(id(carrier))
                    if self._is_opaque(carrier):
                        continue
                    digest = self._code_surface_hash(carrier)
                    if digest is not None:
                        parts.append(f"{self._carrier_name(carrier)}:{digest}")
                        # Its CODE is in the key; the globals that code reads
                        # were not (CAS-113). A callback reading a module
                        # constant served the old result after the constant
                        # changed, while the same read one call level deeper,
                        # or in the cached function itself, invalidated.
                        if self._is_user_code_carrier(carrier):
                            parts.extend(self._carrier_read_global_parts(carrier, func_name))
                    elif warn and self._is_user_code_carrier(carrier):
                        # User code we could not hash: a C-extension type, an
                        # exotic descriptor, a ``functools.partial`` (whose
                        # wrapped function pickles by reference like any
                        # other). We fall back to today's key, which means an
                        # edit will NOT invalidate -- so say so once. This is
                        # the residue where cash genuinely cannot determine the
                        # answer, and silence is the danger.
                        self._warn_unhashable_code_once(carrier)
        except Exception as e:  # noqa: BLE001 - never break a call
            logger.debug("[CORE] code-arg fold failed: %s", e)
            return state_hash
        if not parts:
            return state_hash
        payload = ":".join(sorted(set(parts)))
        return hashlib.sha256(f"{state_hash}:codeargs:{payload}".encode('utf-8')).hexdigest()

    def _carrier_read_global_parts(self, carrier: Any, func_name: str) -> list[str]:
        """Key parts for the module data a code carrier's functions read.

        The same channel the cached function's own globals go through
        (`_read_global_data_names` + `_safe_global_hash`, plus the
        ``module.ATTR`` fold), applied to code that arrived as an ARGUMENT: a
        function, a bound method's function, or a class's own methods -- which
        is how a callable instance's ``__call__`` is reached.
        """
        from .source_norm import _class_functions

        if isinstance(carrier, type):
            functions = _class_functions(carrier)
        else:
            fn = getattr(carrier, "__func__", carrier)
            functions = [fn] if isinstance(fn, types.FunctionType) else []
        parts: list[str] = []
        for fn in functions:
            g = getattr(fn, "__globals__", None)
            if not isinstance(g, dict):
                continue
            owner = getattr(fn, "__qualname__", "?")
            try:
                for name in self._read_global_data_names(fn):
                    if name not in g:
                        continue
                    value = g[name]
                    if isinstance(value, (types.ModuleType, type)):
                        continue
                    if callable(value) and not isinstance(value, (dict, list, tuple, set)):
                        continue
                    h = self._safe_global_hash(value, func_name, f"{owner}.{name}")
                    if h is not None:
                        parts.append(f"argglobal:{owner}.{name}:{h}")
                for label, h in self._module_attr_parts(fn, func_name, g):
                    parts.append(f"argglobal:{owner}:{label}:{h}")
            except Exception as e:  # noqa: BLE001 - never break a call
                logger.debug("[CORE] callback-globals fold failed for %s: %s", owner, e)
        return parts

    @staticmethod
    def _hash_callable_source(fn: Callable) -> str:
        """Return a stable hex digest representing *fn*'s body.

        Used by `register_hasher` to embed the hasher's source
        identity in the cache key, so that changing a hasher's body
        invalidates dependent cache entries even when the hasher's
        output coincidentally matches the old one. Also the
        ``hash_callable`` injected into ``SysModulesHelperResolver``,
        which makes it the live per-call identity of every transitive
        HELPER -- so what this returns decides whether editing a helper
        recomputes its callers.

        Resolution order:

        1. ``inspect.getsource(fn)`` - primary, reduced via
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
        3. ``type(fn).__qualname__`` - last resort. Doesn't differentiate
           instances of the same class; the user gets stability within
           a process but coarse cross-process behavior.
        """
        memo_owner: Any = getattr(fn, "__code__", None)
        if memo_owner is None and isinstance(fn, type):
            memo_owner = fn
        memo_key = id(memo_owner) if memo_owner is not None else None
        if memo_key is not None:
            entry = _SOURCE_HASH_MEMO.get(memo_key)
            # ``is``, not ``==``: confirms this is the SAME object and not a
            # recycled id, and sidesteps CodeType's by-value equality.
            if entry is not None and entry[0] is memo_owner:
                return entry[1]

        # The source on disk may no longer be the code that is running: a file
        # edited after this process imported it (new files land, the restart
        # comes later) gives the NEW text for the OLD code object, and an entry
        # keyed by the new text but computed by the old code was served to the
        # restarted process (CAS-110). Key such a helper by what actually runs.
        # One os.stat in the normal case; see `loaded_code_matches_disk`.
        if not loaded_code_matches_disk(fn):
            digest = (loaded_class_identity(fn) if isinstance(fn, type)
                      else bytecode_identity(fn))
            if digest is not None:
                _warn_source_changed_since_load(fn)
                if memo_key is not None and len(_SOURCE_HASH_MEMO) < _SOURCE_HASH_MEMO_MAX:
                    _SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
                return digest

        try:
            # A `functools.wraps` wrapper is keyed by its OWN code. Plain
            # `getsource` unwraps, so it returned the WRAPPED function's text --
            # memoized under the wrapper's code object, which every function
            # that decorator wraps shares: the second helper wrapped by it was
            # silently keyed by the first one's source (round 18). The wrapped
            # function is followed as a helper of its own.
            if isinstance(fn, types.FunctionType) and hasattr(fn, "__wrapped__"):
                src = inspect.getsource(fn.__code__)
            else:
                src = inspect.getsource(fn)
            digest = source_identity_digest(src)
            if memo_key is not None and len(_SOURCE_HASH_MEMO) < _SOURCE_HASH_MEMO_MAX:
                _SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
            return digest
        except SOURCE_RETRIEVAL_ERRORS:
            pass
        digest = bytecode_identity(fn)
        if digest is not None:
            return digest
        return hashlib.sha256(type(fn).__qualname__.encode("utf-8")).hexdigest()


    @overload
    def cache(self, func: Callable[P, T]) -> Callable[P, T]: ...

    @overload
    def cache(
        self,
        func: None = None,
        *,
        depends_on: list[Callable[..., Any] | DataSource] | None = ...,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = ...,
        file_depends_on: str | list[str] | None = ...,
        ttl: int | None = ...,
        cache_if: Callable[[Any], bool] | None = ...,
        chunk_max_items: int = ...,
        chunk_max_bytes: int = ...,
        strict: bool = ...,
        assume_safe: bool = ...,
        allow_random: bool = ...,
        frozen: bool = ...,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

    def cache(
        self,
        func: Callable[P, T] | None = None,
        *,
        depends_on: list[Callable[..., Any] | DataSource] | None = None,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = None,
        file_depends_on: str | list[str] | None = None,
        ttl: int | None = None,
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
        strict: bool = False,
        assume_safe: bool = False,
        allow_random: bool = False,
        frozen: bool = False,
    ) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
        """Decorator to cache a function's return value.

        Can be used with or without arguments::

            @c.cache
            def f(x): ...

            @c.cache(ttl=3600)
            def g(x): ...

            @c.cache(file_depends_on="data.csv")
            def load_data():
                return pd.read_csv("data.csv")

        Args:
            func: The function to cache (set automatically when used without parens).
            depends_on: Static dependencies (functions or DataSources) to include
                in the cache key.
            dynamic_depends_on: Callable(s) that receive the same args as the
                decorated function and return DataSource(s) for cache key.
            file_depends_on: File path(s) to track as dependencies. The cache
                is automatically invalidated when any tracked file changes.
                Shorthand for ``depends_on=[FileDataSource("path")]``.
            ttl: Time-to-live in seconds. ``None`` means never expires.
            cache_if: Optional predicate ``callable(result) -> bool``. When
                provided, called with the function's return value after
                computation. If it returns a falsy value, the result is
                NOT stored in the cache (but is still returned to the
                caller). Useful for skipping the caching of negative
                results, e.g. ``cache_if=lambda r: r is not None``.
                The predicate is only invoked when the function returns
                normally; an exception from the function is re-raised
                without consulting the predicate. If the predicate itself
                raises, a one-shot `CashCacheIneffectiveWarning`
                fires and the result is treated as not-cacheable (the
                user's call still returns the result).
                If the cache key cannot be built at all (unhashable
                argument with no registered hasher, key-generation
                error), the predicate is not consulted - nothing is
                cached on that fallback path either.
            chunk_max_items: When the decorated function returns an
                iterator, close the current chunk after this many
                items. Default ``1_000_000``. A chunk closes when
                either ``chunk_max_items`` or ``chunk_max_bytes`` is
                reached (whichever comes first). For iterators below
                both thresholds, the entire result lands in a single
                chunk and storage is indistinguishable from a list.
            chunk_max_bytes: When the decorated function returns an
                iterator, close the current chunk after this many
                bytes (estimated via ``estimate_object_size``).
                Default ``1_000_000_000`` (1 GB). See
                ``chunk_max_items`` for the joint behavior.
            strict: When ``True``, raise `CashImpureFunctionError`
                on first call if the analyzer finds any purity issues
                (known-impure calls, scope mutations, explicit
                dynamism, or discarded calls to non-known-pure
                callees). Also promotes the analyzer's optimistic
                opaque-leaf treatment: opaque callees become issues.
                Use in CI to fail builds that introduce caching of
                side-effecting code. Mutually exclusive with
                ``assume_safe``.
            assume_safe: When ``True``, suppress the
                `CashImpurityWarning` even when the analyzer
                finds issues. Use when you've audited the function
                and know caching is correct (the side effect is
                idempotent, the dynamism is bounded, etc.). The
                analyzer still runs because it captures helper
                source hashes for cache invalidation. Mutually
                exclusive with ``strict``.
            allow_random: When ``True``, suppress the one-shot
                `CashRandomnessWarning` raised at decoration time
                if the function's source draws from an unseeded RNG
                (``np.random.randn()``, ``random.random()``,
                ``np.random.default_rng()`` with no seed, ...).
                The decorator-path counterpart of the notebook's
                ``# @cash:allow-random``; that comment is also
                honoured when it appears in the decorated
                function's own source. Suppresses only the
                *warning* - it does not change whether the result
                is cached, and the first call's value is still
                frozen and replayed. Seeding the RNG silences the
                warning on its own, because a seeded draw is
                reproducible.
            frozen: When ``True``, declare that the result is not
                modified after it is returned. A cached function
                receiving it as an argument then keys it by this
                call's identity instead of hashing its contents: no
                hash per call, the same key in every process, and it
                works for objects that cannot be pickled. A numpy
                result is returned read-only. Other objects are
                audited -- re-hashed at an occasional use, every use
                under ``CASH_DEBUG`` -- and a change warns
                KEY-FROZEN-MUTATED and falls back to content hashing
                for that object.

        Returns:
            The decorated function with caching behavior.

        See Also:
            [Caching class methods](../tutorials/feature-guides/caching-class-methods.md)
            for the recipe for caching methods on stateful objects
            (databases, file handles, connections) via
            [`register_hasher`][cash.Cash.register_hasher].
        """
        if strict and assume_safe:
            raise ValueError(
                "@cash.cache: strict=True and assume_safe=True are mutually "
                "exclusive. strict raises on purity issues; assume_safe silences "
                "them. Pick one."
            )

        if func is None:
            return lambda f: self.cache(
                f, depends_on=depends_on, dynamic_depends_on=dynamic_depends_on,
                file_depends_on=file_depends_on, ttl=ttl, cache_if=cache_if,
                chunk_max_items=chunk_max_items, chunk_max_bytes=chunk_max_bytes,
                strict=strict, assume_safe=assume_safe, allow_random=allow_random,
                frozen=frozen,
            )

        func_name = self._register_func(func, depends_on, file_depends_on)
        self._pin_own_source(func, self.source_hashes[func_name])
        if frozen:
            self._frozen_funcs.add(func_name)
        else:
            self._frozen_funcs.discard(func_name)
        self._purity_modes[func_name] =("strict" if strict else "silent" if assume_safe else "warn")
        # Record the declared TTL so a downstream that depends on this function
        # can inherit it (effective TTL = min over the dependency closure).
        self._func_ttls[func_name] = ttl
        self._effective_ttl_cache.clear()

        # Async generators are not cached; warn once and return unwrapped.
        if inspect.isasyncgenfunction(func):
            self._warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: async generators are not cached "
                f"in this release, so the function was returned unwrapped.",
                code="CACHE-ASYNC-GENERATOR",
                fix="move the work into a plain async def that returns a list "
                    "and cache that, or leave the generator undecorated and "
                    "cache the expensive step inside it.",
            )
            return func

        # Unseeded-randomness check. Deliberately here and not in the
        # wrapper: it is a pure function of the source, so it runs ONCE per
        # decorated function and adds nothing to the per-call path. Placed after
        # the async-generator early return because that path is not cached at
        # all, and the hazard being warned about is a frozen cached value.
        self._warn_unseeded_randomness(func, func_name, allow_random)

        if inspect.iscoroutinefunction(func):
            wrapper = self._make_async_wrapper(
                func, func_name, dynamic_depends_on, ttl, cache_if,
                chunk_max_items, chunk_max_bytes,
            )
        else:
            wrapper = self._make_wrapper(
                func, func_name, dynamic_depends_on, ttl, cache_if,
                chunk_max_items, chunk_max_bytes,
            )
        return self._wrap_with_stats(
            func, func_name, wrapper,
            dynamic_depends_on=dynamic_depends_on, ttl=ttl,
            allow_random=allow_random,
        )

    def _register_func(
        self,
        func: Callable,
        depends_on: list[Callable[..., Any] | DataSource] | None,
        file_depends_on: str | list[str] | None,
    ) -> str:
        """Register a function in the cache graph and return its key."""
        func_name = self._get_func_key(func)
        self.functions[func_name] = func
        new_hash = CodeAnalyzer.get_source_hash(func)
        old_hash = self.source_hashes.get(func_name)
        if old_hash and old_hash != new_hash:
            self._analyzed.discard(func_name)
            self._populated.discard(func_name)
        self.source_hashes[func_name] = new_hash
        self.graph.add_node(func_name)
        self._register_static_dependencies(func_name, depends_on)
        if file_depends_on:
            from .data_source import FileDataSource
            file_paths = [file_depends_on] if isinstance(file_depends_on, str) else file_depends_on
            file_deps = [FileDataSource(p) for p in file_paths]
            self._register_static_dependencies(func_name, file_deps)
        return func_name

    def _pin_own_source(self, func: Callable, source_hash: str | None = None) -> str:
        """Identity of *func* itself, pinned per function object.

        The state hash's root component must describe the function the
        wrapper EXECUTES, not the live ``source_hashes[qualname]`` registry
        slot — a redefinition (notebook cell re-run) or a second lambda
        sharing the ``<lambda>`` qualname overwrites that slot, letting a
        stale wrapper store its results under the new function's identity.

        For named functions the pin equals the registration-time source
        hash (byte-identical keys for the normal single-registration case,
        so persisted entries keep hitting). Lambdas additionally fold the
        code fingerprint: two lambdas defined on the SAME source line share
        their source text, and only ``co_code``/consts tell them apart.

        Taken when the decorator runs (*source_hash* is the hash registration
        just computed), because that is when the text on disk is the text the
        import compiled. It used to be taken at the FIRST CALL, and a deploy
        that landed new files between import and that call keyed the old
        body's result by the new body's text; every restarted process then
        served it as an ordinary hit (round 18, two testers independently).
        The first call still compares the loaded code with the file once, to
        say so. A pin not taken at decoration (the table is full) falls back to
        the same loaded-vs-disk check helpers get.
        """
        key = id(func)
        pin = self._own_pins.get(key)
        if pin is not None:
            if self._own_pins_unverified and key in self._own_pins_unverified:
                self._own_pins_unverified.discard(key)
                if not loaded_code_matches_disk(func):
                    _warn_source_changed_since_load(func)
            return pin
        at_decoration = source_hash is not None
        if source_hash is None:
            if loaded_code_matches_disk(func):
                source_hash = CodeAnalyzer.get_source_hash(func)
            else:
                source_hash = bytecode_identity(func) or CodeAnalyzer.get_source_hash(func)
                _warn_source_changed_since_load(func)
        pin = source_hash
        if getattr(func, '__name__', '') == '<lambda>':
            code = getattr(func, '__code__', None)
            if code is not None:
                # Primitive consts only: nested code objects repr with
                # memory addresses, which would destabilise the key.
                consts = tuple(
                    c for c in code.co_consts
                    if isinstance(c, (bool, int, float, complex, str, bytes, type(None)))
                )
                pin = hashlib.sha256(
                    f"{pin}:{code.co_code.hex()}:{consts!r}".encode('utf-8')
                ).hexdigest()
        if len(self._own_pins) < 4096:
            self._own_pins[key] = pin
            if at_decoration:
                self._own_pins_unverified.add(key)
        return pin

    def _fold_rng_epoch(self, func_name: str, state_hash: str) -> str:
        """Fold the current seed epoch into the key, for RNG-drawing functions.

        A function that draws from the global stream has an input the key never
        saw. Change ``np.random.seed(12345)`` to ``seed(999)``, re-run, and the
        model trained under the old seed came straight back, silently, with a
        green badge -- on the exact idiom ``cash.help()`` rule 4 recommends.

        Deliberately narrow on three axes:

        * Only functions OBSERVED to draw (``_rng_drawing_funcs``), so every
          other key is byte-identical to before.
        * Only the *epoch*, never the raw RNG state -- the state advances on
          every draw, so keying on it would miss forever.
        * Empty when the module is unseeded, so an unseeded sample keeps being
          replayed. That is the freeze contract, and it is what makes caching an
          expensive unseeded draw still worth it.

        The verdict is learned on a miss, so the call that first reveals the
        draw has already been stored under an epoch-free key; the next call
        recomputes once and is stable from then on.
        """
        modules = self._rng_drawing_funcs.get(func_name)
        if modules is None:
            modules = self._load_rng_draw_marker(func_name)
        if not modules:
            return state_hash
        try:
            from cash.notebook.randomness import seed_epoch_component
            component = seed_epoch_component(modules)
        except ImportError:  # pragma: no cover - notebook extra absent
            return state_hash
        if not component:
            return state_hash
        return hashlib.sha256(f"{state_hash}{component}".encode('utf-8')).hexdigest()

    @staticmethod
    def _rng_marker_key(func_name: str) -> str:
        """Backend key for the "this function draws" verdict."""
        return f"cash:rngdraw:{func_name}"

    def _load_rng_draw_marker(self, func_name: str) -> set[str]:
        """Read the persisted draw verdict, caching the answer for this process.

        The verdict is learned by OBSERVING a call, so it lives in memory -- and
        a kernel restart or a fresh `python run.py` throws it away. That is fatal
        for the case this exists to fix: restart-and-run-all gets exactly one
        call per function, so an in-memory-only verdict is never applied and the
        stale value comes straight back.

        A tiny per-function marker survives the process and can be read BEFORE
        the real key is built, which the entry's own metadata cannot (that would
        need the key it is supposed to inform). One backend read per function per
        process; misses are remembered as empty so it is not retried.
        """
        cached = self._rng_drawing_funcs.get(func_name)
        if cached is not None:
            return cached
        modules: set[str] = set()
        try:
            stored = self.backend.get(self._rng_marker_key(func_name))
            # Backends answer with ``(metadata, value)``; unwrap before reading.
            # Treating the pair itself as the payload silently yielded an empty
            # set, so every restart re-learned nothing and the stale value came
            # back -- the whole point of persisting the marker.
            if (isinstance(stored, tuple) and len(stored) == 2
                    and isinstance(stored[0], dict)):
                stored = stored[1]
            if isinstance(stored, (set, frozenset, list, tuple)):
                modules = {m for m in stored if isinstance(m, str)}
        except Exception:  # noqa: BLE001 - a marker miss must never break a call
            modules = set()
        self._rng_drawing_funcs[func_name] = modules
        return modules

    def _store_rng_draw_marker(self, func_name: str, modules: set[str]) -> None:
        """Persist the verdict so the next process applies it on its first call."""
        try:
            self.backend.set(self._rng_marker_key(func_name), set(modules))
        except Exception:  # noqa: BLE001 - best effort; correctness degrades to today's
            logger.debug("could not persist RNG draw marker for %s", func_name)

    def _note_rng_draw(self, func_name: str, pre_state: dict | None) -> bool:
        """Record which global RNG modules *func_name* just advanced."""
        if pre_state is None:
            return False
        try:
            from cash.notebook.randomness import capture_rng_state, rng_modules_changed
            changed = rng_modules_changed(pre_state, capture_rng_state())
        except (ImportError, TypeError, AttributeError):  # pragma: no cover
            return False
        # A module merely imported by the call is newly present rather than
        # advanced; only count streams that already existed.
        drew = {m for m in changed if m in pre_state}
        if not drew:
            return False
        known = self._rng_drawing_funcs.setdefault(func_name, set())
        newly = bool(drew - known)
        if newly:
            known.update(drew)
            self._store_rng_draw_marker(func_name, known)
        if not newly:
            return False
        # Only report "newly seen" -- which suppresses this call's write -- when a
        # drawn module is actually SEEDED. An unseeded draw has no epoch that can
        # change, so its frozen value is correct from the first call; skipping the
        # write there would redraw and break the freeze-from-first-call contract.
        try:
            from cash.notebook.randomness import seed_epochs
            return bool(drew & set(seed_epochs()))
        except ImportError:  # pragma: no cover - notebook extra absent
            return False

    @staticmethod
    def _capture_rng_pre_state() -> dict | None:
        """Snapshot the global RNG streams, or None if unavailable."""
        try:
            from cash.notebook.randomness import capture_rng_state
            return capture_rng_state()
        except (ImportError, TypeError, AttributeError):  # pragma: no cover
            return None

    def _resolve_cache_key(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
        call_start: float,
    ) -> Any:
        """Compute and return (cache_key, current_state_hash, args_hash) or call func directly on failure.

        Returns a 3-tuple on success, or a callable-result sentinel tuple
        ``(_CACHE_MISS, None, None)`` when args cannot be hashed, or raises
        nothing (logs and returns func result wrapped in a tuple) on key error.
        Actually returns either:
          - (cache_key_str, state_hash_str, args_hash_str)  - normal
          - (_CACHE_MISS, result, 'unhashable')             - unhashable args
          - (_CACHE_MISS, result, 'error')                  - key generation error
        """
        # One reset per key build: both _fold_closure and
        # _fold_read_globals contribute, so neither may clear it.
        self._pending_capture_watch = {}
        # Outside the try below, which catches TypeError/ValueError from key
        # building: an exception from the user's own body must not be caught
        # there and the body run a second time.
        unkeyable = self._refresh_helper_bindings(func, func_name)
        if unkeyable is not None:
            result = func(*args, **kwargs)
            self._log_decorator_call(
                func_name, cache_hit=False,
                execution_time=time.perf_counter() - call_start,
                args_hash='unkeyable', cache_key='', miss_detail=unkeyable,
            )
            return (_CACHE_MISS, result, 'unkeyable')
        try:
            current_state_hash = self._state_hasher.compute(
                func_name, own_source_override=self._pin_own_source(func),
            )
            current_state_hash = self._fold_closure(func, func_name, current_state_hash)
            folded_defaults = self._fold_defaults(func, func_name, current_state_hash)
            if folded_defaults is None:
                # An unhashable default: we cannot tell whether it changed, so
                # caching at all risks a stale result. Run uncached.
                result = func(*args, **kwargs)
                self._log_decorator_call(func_name, cache_hit=False, execution_time=time.perf_counter() - call_start, args_hash='unhashable', cache_key='')
                return (_CACHE_MISS, result, 'unhashable')
            current_state_hash = folded_defaults
            current_state_hash = self._fold_bound_self(func, func_name, current_state_hash)
            current_state_hash = self._fold_read_globals(func, func_name, current_state_hash)
            current_state_hash = self._fold_helper_read_globals(func, func_name, current_state_hash)
            current_state_hash = self._fold_dependency_read_globals(
                func, func_name, current_state_hash,
            )
            current_state_hash = self._fold_rng_epoch(func_name, current_state_hash)
            current_state_hash = self._fold_method_class_deps(func, args, current_state_hash)
            # ONE canonicalisation, fed to both the code channel and the value
            # channel. `_fold_code_args` on the RAW arguments saw a class
            # passed explicitly but not the identical class arriving as a
            # parameter DEFAULT, so `build()` and `build(Schema)` -- the same
            # logical call -- produced two cache keys and two executions,
            # breaking the unification invariant stated at the top of this
            # class. Measured: 2 executions before, 1 after, with a primitive
            # default (`add(k=7)`) as the control that always was 1.
            normalized_args = self._normalize_call_args(func_name, args, kwargs)
            if func_name in self._seed_params:
                self._warn_if_seed_is_none(func, func_name, args, kwargs)
            current_state_hash = self._fold_code_args(
                *normalized_args, current_state_hash, func_name=func_name)
            dynamic_state_hash = self._resolve_dynamic_dependencies(func_name, dynamic_depends_on, args, kwargs)
            args_hash = self._serialize_args(func_name, args, kwargs, normalized=normalized_args)
            self._note_arg_cost(func_name)
            if args_hash is None:
                arg_type_name = self._first_unhashable_arg_type(args, kwargs)
                if arg_type_name == "<unknown>":
                    which = (
                        "an argument could not be hashed, and cash cannot say "
                        "which -- the value is nested inside a container"
                    )
                    suggestion = (
                        "find the nested value, then register a hasher for its "
                        "type with cash.register_hasher(SomeType, ...) or pass "
                        "something hashable in its place."
                    )
                else:
                    which = f"an argument of type {arg_type_name} could not be hashed"
                    suggestion = _unhashable_arg_fix(
                        self._first_unhashable_arg(args, kwargs), arg_type_name)
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    arg_type_name,
                    f"@cash.cache on {func_name}: {which}, so this call and every "
                    f"call like it does not cache.",
                    code="KEY-UNHASHABLE-ARG",
                    fix=suggestion,
                )
                result = func(*args, **kwargs)
                self._log_decorator_call(func_name, cache_hit=False, execution_time=time.perf_counter() - call_start, args_hash='unhashable', cache_key='')
                return (_CACHE_MISS, result, 'unhashable')
            cache_key = self._compute_cache_key(func_name, current_state_hash, dynamic_state_hash, args_hash)
            return (cache_key, current_state_hash, args_hash)
        except (TypeError, ValueError, pickle.PicklingError, AttributeError) as e:
            arg_type_name = self._first_unhashable_arg_type(args, kwargs)
            if arg_type_name == "<unknown>":
                hint = (
                    "check the function's arguments -- cash could not identify "
                    "the offending type; if the exception does not belong to "
                    "your code, report it as a bug with the traceback."
                )
            elif isinstance(self._first_unhashable_arg(args, kwargs), _CODE_VALUE_TYPES):
                hint = _CODE_ARG_FIX
            else:
                hint = (
                    f"register a hasher with "
                    f"cash.register_hasher({arg_type_name}, ...) if "
                    f"{arg_type_name} is the unhashable argument."
                )
            self._warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                arg_type_name,
                f"@cash.cache on {func_name}: cache-key generation raised "
                f"{type(e).__name__} ({e}) somewhere it did not anticipate, so "
                f"this call does not cache.",
                code="KEY-BUILD-FAILED",
                fix=hint,
            )
            result = func(*args, **kwargs)
            self._log_decorator_call(func_name, cache_hit=False, execution_time=time.perf_counter() - call_start, args_hash='error', cache_key='')
            return (_CACHE_MISS, result, 'error')

    def _explain_call(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl: int | None,
        args: tuple,
        kwargs: dict,
    ) -> CacheExplanation:
        """Return why a call with these args would hit or miss the cache.

        Pure introspection - does NOT call ``func``, does NOT touch
        `Cash` stats, does NOT emit warnings, and does NOT
        mutate the backend. Mirrors the logic of `_resolve_cache_key`
        + `_try_get_cached` so that the answer reflects what would
        actually happen on the next real call.

        See `CacheExplanation` for the return shape.
        """
        if self.config.disable:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_DISABLED,
                func_name=func_name,
                details={'hint': 'Caching is disabled (disable=True / '
                                 'CASH_DISABLE): every call runs the function.'},
            )
        # Populate the dependency closure first so the state hash matches what
        # a real call computes (otherwise explain() reports a stale pre-analysis
        # key and a false `no_entry` - finding #7). This only fills internal
        # analysis caches; it does not warn, run the function, or touch the
        # backend.
        self._ensure_closure_analyzed(func)
        # Same binding check a real call makes first: a patched helper
        # changes the key, and a mock means the call would run uncached.
        unkeyable = self._refresh_helper_bindings(func, func_name)
        if unkeyable is not None:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'error': MISS_MOCKED,
                    'hint': f"{unkeyable}, which has no code to key, so the call "
                            f"would run uncached.",
                },
            )

        # Build cache key (silently - explain() does not warn).
        try:
            current_state_hash = self._state_hasher.compute(
                func_name, own_source_override=self._pin_own_source(func),
            )
            current_state_hash = self._fold_closure(func, func_name, current_state_hash)
            folded_defaults = self._fold_defaults(
                func, func_name, current_state_hash, warn=False,
            )
            if folded_defaults is None:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_KEY_UNCOMPUTABLE,
                    func_name=func_name,
                    details={
                        'error': 'unhashable parameter default',
                        'hint': (
                            'A parameter default could not be hashed, so cash '
                            'cannot detect a change to it and will not cache '
                            'this call.'
                        ),
                    },
                )
            current_state_hash = folded_defaults
            current_state_hash = self._fold_bound_self(func, func_name, current_state_hash, warn=False)
            current_state_hash = self._fold_read_globals(func, func_name, current_state_hash)
            current_state_hash = self._fold_helper_read_globals(func, func_name, current_state_hash)
            current_state_hash = self._fold_dependency_read_globals(
                func, func_name, current_state_hash,
            )
            # Mirrors `_resolve_cache_key`: without this the predicted key
            # would differ from the one a real call builds for exactly the
            # arguments this feature exists for, so `explain()` would report
            # `no_entry` for a call that in fact hits. `warn=False` because
            # inspecting an explanation must stay silent. Canonicalised once
            # and reused below, exactly as `_resolve_cache_key` does.
            normalized_args = self._normalize_call_args(func_name, args, kwargs)
            current_state_hash = self._fold_code_args(
                *normalized_args, current_state_hash, warn=False, func_name=func_name,
            )
        except (TypeError, ValueError, RuntimeError) as e:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'error': f'{type(e).__name__}: {e}',
                    'hint': 'Dependency state hash computation raised.',
                },
            )

        try:
            dynamic_state_hash = self._resolve_dynamic_dependencies_silent(
                dynamic_depends_on, args, kwargs,
            )
        except (TypeError, ValueError, RuntimeError, AttributeError, OSError) as e:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'error': f'{type(e).__name__}: {e}',
                    'hint': 'dynamic_depends_on resolver raised.',
                },
            )

        try:
            args_hash = self._serialize_args(
                func_name, args, kwargs, normalized=normalized_args,
            )
        except (TypeError, ValueError, pickle.PicklingError, AttributeError) as e:
            arg_type_name = self._first_unhashable_arg_type(args, kwargs)
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'arg_type': arg_type_name,
                    'error': f'{type(e).__name__}: {e}',
                    'hint': (
                        _unhashable_arg_fix(
                            self._first_unhashable_arg(args, kwargs), arg_type_name)
                        if arg_type_name != '<unknown>'
                        else 'Could not identify the offending argument.'
                    ),
                },
            )

        if args_hash is None:
            arg_type_name = self._first_unhashable_arg_type(args, kwargs)
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'arg_type': arg_type_name,
                    'hint': (
                        _unhashable_arg_fix(
                            self._first_unhashable_arg(args, kwargs), arg_type_name)
                        if arg_type_name != '<unknown>'
                        else 'Could not identify the offending argument; likely a nested unpicklable value.'
                    ),
                },
            )

        cache_key = self._compute_cache_key(
            func_name, current_state_hash, dynamic_state_hash, args_hash,
        )
        frozen_args = self._frozen_arg_names(normalized_args)

        raw_metadata, _data = self.backend.get(cache_key)
        if raw_metadata is None:
            details = {
                'hint': (
                    'No matching cache entry. First call with these arguments, '
                    'or the cache was cleared.'
                ),
            }
            # A tracked dynamic dependency that changed produces a NEW cache key,
            # so the miss surfaces as no_entry rather than file_changed. Make the
            # explanation say so and list what's tracked (finding #8).
            dyn_ids = self._describe_dynamic_dependencies(dynamic_depends_on, args, kwargs)
            if dyn_ids:
                details['dynamic_dependencies'] = dyn_ids
                details['hint'] = (
                    'No matching cache entry. Either the first call with these '
                    'arguments, or a tracked dynamic dependency changed - a '
                    'dynamic_depends_on change yields a new cache key, so it '
                    'shows up here as no_entry, not file_changed. Tracked '
                    'dynamic dependencies: ' + ', '.join(dyn_ids) + '.'
                )
            # What this process knows about the key says more than "first call
            # or cleared": that it was never stored, why, or that it expired
            # under the ttl it was WRITTEN with -- which a backend drops on
            # read, so the entry looks absent (round 17).
            kind, why = self._absent_entry_reason(func_name, cache_key)
            if kind == MISS_TTL:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_TTL_EXPIRED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={'why': why},
                )
            details['why'] = f"{kind}: {why}"
            if kind != MISS_FIRST and 'dynamic_dependencies' not in details:
                del details['hint']     # the generic guess, now that we know
            if frozen_args:
                details['frozen_args'] = frozen_args
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_NO_ENTRY,
                func_name=func_name,
                cache_key=cache_key,
                details=details,
            )

        metadata = CacheMetadata.from_dict(raw_metadata)

        # TTL check - match _validate_ttl semantics: only if ttl was set
        # at decoration time.
        if ttl is not None:
            timestamp = metadata.timestamp or 0
            age = time.time() - timestamp
            if age > ttl:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_TTL_EXPIRED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={
                        'ttl_seconds': ttl,
                        'age_seconds': age,
                        'cached_at': timestamp,
                    },
                )

        # Auto-tracked file deps freshness. Routed through the SAME
        # content-authoritative helper a real lookup uses - comparing
        # raw mtime/size here made explain() report file_changed / 'mtime
        # changed' after a touch while the actual call hit. A diagnostic that
        # contradicts the behavior it describes is worse than none.
        if metadata.auto_file_deps:
            stale = self._stale_file_deps(metadata)
            if stale:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_FILE_CHANGED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={'changed_files': stale,
                             'file_deps': _describe_file_deps(metadata.auto_file_deps)},
                )

        timestamp = metadata.timestamp or 0
        details = {
            'cached_at': timestamp,
            'cache_age_seconds': time.time() - timestamp if timestamp else None,
            'execution_time_saved': metadata.execution_time or 0.0,
        }
        if metadata.auto_file_deps:
            details['file_deps'] = _describe_file_deps(metadata.auto_file_deps)
        if frozen_args:
            details['frozen_args'] = frozen_args
        return CacheExplanation(
            would_hit=True,
            reason=EXPLAIN_HIT,
            func_name=func_name,
            cache_key=cache_key,
            details=details,
        )

    def _frozen_arg_names(self, normalized_args: tuple[tuple, dict]) -> list[str]:
        """`explain()`'s list of arguments keyed by a frozen=True producer."""
        args, kwargs = normalized_args
        names = []
        for name, value in [*((f"#{i}", v) for i, v in enumerate(args)), *kwargs.items()]:
            if getattr(value, "_cash_lineage_src", None) == LINEAGE_SRC_FROZEN or (
                    self._frozen_arrays and id(value) in self._frozen_arrays):
                producer = getattr(value, "_cash_lineage_producer", None) or (
                    self._frozen_arrays.get(id(value), [None, None])[1])
                names.append(f"{name} (the result of {producer}, declared frozen)")
        return names

    def _resolve_dynamic_dependencies_silent(
        self,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> str:
        """Variant of `_resolve_dynamic_dependencies` that re-raises
        instead of warning - used by `_explain_call` so introspection
        never emits warnings as a side effect."""
        if not dynamic_depends_on:
            return ""
        dynamic_state_parts = []
        resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]
        for resolver in resolvers:
            ds_result = resolver(*args, **kwargs)
            dss = ds_result if isinstance(ds_result, list) else [ds_result]
            for ds in dss:
                if isinstance(ds, DataSource):
                    # state_token() is the source's change token (mtime /
                    # version / digest); it warns on a bool that can't track.
                    dynamic_state_parts.append(str(ds.state_token()))
        if dynamic_state_parts:
            return hashlib.sha256(":".join(sorted(dynamic_state_parts)).encode('utf-8')).hexdigest()
        return ""

    def _describe_dynamic_dependencies(
        self,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> list[str]:
        """Best-effort list of the ``DataSource`` ids a function's
        ``dynamic_depends_on`` resolves to for these args - so ``explain()`` can
        report *what* is being tracked. Returns ``[]`` when there are none or
        resolution fails (introspection must never raise)."""
        if not dynamic_depends_on:
            return []
        resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]
        ids: list[str] = []
        for resolver in resolvers:
            try:
                ds_result = resolver(*args, **kwargs)
            except Exception:  # noqa: BLE001 - explain() is best-effort
                continue
            dss = ds_result if isinstance(ds_result, list) else [ds_result]
            for ds in dss:
                if isinstance(ds, DataSource):
                    try:
                        ids.append(ds.get_id())
                    except Exception:  # noqa: BLE001
                        ids.append(repr(ds))
        return ids

    def _first_unhashable_arg_type(self, args: tuple, kwargs: dict) -> str:
        """Return the qualname of the argument that could not be hashed, or '<unknown>'.

        Used to attribute CashCacheIneffectiveWarning to a concrete type name
        so the user knows which register_hasher() call to add. See
        `_first_unhashable_arg` for how the argument is found.
        """
        suspect = self._first_unhashable_arg(args, kwargs)
        return "<unknown>" if suspect is _NO_SUSPECT else type(suspect).__qualname__

    def _first_unhashable_arg(self, args: tuple, kwargs: dict) -> Any:
        """The argument that could not be hashed, or ``_NO_SUSPECT``.

        Each candidate is hashed ALONE and the first that fails is named. It
        used to be simply the first argument of a non-built-in type, so
        ``score(df, lambda d: d * 2)`` blamed the DataFrame and advised a
        DataFrame hasher -- which cash rejects, and which with override=True
        would re-key every DataFrame function -- while the lambda was the
        culprit (round 18). This runs only on the failure path. Strings,
        numbers, None and built-in containers are skipped: a scalar always
        hashes, and a container holding the culprit is reported as "nested",
        which says more than naming the list. When no single candidate fails
        on its own, the first non-built-in is the best remaining guess.
        """
        BUILTIN_OK = (str, int, float, bool, type(None), bytes, list, dict, tuple, set, frozenset)
        candidates = [a for a in (*args, *kwargs.values()) if not isinstance(a, BUILTIN_OK)]
        for candidate in candidates:
            try:
                self._hash_arg_payload((candidate,), {})
            except Exception:  # noqa: BLE001 - exactly what we are looking for
                return candidate
        return candidates[0] if candidates else _NO_SUSPECT

    def _try_get_cached(
        self,
        cache_key: str,
        metadata: CacheMetadata | None,
        cached_data: Any,
        call_start: float,
        args_hash: str,
        func_name: str,
        ttl: int | None,
    ) -> Any:
        """Return cached_data if valid, else _CACHE_MISS sentinel.

        Key-presence is determined by ``metadata is not None`` - the
        backend contract is that absent keys return ``(None, None)``,
        so a non-None metadata view with a ``None`` data value still
        counts as a hit (a function that legitimately returned ``None``).

        Auto-tracked file dependencies stored in
        ``metadata.auto_file_deps`` are re-checked here; any file whose
        content differs from what was recorded forces a miss so the
        function re-reads the changed file.
        """
        if metadata is None:
            self._note_miss(func_name, cache_key, self._absent_entry_reason(func_name, cache_key))
            return _CACHE_MISS
        try:
            self._validate_ttl(metadata, ttl)
            if not self._auto_file_deps_fresh(metadata):
                self._note_miss(func_name, cache_key, (
                    MISS_FILE, self._describe_stale_files(metadata)))
                return _CACHE_MISS
            if not self._chunks_are_intact(cache_key, metadata):
                self._note_miss(func_name, cache_key, (
                    MISS_INCOMPLETE, "a chunk of the stored result is missing"))
                return _CACHE_MISS
            # If this hit happens *inside* another cached function's
            # computation, replay the files this entry depends on into the
            # enclosing tracker, so the outer function records them too.
            # Without this, a dependency that was already cached before the
            # consumer's first run hides its file deps behind a cache hit
            # and the consumer never invalidates when that file changes.
            self._propagate_file_deps_to_active_tracker(metadata)
            # Re-attach the lineage hash to the restored value. It's a plain
            # attribute that doesn't survive pickling, so a value restored
            # from disk would otherwise lose it - and a downstream cached
            # function would fall back to content-hashing under a DIFFERENT
            # key than when the upstream was freshly computed, recomputing
            # needlessly. The hash is deterministic from (cache_key,
            # auto_file_deps), both available here.
            self._attach_lineage(cached_data, cache_key, metadata.auto_file_deps, ttl=ttl,
                                 func_name=func_name)
            self._last_key[func_name] = cache_key
            self._log_decorator_call(
                func_name, cache_hit=True,
                execution_time=time.perf_counter() - call_start,
                args_hash=args_hash, cache_key=cache_key,
                time_saved=metadata.execution_time or 0.0,
            )
            return cached_data
        except CacheExpiredError:
            age = time.time() - (metadata.timestamp or 0)
            self._note_miss(func_name, cache_key, (
                MISS_TTL, f"the entry is {age:.1f}s old and ttl={ttl}s"))
        except (TypeError, KeyError) as e:
            self._warn_metadata_invalid(func_name, e)
            self._note_miss(func_name, cache_key, (
                MISS_INCOMPLETE, "the stored entry's metadata did not validate"))
        return _CACHE_MISS

    # -- why a call missed ---------------------------------------------------
    #
    # Round 17: four of five testers could not find out why a call recomputed.
    # The reasons below are decided where the lookup fails, from what that
    # lookup saw plus what this process remembers about the key -- never by
    # re-deriving the key, which would cost every call to explain a few.

    def _note_miss(self, func_name: str, cache_key: str, reason: tuple[str, str]) -> None:
        """Hold *reason* for the `_log_decorator_call` that reports this miss."""
        if len(self._pending_miss) > _STORE_OUTCOMES_MAX:
            # Only a call that raised leaves one behind; never let those pile up.
            self._pending_miss.clear()
        self._pending_miss[cache_key] = reason
        self._last_key[func_name] = cache_key

    def _absent_entry_reason(self, func_name: str, cache_key: str) -> tuple[str, str]:
        """Why there is no entry for *cache_key*. Reads state; changes none.

        This process's own history first. With none -- the first call of a
        function in a fresh process, which is where a script's misses are --
        the keys earlier runs stored for this function, recorded beside the
        cache (`_record_stored_key`). Without them every such miss read "no
        earlier run left one on disk", including after a code edit and a TTL
        expiry, whose entries were in fact on disk (round 18, all five
        testers).
        """
        outcome = self._store_outcomes.get(cache_key)
        if outcome is not None:
            if outcome.get("not_stored"):
                return MISS_NOT_STORED, outcome["not_stored"]
            written_ttl = outcome.get("ttl")
            age = time.time() - outcome.get("stored_at", 0)
            if written_ttl is not None and age > written_ttl:
                return MISS_TTL, f"written {age:.1f}s ago with ttl={written_ttl}s"
            return MISS_GONE, ("stored earlier in this process and since "
                               "evicted or cleared")
        previous = self._last_key.get(func_name)
        since = "since the last call"
        if previous is None or previous == cache_key:
            record = self._stored_keys(func_name)
            if cache_key in record:
                stored_at, written_ttl = record[cache_key]
                age = time.time() - stored_at
                if written_ttl is not None and age > written_ttl:
                    return MISS_TTL, (f"stored {age:.0f}s ago by an earlier run, "
                                      f"with ttl={written_ttl}s")
                return MISS_GONE, ("an earlier run stored it; it has since been "
                                   "evicted or cleared")
            others = [key for key in record if key != cache_key]
            if previous is None and others:
                previous = others[-1]
                since = "since an earlier run stored it"
            if previous is None or previous == cache_key:
                return MISS_FIRST, ("the first call with these arguments in this "
                                    "process, and no earlier run stored one")
        # Keys are `func:state:dynamic:args`; the parts that moved say why.
        old = previous.rsplit(":", 3)
        new = cache_key.rsplit(":", 3)
        if len(old) != 4 or len(new) != 4:
            return MISS_FIRST, "no entry for this key"
        moved = []
        if old[1] != new[1]:
            moved.append((MISS_CODE, "the function's code, a helper it calls, or "
                                     f"a value it reads changed {since}"))
        if old[2] != new[2]:
            moved.append((MISS_DYNAMIC, "a dynamic_depends_on source changed"))
        if old[3] != new[3]:
            moved.append((MISS_ARGS, "called with arguments not seen "
                                     + ("on the last call" if since == "since the last call"
                                        else "in the last run")))
        if not moved:
            return MISS_FIRST, "no entry for this key"
        return moved[0][0], "; and ".join(detail for _, detail in moved)

    @staticmethod
    def _stale_file_deps(metadata: CacheMetadata) -> dict[str, str]:
        """``{path: what changed}`` for each recorded dependency that moved.

        The same freshness check a lookup makes, so the answer cannot
        contradict the behaviour it explains.
        """
        from cash.notebook.file_dep_snapshot import (
            dep_path_for_this_process,
            file_dep_is_fresh,
        )
        stale: dict[str, str] = {}
        for path, recorded in (metadata.auto_file_deps or {}).items():
            here = dep_path_for_this_process(path, recorded)
            is_fresh, why = file_dep_is_fresh(here, recorded)
            if not is_fresh:
                stale[here] = _STALE_REASON_TEXT.get(why or "", "changed")
        return stale

    def _describe_stale_files(self, metadata: CacheMetadata) -> str:
        stale = self._stale_file_deps(metadata)
        if not stale:
            return "a file it read"
        path, why = next(iter(stale.items()))
        more = f" and {len(stale) - 1} more" if len(stale) > 1 else ""
        return f"{path} ({why}){more}"

    #: Keys remembered per function beside the cache, most recent last.
    _STORED_KEYS_MAX = 64

    def _stored_keys_path(self, func_name: str) -> str | None:
        """Where this function's recently stored keys are recorded, or None.

        Beside the entries, in the directory of the backend actually built --
        never a configured path, so reading a miss reason cannot create a
        cache directory -- and only for a backend that has a local directory.
        """
        backend = self._backend
        for candidate in (backend, *getattr(backend, "backends", ())):
            path = getattr(candidate, "cache_dir", None)
            if isinstance(path, str) and path:
                name = hashlib.sha256(func_name.encode("utf-8")).hexdigest()[:32]
                return os.path.join(path, ".keys", f"{name}.json")
        return None

    def _stored_keys(self, func_name: str) -> dict[str, list]:
        """``{cache_key: [stored_at, ttl]}`` earlier runs recorded, oldest first."""
        path = self._stored_keys_path(func_name)
        if path is None:
            return {}
        from cash.notebook.file_tracker import untracked
        try:
            # Cash's own bookkeeping: a nested call reads this while the OUTER
            # call's file tracker is live, and it must not become that entry's
            # dependency.
            with untracked(), open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        keys = data.get("keys") if isinstance(data, dict) else None
        return keys if isinstance(keys, dict) else {}

    def _record_stored_key(self, func_name: str, cache_key: str, ttl: int | None) -> None:
        """Remember that *cache_key* reached disk, for the next process's reasons.

        One small file per function, rewritten on each persisted store (the
        compute that just ran dwarfs it). Concurrent writers race to the last
        rename; the loser's key is missing from the record, which costs a
        vaguer reason, never a wrong answer. Never raises.
        """
        path = self._stored_keys_path(func_name)
        if path is None:
            return
        try:
            keys = self._stored_keys(func_name)
            keys.pop(cache_key, None)
            keys[cache_key] = [time.time(), ttl]
            while len(keys) > self._STORED_KEYS_MAX:
                keys.pop(next(iter(keys)))
            from cash.notebook.file_tracker import untracked
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with untracked():
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"func": func_name, "keys": keys}, fh)
                os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - a diagnostic aid; the store succeeded
            logger.debug("could not record the stored key for %s", func_name, exc_info=True)

    def _remember_outcome(self, cache_key: str, outcome: dict[str, Any]) -> None:
        outcome.setdefault("at", time.time())
        self._store_outcomes[cache_key] = outcome
        self._store_outcomes.move_to_end(cache_key)
        while len(self._store_outcomes) > _STORE_OUTCOMES_MAX:
            self._store_outcomes.popitem(last=False)

    def _store_refusal(
        self, func: Callable, func_name: str, res: Any, rng_new: bool,
        cache_if: Callable[[Any], bool] | None, tracker: Any,
        capture_watch: Any = _NO_WATCH,
    ) -> str | None:
        """Why *res* must not be stored, or ``None`` to store it.

        One decision for the sync, async and streaming paths, which used to
        carry three copies of it -- and it now says WHY, because "not stored"
        is the answer to the next call's "why did that miss?".
        """
        # Skip the write exactly once when THIS call revealed that the
        # function draws: its key was built before we knew, so an entry stored
        # now carries no seed epoch and would be rebuilt and matched forever --
        # serving a result computed under a seed the user has since changed.
        # The next call keys it correctly.
        refusal = ("its first call drew random numbers the key did not yet "
                   "cover; the next call keys them" if rng_new else None)
        if refusal is None and cache_if is not None:
            try:
                refusal = None if cache_if(res) else "cache_if returned False"
            except Exception as e:  # noqa: BLE001 - user predicate
                self._warn_cache_if_raised(func_name, e)
                refusal = "cache_if raised"
        # After the body ran, before deciding to store: a provisional global
        # this call moved must stop being folded (CAS-270).
        if capture_watch is not _NO_WATCH:
            self._learn_mutating_captures(func, func_name, capture_watch)
        if refusal is None and self._refuses_identity_coupled(func_name, res):
            refusal = "the result is tied to the identity of an object in memory"
        if refusal is None and self._inputs_moved_during_call(func_name, tracker):
            refusal = "a file it read changed while it ran"
        return refusal

    def _note_not_stored(self, cache_key: str, refusal: str) -> None:
        self._remember_outcome(cache_key, {"not_stored": refusal})

    @staticmethod
    def _not_persisted_reason(stored_meta: dict[str, Any], execution_time: float) -> str | None:
        """Why a stored value reached only RAM, or ``None`` if it went further.

        Only a tiered backend says where a value landed; anything else reports
        nothing, and nothing is claimed.
        """
        tiers = stored_meta.get("storage")
        skipped = stored_meta.get("persist_skipped")
        if not isinstance(tiers, list) or skipped is None or any(t != "RAM" for t in tiers):
            return None
        if skipped == "size":
            return "too big for the persistent tier's size cap"
        from cash.backends.factory import _SMART_PERSIST_COMPUTE_FLOOR_S as floor
        if execution_time < floor:
            return f"under the {floor:g}s persistence floor"
        return "the cost model judged restoring it no cheaper than recomputing it"

    @staticmethod
    def _snapshot_tracked_deps(tracker: Any,
                               code_module: str | None = None) -> dict[str, dict[str, Any]] | None:
        """Snapshot everything *tracker* saw this call read - local and remote.

        Both land in one dict: they answer the same question ("did what this
        call read change since?"), and every consumer already routes that
        question through ``file_dep_is_fresh``, which branches on the entry.
        Remote entries cost one metadata request each to snapshot; that is the
        price of the read being tracked at all, and it is small against the
        download the entry exists to avoid.
        """
        from cash.notebook.file_dep_snapshot import (
            attach_code_relative,
            snapshot_dependencies,
        )
        deps = snapshot_dependencies(
            tracker.get_accessed_files(),
            tracker.get_accessed_remote_urls(),
            tracker.get_absent_files(),
        )
        # A file beside the function's own code is part of this INSTALL, not a
        # fixed location: record where it sits relative to the code, so another
        # install or release checks its own copy (CAS-108).
        return attach_code_relative(deps, code_module) or None

    @staticmethod
    def _propagate_file_deps_to_active_tracker(metadata: CacheMetadata) -> None:
        """Register this entry's recorded deps with the enclosing
        ``FileAccessTracker`` (if any), so a cached function that calls this
        one on a *hit* still inherits its dependencies. Best-effort: any
        failure (no tracker active, import issue) is silently ignored."""
        snap = getattr(metadata, "auto_file_deps", None)
        if not snap:
            return
        try:
            from cash.notebook.file_tracker import _active_tracker
            tracker = _active_tracker.get()
        except Exception:  # noqa: BLE001 - tracking is best-effort
            return
        if tracker is None:
            return
        for path, recorded in snap.items():
            # A remote entry must go back onto the remote channel: routed to
            # ``_add_tracked`` it would enter the file set, be stat'ed, and be
            # dropped - so the outer entry would silently lose the dependency.
            if isinstance(recorded, dict) and recorded.get("remote"):
                tracker._add_tracked_remote(path)
            else:
                # The file THIS process would read -- another install's copy
                # would give the enclosing entry the writer's path (CAS-108).
                from cash.notebook.file_dep_snapshot import dep_path_for_this_process
                tracker._add_tracked(dep_path_for_this_process(path, recorded))

    def _auto_file_deps_fresh(self, metadata: CacheMetadata) -> bool:
        """Return True if every file recorded in ``metadata.auto_file_deps``
        still matches on disk.

        Auto-tracked deps are captured during the first compute via
        `cash.notebook.file_tracker.FileAccessTracker` and stored as
        ``{path: {'mtime': float, 'size': int, 'hash': str}}``. If a recorded
        path is gone or its content changed, we invalidate the cache so the next
        compute re-reads the file. A path that disappears is also a change.

        Freshness is decided by the shared
        :func:`cash.notebook.file_dep_snapshot.file_dep_is_fresh` - the same
        content-authoritative check the notebook path uses, so
        the two subsystems can't drift. ``(mtime, size)`` alone was ambiguous in
        both directions: a touch (identical content, bumped mtime)
        recomputed needlessly, and a same-size edit under an indistinguishable
        mtime was missed and served stale. The helper checks the cheap size
        first and only hashes when the size matches.
        """
        snap = metadata.auto_file_deps or {}
        if not snap:
            return True  # nothing to check
        from cash.notebook.file_dep_snapshot import (
            _full_hash_max_bytes,
            dep_path_for_this_process,
            file_dep_is_fresh,
        )
        from cash.remote_source import measured_validation

        # The full-hash threshold, resolved ONCE for the pass. Reading it per
        # file costs a config merge each time, and a config merge walks the
        # directory tree looking for the project marker: profiling a 50-file hit
        # found 7,000 stat calls and 130 ms in there, three times the checking
        # it was guarding.
        full_hash_max = _full_hash_max_bytes()

        # Remote entries cost a network round trip each to check, so the check
        # itself is worth measuring - see _warn_if_validation_is_expensive.
        #
        # Local ones are measured too, on their own clock. Hashing is not free
        # either, and file deps PROPAGATE: an aggregate that calls ten cached
        # functions inherits their inputs, so a fifty-file pipeline paid for
        # fifty checks on every one of those hits. Measured at 168 ms a hit
        # before the digest memo landed, with nothing anywhere to say so -- the
        # remote channel had a cost warning and the local one, which every user
        # has, did not.
        local_seconds = 0.0
        local_count = 0
        with measured_validation() as validation:
            fresh = True
            for path, recorded in snap.items():
                is_remote = isinstance(recorded, dict) and recorded.get("remote")
                started = time.perf_counter()
                is_fresh, reason = file_dep_is_fresh(
                    dep_path_for_this_process(path, recorded), recorded, full_hash_max)
                if not is_remote:
                    local_seconds += time.perf_counter() - started
                    local_count += 1
                if not is_fresh:
                    logger.debug("[FILE_DEP] stale (%s): %s", reason, path)
                    fresh = False
                    break
        Cash._warn_if_validation_is_expensive(validation, metadata)
        self._warn_if_local_validation_is_expensive(local_seconds, local_count, metadata)
        return fresh

    def _warn_if_local_validation_is_expensive(
        self, seconds: float, count: int, metadata: CacheMetadata,
    ) -> None:
        """Say so when hashing this entry's own files costs a real share of the
        saving.

        The same rule the remote channel uses (``validation_is_expensive``):
        more than half the compute it avoids past a 0.25 s floor, or more than
        2 s outright. Shared deliberately -- "proving it fresh cost more than
        recomputing would" is one judgement, and it should not depend on whether
        the input was a file or a URL.

        After the first check of a file this is microseconds (the digest is
        memoized per process), so reaching the threshold means many
        dependencies, very large ones, or a slow filesystem. Each of those is
        something the user can act on, and none of them shows up anywhere else.
        """
        if not count or not seconds:
            return
        from cash.remote_source import validation_is_expensive

        saved = metadata.execution_time
        if not validation_is_expensive(seconds, saved):
            return
        label = metadata.func_name or "a cached call"
        against = (
            f", against {saved:.2f}s of compute it avoids"
            if saved and saved > 0 else ""
        )
        self._warn_once(
            CashCacheIneffectiveWarning,
            label,
            "local-freshness-cost",
            f"cash spent {seconds:.2f}s checking {count} tracked "
            f"{'file' if count == 1 else 'files'} for freshness on {label}"
            f"{against}, so proving the result fresh costs a serious share of "
            f"what it saves.",
            code="CACHE-FRESHNESS-COST",
            fix="depend on fewer or smaller files -- cache a summary rather than "
                "every input -- or split the function so the expensive inputs are "
                "read by a callee whose deps the aggregates do not inherit. Note "
                "that files above file_hash_full_max_bytes are sampled rather "
                "than hashed in full, which is cheaper per file but not per file "
                "COUNT.",
        )

    @staticmethod
    def _warn_if_validation_is_expensive(
        validation: Any, metadata: CacheMetadata
    ) -> None:
        """Say so when checking freshness costs a serious share of the saving.

        A freshness check that has to ask the network is the one overhead a user
        cannot see: it happens on the HIT path, where the badge shows a saving
        and nothing shows what the saving cost to establish.
        """
        if not validation.count:
            return
        from cash.remote_source import validation_is_expensive, warn_validation_cost_once

        saved = metadata.execution_time
        if validation_is_expensive(validation.seconds, saved):
            warn_validation_cost_once(
                metadata.func_name or "a cached call",
                validation.count,
                validation.seconds,
                saved,
            )

    def _chunks_are_intact(self, cache_key: str, metadata: CacheMetadata) -> bool:
        """True unless this is a chunked manifest missing some of its chunks.

        A manifest can outlive its chunks -- eviction reaches them separately,
        and until chunks carried the producer's execution_time the persistence
        gate dropped them while keeping the manifest. The reader terminates
        iteration on a missing chunk, so the result of that split was an
        entry that returned FEWER items than it stored, or none at all,
        without a word. A truncated answer is worse than a slow one, so the
        entry is treated as absent and recomputed.

        Metadata-only reads: the point is to check presence, not to load the
        payload and undo the laziness chunking exists for.

        Scope, because the docs depend on it: BOTH read paths apply this --
        ``_try_get_cached`` for the default one, and the double-checked re-read
        inside ``_compute_with_lock`` for ``use_locking=True``. Keep it that
        way. The locking path skipped it until 2026-09-06 and served a short
        iterator with no recompute, no error and no warning: measured 3 of 10
        items when a later chunk was missing, and 0 items when the first one
        was. Both paths are pinned by
        ``tests/test_core/test_iterator_caching.py``.
        """
        if getattr(metadata, "iterator_storage", None) != "chunked":
            return True
        try:
            for index in range(metadata.n_chunks or 0):
                if self.backend.get_metadata(f"{cache_key}:chunk_{index}") is None:
                    logger.debug(
                        "[CORE] chunk %d of %s is missing; treating the entry "
                        "as a miss rather than serving a short result",
                        index, cache_key,
                    )
                    return False
        except Exception:  # noqa: BLE001 - an integrity check must not break a call
            return True
        return True

    def _wrap_iterator_hit(
        self,
        cache_key: str,
        metadata: CacheMetadata | None,
        hit: Any,
    ) -> Any:
        """Wrap a cache-hit value in the right iterator class.

        Iterators (including the single-chunk case) are stored under
        an ``iterator_storage='chunked'`` manifest plus N chunk
        entries; on hit they're returned as a fresh
        ``_ChunkedCachedIterator``. Non-iterator return types live as
        a single blob and are returned as *hit* directly.

        Used by all three cache-hit paths in this module:
        `_make_wrapper` (sync unlocked), `_compute_with_lock`
        (sync locked re-read), and `_make_async_wrapper`. Keeping
        the dispatch in one place ensures the three paths can't drift.
        """
        if metadata and metadata.iterator_storage == 'chunked':
            n_chunks = metadata.n_chunks or 0
            return _ChunkedCachedIterator(self, cache_key, n_chunks)
        return hit

    def _make_wrapper(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl_decl: int | None,
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
    ) -> Callable:
        """Build and return the core caching wrapper for *func*."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_start = time.perf_counter()

            if func_name not in self._analyzed:
                # Double-checked under a per-function lock: the key is built
                # from what this populates, so two threads must not race it.
                with self._analysis_lock:
                    if func_name not in self._analyzed:
                        self._analyze_dependencies(func)
                        self._analyzed.add(func_name)
            # Inherit the shortest TTL of any TTL'd dependency (computed after
            # analysis populates the graph). `ttl` shadows the declared one for
            # the rest of the wrapper.
            ttl = self._effective_ttl(func_name, ttl_decl)

            # Everything from here to the hit/miss verdict is cash's own cost,
            # not the user's work. Two perf_counter pairs measured at 196ns
            # against a 25.5us floor for the cheapest possible cached call --
            # 0.8%, so this is not gated behind a heuristic.
            overhead_t0 = time.perf_counter()
            key_result = self._resolve_cache_key(func, func_name, dynamic_depends_on, args, kwargs, call_start)
            # Snapshot into a LOCAL immediately: a nested cached call would
            # overwrite the instance scratch before this body finishes (CAS-270).
            capture_watch = self._pending_capture_watch
            if key_result[0] is _CACHE_MISS:
                return key_result[1]
            cache_key, current_state_hash, args_hash = key_result

            raw_metadata, cached_data = self.backend.get(cache_key)
            metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
            hit = self._try_get_cached(cache_key, metadata, cached_data, call_start, args_hash, func_name, ttl)
            cash_overhead = time.perf_counter() - overhead_t0
            if hit is not _CACHE_MISS:
                self._note_effectiveness(
                    func_name, cash_overhead,
                    body_seconds=getattr(metadata, "body_seconds", None),
                    was_hit=True,
                )
                return self._wrap_iterator_hit(cache_key, metadata, hit)

            def _compute_and_store() -> Any:
                # Wrap the function call in FileAccessTracker so any
                # auto-tracked file reads (pandas/numpy/joblib/open/...)
                # are recorded as implicit cache dependencies - a later
                # content change forces a recompute.
                from cash.notebook.file_tracker import FileAccessTracker
                tracker = FileAccessTracker(getattr(func, '__globals__', None), propagate_to_parent=True)
                # Watch for side effects the STATIC analyzer cannot see, which
                # is anything happening inside an installed library. Only on
                # this (missing) path: a hit runs no body, so there is nothing
                # to observe and nothing to pay for.
                observer = self._make_effect_observer()
                # Watch the global RNG across the call: a draw inside the body is
                # an input the key cannot see statically.
                rng_pre = self._capture_rng_pre_state()
                body_seconds: float | None = None
                with tracker, observer:
                    body_t0 = time.perf_counter()
                    res = func(*args, **kwargs)
                    # The user's own work, isolated. Everything cash does sits
                    # outside this pair, which is the whole point: it is the
                    # only number that can answer "did caching pay?".
                    body_seconds = time.perf_counter() - body_t0
                    rng_new = self._note_rng_draw(func_name, rng_pre)
                    is_iter = _is_one_shot_iterator(res)

                # A generator is handed straight back, wrapped, and cached only
                # once the caller has drained it. Draining it here instead --
                # which is what this did -- meant a streamed response arrived
                # in one lump after the full latency, so `@cash.cache` changed
                # how the function behaved. `_stream_and_store` carries the
                # tracker into each production step so the lazy file reads that
                # motivated the old placement are still recorded.
                if is_iter:
                    # Logged HERE, not at exhaustion. `stats_wrapper` drains
                    # this log the moment the wrapper returns, so an entry
                    # written when the caller finishes iterating is never
                    # counted -- and the next call re-counts the stale one.
                    # The miss is a fact about the LOOKUP, which has already
                    # happened. The produce time still reaches the entry, via
                    # the manifest, which is what a later hit reports as saved.
                    self._log_decorator_call(
                        func_name, cache_hit=False,
                        execution_time=time.perf_counter() - call_start,
                        args_hash=args_hash, cache_key=cache_key,
                    )
                    return _StreamingCachedIterator(self._stream_and_store(
                        res, cache_key=cache_key, func_name=func_name,
                        metadata=metadata, tracker=tracker, observer=observer,
                        rng_new=rng_new, args=args, kwargs=kwargs,
                        args_hash=args_hash,
                        current_state_hash=current_state_hash, ttl=ttl,
                        cache_if=cache_if, chunk_max_items=chunk_max_items,
                        chunk_max_bytes=chunk_max_bytes,
                        code_module=func.__module__,
                    ))

                self._check_argument_mutation(
                    func_name, args, kwargs, args_hash, observer)
                self._report_observed_effects(func_name, observer)
                auto_file_deps = self._snapshot_tracked_deps(tracker, func.__module__)

                # Non-iterator return: existing single-blob path.
                execution_time = time.perf_counter() - call_start

                refusal = self._store_refusal(
                    func, func_name, res, rng_new, cache_if, tracker, capture_watch)
                if refusal is not None:
                    self._note_not_stored(cache_key, refusal)
                else:
                    # Attach lineage only when the value is actually stored: a
                    # lineage hash points downstream at THIS cache entry, so a
                    # cache_if-rejected (uncached) value must not carry one - it
                    # would reference an entry that was never written.
                    self._attach_lineage(res, cache_key, auto_file_deps, ttl=ttl, func_name=func_name)
                    self._store_in_cache(
                        cache_key, func_name, res, metadata, ttl,
                        current_state_hash, args_hash, execution_time,
                        auto_file_deps=auto_file_deps,
                        body_seconds=body_seconds,
                    )
                self._log_decorator_call(
                    func_name, cache_hit=False,
                    execution_time=execution_time,
                    args_hash=args_hash, cache_key=cache_key,
                )
                self._note_effectiveness(
                    func_name, cash_overhead,
                    body_seconds=body_seconds, was_hit=False,
                )
                return res

            if self.use_locking:
                return self._compute_with_lock(cache_key, func_name, ttl, args_hash, call_start, _compute_and_store)
            return _compute_and_store()

        return wrapper

    def _make_async_wrapper(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl_decl: int | None,
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
    ) -> Callable:
        """Build and return the async caching wrapper for *func*.

        Mirrors `_make_wrapper` but the wrapper is ``async def``
        and the underlying ``func()`` invocation is awaited inside a
        ``FileAccessTracker`` block. Shared helpers (``_resolve_cache_key``,
        ``_try_get_cached``, ``_store_in_cache``, ``_log_decorator_call``)
        are sync and reused as-is.
        """

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_start = time.perf_counter()

            if func_name not in self._analyzed:
                # Double-checked under a per-function lock: the key is built
                # from what this populates, so two threads must not race it.
                with self._analysis_lock:
                    if func_name not in self._analyzed:
                        self._analyze_dependencies(func)
                        self._analyzed.add(func_name)
            # Inherit the shortest TTL of any TTL'd dependency (see sync wrapper).
            ttl = self._effective_ttl(func_name, ttl_decl)

            # See the sync wrapper: this span is cash's own cost, not the
            # user's work.
            overhead_t0 = time.perf_counter()
            key_result = self._resolve_cache_key(
                func, func_name, dynamic_depends_on, args, kwargs, call_start
            )
            # See the sync wrapper: snapshot before anything else can run.
            capture_watch = self._pending_capture_watch
            if key_result[0] is _CACHE_MISS:
                # _resolve_cache_key called `func(*args, **kwargs)` on the
                # unhashable/error path. For an async function that returns
                # a coroutine - we must await it before handing back.
                result_or_coro = key_result[1]
                if inspect.iscoroutine(result_or_coro):
                    return await result_or_coro
                return result_or_coro
            cache_key, current_state_hash, args_hash = key_result

            raw_metadata, cached_data = self.backend.get(cache_key)
            metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
            hit = self._try_get_cached(
                cache_key, metadata, cached_data, call_start,
                args_hash, func_name, ttl,
            )
            cash_overhead = time.perf_counter() - overhead_t0
            if hit is not _CACHE_MISS:
                self._note_effectiveness(
                    func_name, cash_overhead,
                    body_seconds=getattr(metadata, "body_seconds", None),
                    was_hit=True,
                )
                return self._wrap_iterator_hit(cache_key, metadata, hit)

            # Async single-flight: with use_locking, coalesce concurrent awaits
            # of the same key in-process so an expensive idempotent coroutine
            # (e.g. a paid API call) under asyncio.gather computes once instead
            # of N times. The leader computes and stores; followers wait on an
            # event and then read the stored result.
            single_flight_event = None
            if self.use_locking:
                # Imported HERE, not at module scope: asyncio costs ~76ms of a
                # ~290ms `import cash`, and a synchronous user never needs it.
                # Inside a running loop it is necessarily already imported, so
                # this lookup is free exactly where it is used.
                import asyncio

                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None
                if running_loop is not None:
                    existing = self._async_inflight.get(cache_key)
                    if existing is not None and existing[0] is running_loop:
                        # Follower: wait for the leader, then read the stored value.
                        await existing[1].wait()
                        raw_metadata, cached_data = self.backend.get(cache_key)
                        if raw_metadata is not None:
                            metadata = CacheMetadata.from_dict(raw_metadata)
                            hit = self._try_get_cached(
                                cache_key, metadata, cached_data, call_start,
                                args_hash, func_name, ttl,
                            )
                            if hit is not _CACHE_MISS:
                                return self._wrap_iterator_hit(cache_key, metadata, hit)
                        # Leader stored nothing (cache_if rejected / errored):
                        # fall through and compute ourselves.
                    else:
                        # Leader: register an event the followers wait on.
                        single_flight_event = asyncio.Event()
                        self._async_inflight[cache_key] = (running_loop, single_flight_event)

            async def _compute_and_store() -> Any:
                from cash.notebook.file_tracker import FileAccessTracker
                tracker = FileAccessTracker(getattr(func, '__globals__', None), propagate_to_parent=True)
                observer = self._make_effect_observer()
                rng_pre = self._capture_rng_pre_state()
                body_seconds: float | None = None
                with tracker, observer:
                    body_t0 = time.perf_counter()
                    res = await func(*args, **kwargs)
                    body_seconds = time.perf_counter() - body_t0
                    rng_new = self._note_rng_draw(func_name, rng_pre)
                    is_iter = _is_one_shot_iterator(res)

                # An async function returning a SYNC generator streams through
                # the same way -- see `_stream_and_store`. Leaving this branch
                # on the drain-first path would have made the behaviour differ
                # between `def` and `async def` for the same generator, which
                # is exactly the kind of split nobody finds until it bites.
                if is_iter:
                    # Logged HERE, not at exhaustion. `stats_wrapper` drains
                    # this log the moment the wrapper returns, so an entry
                    # written when the caller finishes iterating is never
                    # counted -- and the next call re-counts the stale one.
                    # The miss is a fact about the LOOKUP, which has already
                    # happened. The produce time still reaches the entry, via
                    # the manifest, which is what a later hit reports as saved.
                    self._log_decorator_call(
                        func_name, cache_hit=False,
                        execution_time=time.perf_counter() - call_start,
                        args_hash=args_hash, cache_key=cache_key,
                    )
                    return _StreamingCachedIterator(self._stream_and_store(
                        res, cache_key=cache_key, func_name=func_name,
                        metadata=metadata, tracker=tracker, observer=observer,
                        rng_new=rng_new, args=args, kwargs=kwargs,
                        args_hash=args_hash,
                        current_state_hash=current_state_hash, ttl=ttl,
                        cache_if=cache_if, chunk_max_items=chunk_max_items,
                        chunk_max_bytes=chunk_max_bytes,
                        code_module=func.__module__,
                    ))

                self._check_argument_mutation(
                    func_name, args, kwargs, args_hash, observer)
                self._report_observed_effects(func_name, observer)
                auto_file_deps = self._snapshot_tracked_deps(tracker, func.__module__)

                # Non-iterator return: single-blob path (unchanged).
                execution_time = time.perf_counter() - call_start

                refusal = self._store_refusal(
                    func, func_name, res, rng_new, cache_if, tracker, capture_watch)
                if refusal is not None:
                    self._note_not_stored(cache_key, refusal)
                else:
                    # Attach lineage only when actually stored (see sync path):
                    # a cache_if-rejected value must not reference an entry that
                    # was never written.
                    self._attach_lineage(res, cache_key, auto_file_deps, ttl=ttl, func_name=func_name)
                    self._store_in_cache(
                        cache_key, func_name, res, metadata, ttl,
                        current_state_hash, args_hash, execution_time,
                        auto_file_deps=auto_file_deps,
                        body_seconds=body_seconds,
                    )
                self._log_decorator_call(
                    func_name, cache_hit=False,
                    execution_time=execution_time,
                    args_hash=args_hash, cache_key=cache_key,
                )
                self._note_effectiveness(
                    func_name, cash_overhead,
                    body_seconds=body_seconds, was_hit=False,
                )
                return res

            if single_flight_event is not None:
                try:
                    return await _compute_and_store()
                finally:
                    # Signal followers (success or failure) and free the slot.
                    self._async_inflight.pop(cache_key, None)
                    single_flight_event.set()
            return await _compute_and_store()

        return wrapper

    def _delete_backend_entries(self, func_name: str) -> None:
        """Delete all backend cache entries whose key starts with *func_name*."""
        try:
            prefix = f"{func_name}:"
            for entry in self.backend.list_entries():
                key = CacheMetadata.from_dict(entry).key or ''
                if key.startswith(prefix):
                    self.backend.delete(key)
        except (OSError, RuntimeError, KeyError):
            logger.debug("Failed to clear cache entries for %s", func_name)

    def _wrap_with_stats(
        self,
        func: Callable,
        func_name: str,
        wrapper: Callable,
        *,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = None,
        ttl: int | None = None,
        allow_random: bool = False,
    ) -> Callable:
        """Wrap *wrapper* with hit/miss stat tracking and attach introspection API.

        Dispatches on whether *func* is a coroutine function so the stats
        drain (reading ``_decorator_call_log`` for the just-finished call)
        happens AFTER the await for async, and synchronously otherwise.

        Attaches the introspection API:

        * ``cache_info()`` - hit/miss stats plus a rolling list of recent
          warnings emitted for this function.
        * ``cache_clear()`` - drop backend entries, reset stats, drop the
          warning log + dedup marks so re-warnings can fire.
        * ``explain(*args, **kwargs)`` - return a `CacheExplanation`
          for that specific call (sync, even on async wrappers).
        """
        _stats = {'hits': 0, 'misses': 0, 'total_time_saved': 0.0,
                  # What the misses were, and which results did not reach
                  # disk: the two things "1 miss" alone could not tell anyone.
                  'miss_reasons': Counter(), 'not_persisted': Counter(),
                  'not_stored': Counter(),
                  # Calls that went straight through because caching is off.
                  'bypassed': 0}
        # Shared by reference with the end-of-run summary, which otherwise has
        # no way to reach a per-wrapper closure. Last registration wins for a
        # redefined function, which matches what `cache_info()` reports.
        self._function_stats[func_name] = _stats

        def _drain_stats() -> None:
            with self._decorator_call_log_lock:
                for call in reversed(self._decorator_call_log):
                    if call['func_name'] == func_name:
                        if call['cache_hit']:
                            _stats['hits'] += 1
                            _stats['total_time_saved'] += call.get('time_saved', 0.0)
                        else:
                            _stats['misses'] += 1
                            kind = (call.get('miss_reason') or (MISS_FIRST, ""))[0]
                            _stats['miss_reasons'][kind] += 1
                            if call.get('not_stored'):
                                _stats['not_stored'][call['not_stored']] += 1
                            elif call.get('not_persisted'):
                                _stats['not_persisted'][call['not_persisted']] += 1
                        break

        def _bypass(args: tuple, kwargs: dict) -> Any:
            # A helper, not inline: a caller that captures this wrapper in a
            # closure has the wrapper's own captures folded into ITS key, and
            # `_stats` read directly there is content-hashed -- so every call
            # moved the caller's key (tests/test_core/test_cold_process_key_stability).
            _stats['bypassed'] += 1
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def stats_wrapper(*args: Any, **kwargs: Any) -> Any:
                if self.config.disable:
                    return await _bypass(args, kwargs)
                token = ACTIVE_CONFIG.set(self.config)
                try:
                    result = await wrapper(*args, **kwargs)
                finally:
                    ACTIVE_CONFIG.reset(token)
                _drain_stats()
                self._warn_unseeded_estimator_result(
                    func_name, result, allow_random)
                return result
        else:
            @functools.wraps(func)
            def stats_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Before anything else: disabled means the function, and
                # nothing of cash's -- no key, no analysis, no lookup, no store.
                if self.config.disable:
                    return _bypass(args, kwargs)
                # This instance's settings for the file checks the call makes
                # (`file_hash_full_max_bytes`); see ACTIVE_CONFIG.
                token = ACTIVE_CONFIG.set(self.config)
                try:
                    result = wrapper(*args, **kwargs)
                finally:
                    ACTIVE_CONFIG.reset(token)
                _drain_stats()
                self._warn_unseeded_estimator_result(
                    func_name, result, allow_random)
                return result

        def cache_info() -> dict[str, Any]:
            """Return cache statistics + recent warnings for this function.

            Returns:
                Dict with keys:

                * ``hits`` (int) - cache hits since this wrapper was created.
                * ``misses`` (int) - cache misses (including key-uncomputable
                  and store-failed paths).
                * ``hit_rate`` (float) - ``hits / (hits + misses)``, or 0.0.
                * ``total_time_saved`` (float) - sum of execution times that
                  were avoided by serving from cache.
                * ``miss_reasons`` (dict[str, int]) - the misses by why:
                  ``"no entry yet"``, ``"new arguments"``, ``"code or state
                  changed"``, ``"file changed"``, ``"ttl expired"``,
                  ``"not stored last time"`` and so on.
                * ``warnings`` (list[dict]) - rolling log of recent warning
                  emissions for this function. Each entry has ``category``,
                  ``message``, ``timestamp``. Capped at the last
                  ``_func_warnings_max`` (default 20) so it can't grow
                  unboundedly. Useful for spotting silent misbehavior
                  (cache_if predicate raised, lock failure, etc.) when
                  ``warnings.simplefilter`` swallowed the stderr emission.
            """
            total = _stats['hits'] + _stats['misses']
            hit_rate = _stats['hits'] / total if total > 0 else 0.0
            with self._decorator_call_log_lock:
                warnings_log = list(self._func_warnings.get(func_name, []))
            return {
                'hits': _stats['hits'],
                'misses': _stats['misses'],
                'hit_rate': hit_rate,
                'total_time_saved': _stats['total_time_saved'],
                'miss_reasons': dict(_stats['miss_reasons']),
                'warnings': warnings_log,
            }

        def cache_clear() -> None:
            """Clear all cached results for this function.

            Removes all cache entries whose key starts with the function name.
            Resets hit/miss statistics, drops the per-function warnings log,
            and forgets ``_warn_once`` dedup marks for this function so the
            next misbehavior re-warns instead of being silently swallowed.
            """
            _stats['hits'] = 0
            _stats['misses'] = 0
            _stats['total_time_saved'] = 0.0
            _stats['bypassed'] = 0
            for tally in ('miss_reasons', 'not_persisted', 'not_stored'):
                _stats[tally].clear()
            self._delete_backend_entries(func_name)
            with self._decorator_call_log_lock:
                self._func_warnings.pop(func_name, None)
                # Drop dedup marks for this function so future misbehavior
                # re-warns the user instead of staying silent.
                self._warning_keys_seen = {
                    k for k in self._warning_keys_seen if k[1] != func_name
                }

        def explain(*args: Any, **kwargs: Any) -> CacheExplanation:
            """Return why the next call with these args would hit or miss.

            See `CacheExplanation` for the return shape. Inspection
            only - does not call the underlying function, mutate stats,
            or write to the backend. Safe to call from sync code even
            on async-wrapped functions.
            """
            token = ACTIVE_CONFIG.set(self.config)
            try:
                return self._explain_call(
                    func, func_name, dynamic_depends_on, ttl, args, kwargs,
                )
            finally:
                ACTIVE_CONFIG.reset(token)

        stats_wrapper.cache_info = cache_info
        stats_wrapper.cache_clear = cache_clear
        stats_wrapper.explain = explain
        stats_wrapper.__wrapped__ = func
        # Marker so the purity analyzer treats a call to this wrapper as a
        # dependency-graph edge rather than recursing into cash's own wrapper
        # machinery (finding #9). functools.wraps copies __module__, which would
        # otherwise make the wrapper look like same-package user code.
        stats_wrapper._cash_cached = True
        # Declared TTL, exposed so the notebook statement cache can see it. A
        # ``ttl=0`` function must recompute every call; without this the
        # statement ``x = f()`` gets cached with no TTL under %cash_on and
        # freezes the value the decorator promised to refresh.
        stats_wrapper._cash_declared_ttl = ttl
        self._wrapped_funcs[func_name] = stats_wrapper
        return stats_wrapper

    def _effective_ttl(self, func_name: str, own_ttl: int | None) -> int | None:
        """The TTL actually used for *func_name*: the minimum of its own TTL and
        the TTLs of cached functions it (transitively) depends on.

        A function whose result derives from a TTL'd dependency must refresh at
        least as often as that dependency - otherwise, because ``depends_on``
        tracks source (not runtime freshness), the downstream keeps returning a
        stale value after the dependency's TTL refresh. Functions with no TTL'd
        dependency are unaffected (effective TTL == own TTL)."""
        cached = self._effective_ttl_cache.get(func_name)
        if cached is not None or func_name in self._effective_ttl_cache:
            return cached
        ttls = [t for t in self._collect_dep_ttls(func_name, set()) if t is not None]
        if own_ttl is not None:
            ttls.append(own_ttl)
        eff = min(ttls) if ttls else None
        self._effective_ttl_cache[func_name] = eff
        return eff

    def _collect_dep_ttls(self, func_name: str, visited: set[str]) -> list[int | None]:
        """TTLs of every cached function reachable from *func_name* via the
        dependency graph (cycle-guarded)."""
        if func_name in visited:
            return []
        visited.add(func_name)
        out: list[int | None] = []
        for dep in self.graph.get_dependencies(func_name):
            if dep in self._func_ttls:
                out.append(self._func_ttls[dep])
                out.extend(self._collect_dep_ttls(dep, visited))
        return out

    def _register_static_dependencies(
        self, func_name: str, depends_on: list[Callable[..., Any] | DataSource] | None
    ) -> None:
        if not depends_on:
            return
        for dep in depends_on:
            if isinstance(dep, DataSource):
                dep_id = dep.get_id()
                self.data_sources[dep_id] = dep
                self.graph.add_dependency(func_name, dep_id)
            elif callable(dep):
                dep_key = self._get_func_key(dep)
                self.graph.add_dependency(func_name, dep_key)
                # A declared callable dep that is NOT a decorated cached function
                # would contribute nothing to the state hash (the hasher only
                # folds functions/data-sources), silently breaking the documented
                # ``depends_on`` promise. Snapshot its source + a live
                # resolution path so edits/reloads invalidate the parent key.
                if dep_key not in self.functions:
                    self._register_declared_callable_dep(dep, dep_key, func_name)

    def _register_declared_callable_dep(
        self, dep: Callable[..., Any], dep_key: str, func_name: str
    ) -> None:
        """Record a plain-callable ``depends_on`` dependency's source identity.

        Stores a source-hash snapshot and a ``(module, attr_chain)`` path for
        live re-resolution (so an on-disk edit + ``importlib.reload`` is seen).
        If the dep's source cannot be hashed (builtin / C-extension), warn once
        that the declared dependency is inert rather than silently ignore it.
        """
        try:
            snapshot = self._hash_callable_source(dep)
        except (OSError, TypeError, ValueError):
            snapshot = None
        if snapshot is None:
            self._warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: depends_on={getattr(dep, '__qualname__', dep)!r} "
                f"is a callable whose source cannot be read (builtin / C-extension), "
                f"so the declaration is inert and changes to it will NOT "
                f"invalidate the cache.",
                code="KEY-DEPENDS-ON-OPAQUE",
                fix="depend on something cash can read: pass the extension's "
                    "version in as an argument, or declare a DataSource whose "
                    "token is that version or build id.",
            )
            return
        self._declared_dep_snapshots[dep_key] = snapshot
        module = getattr(dep, '__module__', None)
        qualname = getattr(dep, '__qualname__', None) or getattr(dep, '__name__', None)
        if module and qualname and '<locals>' not in qualname:
            self._declared_dep_paths[dep_key] = (module, tuple(qualname.split('.')))

    def _resolve_declared_dep_hash(self, dep_key: str) -> str | None:
        """Re-resolve a declared plain-callable dep's live source hash.

        Walks the stored ``(module, attr_chain)`` path via ``sys.modules`` and
        hashes the resolved callable's current source. Returns ``None`` on any
        resolution/hash failure so the hasher falls back to the snapshot.
        """
        path = self._declared_dep_paths.get(dep_key)
        if path is None:
            return None
        mod_name, attr_chain = path
        obj: Any = sys.modules.get(mod_name)
        if obj is None:
            return None
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        if not callable(obj):
            return None
        try:
            return self._hash_callable_source(obj)
        except (OSError, TypeError, ValueError):
            return None

    def _resolve_dynamic_dependencies(
        self,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> str:
        if not dynamic_depends_on:
            return ""

        dynamic_state_parts = []
        resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]

        for resolver in resolvers:
            try:
                # Resolver receives the same args as the function
                ds_result = resolver(*args, **kwargs)

                # Normalize to list
                dss = ds_result if isinstance(ds_result, list) else [ds_result]

                for ds in dss:
                    if isinstance(ds, DataSource):
                        dynamic_state_parts.append(str(ds.state_token()))
            except (OSError, TypeError, ValueError, AttributeError, RuntimeError) as e:
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    "",
                    f"@cash.cache on {func_name}: dynamic_depends_on resolver raised "
                    f"{type(e).__name__} ({e}), so that dependency is missing from "
                    f"the cache key and results may be stale when the underlying "
                    f"data changes.",
                    code="KEY-DYNAMIC-DEP-FAILED",
                    fix="fix the resolver -- it is called with exactly the same "
                        "arguments as the function -- and clear this function's "
                        "entries after changing the source data.",
                )

        if dynamic_state_parts:
            # Sort to ensure deterministic order if multiple sources
            return hashlib.sha256(":".join(sorted(dynamic_state_parts)).encode('utf-8')).hexdigest()
        return ""

    def _closure_written_freevars(self, code: Any) -> frozenset:
        """Free-variable names the function reassigns (``STORE_DEREF`` /
        ``DELETE_DEREF``) - i.e. ``nonlocal`` counters that drift between calls.
        Cached per code object (closures share a code object per factory)."""
        cache = self._deref_writes
        hit = cache.get(code)
        if hit is not None:
            return hit
        import dis
        written = frozenset(
            instr.argval for instr in dis.get_instructions(code)
            if instr.opname in ("STORE_DEREF", "DELETE_DEREF")
        )
        if len(cache) < 4096:
            cache[code] = written
        return written

    @staticmethod
    def _is_immutable_capture(v: Any, _depth: int = 0) -> bool:
        """True for values that are immutable and so define a closure's
        behaviour without drifting between calls. Mutable captures (dict/list/
        set/objects) are excluded: they are typically side-effect accumulators
        (e.g. a hit counter) whose value changes every call - folding those into
        the key would make every call miss."""
        if _depth > 8:
            return False
        if isinstance(v, (bool, int, float, complex, str, bytes, type(None))):
            return True
        if isinstance(v, (tuple, frozenset)):
            return all(Cash._is_immutable_capture(x, _depth + 1) for x in v)
        return False

    def _capture_unsafe_uses(self, func: Callable) -> frozenset:
        """Free-variable names whose captured object *may be mutated* by the
        function body.

        A capture is only content-foldable into the cache key when the body
        provably just READS it. Disqualifying uses of a free variable ``n``:
        method calls on it (``n.append(...)`` — any method, since we can't
        prove purity), passing it as a bare argument (the callee may mutate),
        subscript/attribute stores or aug-assigns rooted at it, and ``del``.
        Iteration, subscript reads, and arithmetic stay safe.

        Cached per code object (closures from one factory share it). When the
        source is unavailable, every free var is reported unsafe.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return frozenset()
        cached = self._capture_use_cache.get(code)
        if cached is not None:
            return cached
        freevars = set(code.co_freevars or ())
        provisional: frozenset = frozenset()
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        except SOURCE_RETRIEVAL_ERRORS:
            # No source: keep the old conservative answer. Nothing can be told
            # apart, so nothing is folded and nothing is provisional.
            result = frozenset(freevars)
        else:
            # Only mutations visible in this function's own source disqualify a
            # capture outright. "Passed to a call" is provisional: folded, then
            # confirmed by observation (CAS-270). Before this split, `sum(data)`
            # put `data` beyond the fold and the closure served stale forever.
            result = Cash._unsafe_uses_of(
                tree, freevars, bare_args=False, mutating_methods_only=True,
            )
            provisional = Cash._unsafe_uses_of(tree, freevars) - result
        if len(self._capture_use_cache) < 4096:
            self._capture_use_cache[code] = result
            # Kept in lockstep with the cache above so the two can never
            # disagree about a code object.
            self._provisional_capture_cache[code] = provisional
        return result

    @staticmethod
    def _unsafe_uses_of(tree: ast.AST, names: set[str], *,
                        bare_args: bool = True,
                        mutating_methods_only: bool = False) -> frozenset:
        """Return the subset of *names* the AST body *may mutate*.

        Disqualifying uses of a name ``n``: method calls on it
        (``n.append(...)``), passing it as a bare argument (the callee may
        mutate), subscript/attribute stores or aug-assigns rooted at it, and
        ``del``. Iteration, subscript reads, and arithmetic stay safe. Shared by
        the closure-capture and module-global folds.

        Two knobs, and both exist to move a *suspicion* out of the HARD set and
        into the provisional one, where it is folded and then confirmed at
        runtime by ``_learn_mutating_captures``.

        ``bare_args=False`` drops the "passed as an argument" rule. That refusal
        was over-broad and chose the worse failure: `sum(G)`, `len(G)`,
        `helper(G)` and `model.predict(G)` all put `G` beyond it, so a later
        `G = ...` never reached the key and the function served a stale value
        for ever, silently (CAS-270).

        ``mutating_methods_only=True`` narrows the method-call rule to methods
        that actually write -- ``append``, ``update``, ``sort`` and their
        relatives, the same table the purity analyzer uses. "Any method, since
        we cannot prove purity" made the same over-broad choice one level down,
        and a round-16 tester paid for it: a lookup table read as
        ``ALIASES.get(v, v)`` never reached the key, so editing the table
        published stale labels with nothing to see. `ALIASES[v]`, `v in
        ALIASES`, `d = ALIASES; d.get(v)` and a bare read all tracked
        correctly, which is what made it so hard to believe.
        """
        unsafe: set[str] = set()
        write_methods: frozenset[str] = frozenset()
        if mutating_methods_only:
            from cash.notebook.purity import _WRITE_METHODS
            write_methods = _WRITE_METHODS
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id in names
                        and (not mutating_methods_only or f.attr in write_methods)):
                    unsafe.add(f.value.id)
                if bare_args:
                    for a in list(node.args) + [kw.value for kw in node.keywords]:
                        if isinstance(a, ast.Starred):
                            a = a.value
                        if isinstance(a, ast.Name) and a.id in names:
                            unsafe.add(a.id)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                else:
                    targets = node.targets
                for t in targets:
                    root = t
                    while isinstance(root, (ast.Subscript, ast.Attribute, ast.Starred)):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in names:
                        unsafe.add(root.id)
        return frozenset(unsafe)

    def _fold_closure(self, func: Callable, func_name: str, state_hash: str,
                      _depth: int = 0) -> str:
        """Mix a fingerprint of *func*'s captured free variables into the
        state hash.

        Two closures produced by the same factory share a source AND a qualname
        (``factory.<locals>.f``) but capture different values - so without this
        they collide on the same cache key and return each other's results (a
        silent wrong answer, e.g. ``make(2)`` vs ``make(5)``).

        Immutable captures are folded by value. Mutable captures (lists,
        dicts, arrays — e.g. a weights vector) are folded by CONTENT HASH,
        but only when the body provably just reads them: captures
        the function reassigns (``nonlocal`` counters) or may mutate in place
        (accumulators) are skipped, so their keys don't drift call-to-call.
        A side effect of per-call content hashing: externally mutating a
        folded capture correctly invalidates the closure's entries.
        """
        closure = getattr(func, "__closure__", None)
        code = getattr(func, "__code__", None)
        if not closure or code is None:
            return state_hash
        freevars = getattr(code, "co_freevars", ()) or ()
        # Free vars the function REASSIGNS (nonlocal counters) drift across calls
        # even when their value type is immutable - exclude them.
        written = self._closure_written_freevars(code)
        unsafe = self._capture_unsafe_uses(func)
        # Captures excluded ONLY because they were passed to a call. Folded, then
        # confirmed at runtime -- same treatment as module globals (CAS-270). A
        # missing entry means "unknown": watch it rather than fold it blind.
        provisional = self._provisional_capture_cache.get(code)
        learned_mutating = self._mutating_globals.get((code, "closure"), frozenset())
        captures = []
        for name, cell in zip(freevars, closure):
            if name in written or name in learned_mutating:
                continue
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            # A captured FUNCTION is its code, so fold its source. Reaching
            # this before the `unsafe` check is the point: a capture the body
            # PASSES TO A CALL is marked unsafe and skipped (watch it, don't
            # fold it blind -- CAS-270), and calling is exactly what you do
            # with a captured function. So the strategy-factory shape
            #
            #     def make(weight_fn):
            #         @cash.cache
            #         def score(px, n): return f(px, weight_fn(n))
            #         return score
            #
            # folded NOTHING: two scorers built with different weightings share
            # a source and a qualname (`make.<locals>.score`), collided on one
            # key, and returned each other's results. Measured: `flat` and
            # `ramp` both returning 0.025001250062501867, one body execution.
            #
            # A call cannot mutate a function, so the reason `unsafe` exists
            # does not apply. Same predicate as `_fingerprint_default`, and the
            # same deliberate limit: functions, methods and builtins only. An
            # arbitrary callable INSTANCE keeps the old path rather than being
            # keyed on its class and silently sharing entries across instances
            # holding different state.
            fingerprint = self._fingerprint_default(v)
            if fingerprint is not v:
                # Source text alone collides for two lambdas sharing a line
                # (`a(lambda: "AAA"), a(lambda: "BBB")` is ONE line, so
                # `inspect.getsource` returns the same string for both).
                # Measured: both arms returned "AAA". Their code objects differ.
                inner_code = getattr(v, "__code__", None)
                if inner_code is not None:
                    fingerprint = f"{fingerprint}:{self._code_fingerprint(inner_code)}"
                # Source alone is not enough: a factory-built helper has the
                # SAME source for every parameter it was built with, so
                # `outer(2)` and `outer(3)` fingerprint identically and collide
                # again one level down (measured: both returned 20). Recurse so
                # the captured function's own captures fold under the same
                # rules. Bounded, because a wrong answer is worth a few frames
                # and a cycle is not.
                if _depth < 4:
                    fingerprint = self._fold_closure(
                        v, f"{func_name}.{name}", str(fingerprint), _depth + 1,
                    )
                captures.append((name, fingerprint))
                continue

            if self._is_immutable_capture(v):
                captures.append((name, v))
            elif name not in unsafe:
                # Read-only mutable capture: fold its content hash.
                # Unhashable content keeps the old skip behavior.
                try:
                    h = self._hash_arg_payload((v,), {})
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
                    continue
                captures.append((name, h))
                if provisional is None or name in provisional:
                    self._pending_capture_watch[name] = (h, "closure", None)
        if not captures:
            return state_hash
        clo = self._serialize_args(func_name, tuple(captures), {})
        if not clo:
            return state_hash
        return hashlib.sha256(f"{state_hash}:closure:{clo}".encode()).hexdigest()

    def _helper_capture_part(self, fn: Callable) -> str:
        """Digest of the IMMUTABLE values a helper's closure captured, or "".

        A decorator's arguments live there: ``@scale(10)`` builds a wrapper
        whose closure holds ``k=10``, so ``@scale(100)`` -- or ``@scale(K)``
        after ``K`` changed -- ran different code under an identical source and
        was served stale. Immutable values only, and never a variable the
        function reassigns (``nonlocal calls; calls += 1``): decorators often
        keep caches, counters and registries in their closures, and folding
        state that drifts on every call would make every call miss. Captured
        FUNCTIONS are followed as helpers in their own right, not here.
        """
        closure = getattr(fn, "__closure__", None)
        code = getattr(fn, "__code__", None)
        if not closure or code is None:
            return ""
        written = self._closure_written_freevars(code)
        captures = []
        for name, cell in zip(code.co_freevars, closure):
            if name in written:
                continue
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) or not self._is_immutable_capture(value):
                continue
            captures.append((name, value))
        if not captures:
            return ""
        try:
            return self._hash_arg_payload(tuple(captures), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
            return ""

    def _hash_helper_identity(self, fn: Callable) -> str:
        """A helper's identity for the key: its code, AND its parameter defaults.

        A default is evaluated once, at ``def`` time, and lives on the function
        object -- so ``def shrink(v, alpha=ALPHA)`` reads the same after
        ``ALPHA`` changes, the source digest does not move, and global folding
        never sees the name (it is not read in the body). Round 17 served 8
        wrong answers in 8 from a service whose ridge penalty was a helper's
        default (CAS-112). The cached function's own defaults were already
        folded (``_fold_defaults``); now every followed helper's are, by value,
        through the same payload hasher and the same callable fallback.

        Not inside ``_hash_callable_source``'s memo: that is keyed per CODE
        object, and two closures from one factory share a code object while
        holding different defaults. A helper with no defaults returns exactly
        the old digest, so entries already on disk keep hitting.
        """
        source = self._hash_callable_source(fn)
        captured = self._helper_capture_part(fn)
        if captured:
            source = f"{source}:captures:{captured}"
        defaults = getattr(fn, "__defaults__", None)
        kwdefaults = getattr(fn, "__kwdefaults__", None)
        wrapped = getattr(fn, "__wrapped__", None)
        if not defaults and not kwdefaults and wrapped is None:
            return source
        memo_key = id(fn)
        cached = self._helper_defaults_memo.get(memo_key)
        if (cached is not None and cached[0] is fn and cached[1] is defaults
                and cached[2] is kwdefaults):
            return cached[3]
        pos, kwd = self._defaults_of(fn)
        if not pos and not kwd:
            return source
        try:
            digest = self._hash_arg_payload(pos, kwd)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
            try:
                digest = self._hash_arg_payload(
                    tuple(self._fingerprint_default(v) for v in pos),
                    {k: self._fingerprint_default(v) for k, v in kwd.items()},
                )
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
                return source           # as before: this helper's defaults stay unfolded
        identity = f"{source}:defaults:{digest}"
        if len(self._helper_defaults_memo) >= 4096:
            self._helper_defaults_memo.clear()
        self._helper_defaults_memo[memo_key] = (fn, defaults, kwdefaults, identity)
        return identity

    @staticmethod
    def _defaults_of(func: Callable) -> tuple[tuple, dict]:
        """The parameter defaults that decide what *func* computes.

        ``__defaults__`` (positional/keyword params) and ``__kwdefaults__``
        (keyword-only params) are separate containers; both are collected.

        Wrapped callees are walked too. ``func.__defaults__`` is what the call
        literally binds, but when *func* is a ``functools.wraps`` wrapper its own
        defaults are typically empty (a ``*args, **kwargs`` passthrough) while the
        values that actually decide the result sit on ``__wrapped__`` — which is
        also what ``inspect.signature`` reports and therefore what
        ``_normalize_call_args`` binds. Folding every level is the conservative
        choice: folding a default that turns out not to bind costs at most a
        one-time miss, whereas missing one that does bind is a silent wrong
        answer.
        """
        pos: list[Any] = []
        kwd: dict[str, Any] = {}
        seen: set[int] = set()
        fn: Any = func
        depth = 0
        while fn is not None and id(fn) not in seen and depth < 8:
            seen.add(id(fn))
            pos.extend(getattr(fn, "__defaults__", None) or ())
            # Qualify by depth so a wrapper and its wrappee can't collide on a
            # shared kwonly name; sort so dict order never leaks into the key.
            level_kwd = getattr(fn, "__kwdefaults__", None) or {}
            for name in sorted(level_kwd):
                kwd[f"{depth}:{name}"] = level_kwd[name]
            fn = getattr(fn, "__wrapped__", None)
            depth += 1
        return tuple(pos), kwd

    def _fold_defaults(
        self, func: Callable, func_name: str, state_hash: str, warn: bool = True,
    ) -> str | None:
        """Mix the callee's parameter defaults into the state hash.

        A default is an input to the result exactly like a passed argument, but
        it lives on the FUNCTION OBJECT, not in the code object — so the bytecode
        fingerprint the state hash falls back to when source is unavailable
        (functions defined in an IPython cell, the documented ML path) cannot see
        it. Editing ``n_estimators=300`` to ``400`` left the key byte-identical
        and returned the 300-tree model on an instant HIT while
        ``inspect.signature`` reported 400 — a wrong answer that reads as a
        finding ("accuracy has plateaued") rather than as a bug.

        Defaults are hashed by VALUE through the same payload hasher arguments
        use, so ``register_hasher`` and the pandas/numpy-aware hashers apply
        identically. Returns ``None`` when a default cannot be hashed; the caller
        must then refuse to cache, because silently ignoring it would resurrect
        exactly the silent staleness this fold exists to prevent.
        """
        # Memo first: this runs on EVERY decorated call, so the hot path must be
        # one lookup plus one hash, with no re-walk of the function.
        # Not every callable can be weak-referenced (numpy's dispatcher can't),
        # and WeakKeyDictionary raises on LOOKUP too, not just on store.
        try:
            entry = self._defaults_pins.get(func)
            pinnable = True
        except TypeError:
            entry = None
            pinnable = False
        if entry is not None:
            # Validate against the live containers rather than trusting the
            # function object's identity: `f.__defaults__ = (400,)` rebinds them
            # on the SAME object, and a memo keyed on identity alone would pin
            # the old digest and hand back a stale result — the very failure
            # this fold exists to prevent. Only immutable defaults are pinned,
            # so comparing the containers by value is sound (and is a cheap
            # C-level compare of a tiny tuple/dict).
            pin_pos, pin_kwd, digest = entry
            if pin_pos == getattr(func, "__defaults__", None) and pin_kwd == (
                getattr(func, "__kwdefaults__", None) or {}
            ):
                return hashlib.sha256(
                    f"{state_hash}:defaults:{digest}".encode('utf-8')
                ).hexdigest()
        pos, kwd = self._defaults_of(func)
        if not pos and not kwd:
            # No defaults: leave the hash byte-identical so entries already on
            # disk for such functions keep hitting.
            return state_hash
        try:
            digest = self._hash_arg_payload(pos, kwd)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
            # A callback default (`def f(x, key=lambda v: v)`) is unpicklable but
            # is not opaque: its SOURCE defines it, which is the same fingerprint
            # register_hasher embeds for a hasher. Retry with function-valued
            # defaults replaced by that -- strictly better than dropping them (an
            # edited lambda now invalidates) and it keeps such functions
            # cacheable, which a bare refuse-to-cache would not.
            try:
                digest = self._hash_arg_payload(
                    tuple(self._fingerprint_callable_default(v) for v in pos),
                    {k: self._fingerprint_callable_default(v) for k, v in kwd.items()},
                )
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
                return self._defaults_unhashable(func_name, pos, kwd, e, warn)
        return self._finish_defaults_fold(func, state_hash, digest, pos, kwd, pinnable)

    @staticmethod
    def _code_fingerprint(code: types.CodeType, _depth: int = 0) -> str:
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
        recurses instead, bounded.
        """
        parts: list[str] = [
            code.co_code.hex(),
            repr(code.co_names),
            repr(code.co_varnames),
            repr(code.co_freevars),
        ]
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                parts.append(
                    Cash._code_fingerprint(const, _depth + 1)
                    if _depth < 4 else "<deep>"
                )
            else:
                parts.append(repr(const))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    @staticmethod
    def _fingerprint_default(v: Any) -> Any:
        """Replace a plain function/method default with a digest of its source.

        Restricted to functions, methods and builtins: their behaviour IS their
        code. An arbitrary callable INSTANCE is left alone so it takes the
        unhashable path rather than being keyed on its class and silently
        sharing entries across instances with different state.
        """
        if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(v):
            return f"__cash_callable__:{Cash._hash_callable_source(v)}"
        return v

    def _fingerprint_callable_default(self, v: Any) -> Any:
        """`_fingerprint_default`, plus what a FUNCTION default carries.

        A factory-built callable as a default (`def run(xs, fn=make(3))`)
        shares its source with every other one the factory makes; the value it
        was built with lives in its closure, and was not keyed -- `make(3)` ->
        `make(1)` served the old result (round 18). `_hash_helper_identity`
        adds its immutable captures and its own defaults; a plain function
        with neither gets exactly the old fingerprint.
        """
        if inspect.isfunction(v):
            return f"__cash_callable__:{self._hash_helper_identity(v)}"
        return self._fingerprint_default(v)

    def _defaults_unhashable(
        self, func_name: str, pos: tuple, kwd: dict, e: Exception, warn: bool,
    ) -> None:
        """Warn (once) that a default is unhashable; ``None`` = refuse to cache."""
        bad_type = self._first_unhashable_arg_type(pos, kwd)
        if warn:
            self._warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                bad_type,
                f"@cash.cache on {func_name}: a parameter default of type "
                f"{bad_type} could not be hashed ({type(e).__name__}), so the "
                f"function does not cache for any caller rather than risk "
                f"serving a result computed under a default that changed.",
                code="KEY-UNHASHABLE-DEFAULT",
                fix="get the value out of the signature -- build it in the body "
                    "or require it at the call site"
                    + ("." if isinstance(self._first_unhashable_arg(pos, kwd),
                                         _CODE_VALUE_TYPES)
                       else f" -- or register a hasher with "
                            f"cash.register_hasher({bad_type}, ...)."),
            )
        else:
            logger.debug("defaults hash failed for %s: %s", func_name, e)
        return None

    def _finish_defaults_fold(
        self, func: Callable, state_hash: str, digest: str,
        pos: tuple, kwd: dict, pinnable: bool,
    ) -> str:
        """Memoize *digest* when it cannot drift, then mix it into *state_hash*."""
        # Two conditions gate the memo, and both are load-bearing:
        #
        # 1. Every default must be immutable. A mutable default is shared across
        #    calls and can be mutated in place (`def f(xs=[])`), so its content
        #    must be re-read every call; a memo would pin the first call's value
        #    and hand that result back forever.
        # 2. The callee must not wrap another function. The memo is validated
        #    against `func`'s OWN containers, which say nothing about defaults
        #    reached through `__wrapped__`; re-hashing those per call keeps the
        #    validation honest rather than merely cheap.
        if (
            pinnable
            and getattr(func, "__wrapped__", None) is None
            and all(self._is_immutable_capture(v) for v in pos)
            and all(self._is_immutable_capture(v) for v in kwd.values())
        ):
            try:
                self._defaults_pins[func] = (
                    getattr(func, "__defaults__", None),
                    dict(getattr(func, "__kwdefaults__", None) or {}),
                    digest,
                )
            except TypeError:
                pass  # not weak-referenceable; recompute per call
        return hashlib.sha256(
            f"{state_hash}:defaults:{digest}".encode('utf-8')
        ).hexdigest()

    def _fold_bound_self(
        self, func: Callable, func_name: str, state_hash: str, warn: bool = True,
    ) -> str:
        """Mix a bound method's instance state into the key.

        ``c.cache(obj.method)`` wraps an already-bound method: ``self`` never
        appears in ``args``, so two instances with different state shared one
        cache key and silently returned each other's results. Fold
        ``func.__self__`` through the same machinery as an ordinary argument
        (so ``register_hasher`` applies exactly as it does for in-class
        decoration, where ``self`` arrives via ``args``). Hashed per call,
        not at decoration: instance state may change between calls.

        Unhashable instances fall back to ``id(self)`` — correct (distinct
        instances stay distinct) but process-local; a one-shot warning points
        at ``register_hasher``.
        """
        if not inspect.ismethod(func):
            return state_hash
        owner = func.__self__
        try:
            self_hash = self._hash_arg_payload((owner,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            owner_type = type(owner).__name__
            if warn:
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    owner_type,
                    f"@cash.cache on bound method {func_name}: the instance's state "
                    f"could not be hashed ({type(e).__name__}), so cash fell back "
                    f"to the instance's identity - entries are not shared across "
                    f"equal instances and do not survive the process.",
                    code="KEY-INSTANCE-STATE",
                    fix=f"register a hasher with "
                        f"cash.register_hasher({owner_type}, ...), returning "
                        f"something derived from the state that affects the "
                        f"result.",
                    # Explicit: ``_warn_once``'s default of 5 stops one frame
                    # short and blames the key-resolution code inside core.py.
                    # Measured against a known call line.,
                )
            else:
                logger.debug("bound-self hash failed for %s: %s", func_name, e)
            self_hash = f"selfid:{id(owner)}"
        return hashlib.sha256(
            f"{state_hash}:boundself:{self_hash}".encode('utf-8')
        ).hexdigest()

    @staticmethod
    def _iter_code_scopes(code: types.CodeType) -> Iterator[types.CodeType]:
        """Yield *code* and every code object nested inside it, recursively.

        A generator expression, comprehension, or ``lambda`` compiles to its
        OWN code object hung off the enclosing ``co_consts``, so anything it
        references is invisible in the outer ``co_names`` / instruction stream.
        Walking the const tree is the same trick the
        bytecode hash uses (``notebook/function_tracker.py``
        ``_update_code_object_hash``) for exactly this reason.

        Comprehensions nest, so this recurses. Note that CPython 3.12+ inlines
        list/set/dict comprehensions into the enclosing scope (PEP 709) — those
        already land in the outer ``co_names``; generator expressions and
        lambdas still get their own scope on every version.
        """
        yield code
        for const in code.co_consts or ():
            if isinstance(const, types.CodeType):
                yield from Cash._iter_code_scopes(const)

    #: Dunder globals that are machine or import machinery, never user data.
    #:
    #: Every dunder used to be skipped, which is right for these -- ``__file__``
    #: and ``__name__`` differ per checkout and per invocation, so folding them
    #: would make a cache key un-shareable between two machines and between
    #: ``python job.py`` and ``python -m job``. It is wrong for the ones a
    #: library actually declares: a round-16 tester bumped ``__version__``,
    #: watched it invalidate NOTHING, and kept publishing a report stamped with
    #: the old version through three further edits that each correctly
    #: invalidated other stages. ``RELEASE`` and ``LEVEL`` in the same file were
    #: tracked; only the dunder spelling was not.
    _MACHINERY_DUNDERS = frozenset({
        "__name__", "__file__", "__doc__", "__package__", "__loader__",
        "__spec__", "__builtins__", "__path__", "__cached__", "__debug__",
        "__annotations__", "__dict__", "__module__", "__qualname__",
    })

    def _read_global_data_names(self, func: Callable) -> tuple[str, ...]:
        """Global names *func* references that are candidates for data-folding.

        ``co_names`` intersected with the function's globals, minus the import
        machinery dunders (``_MACHINERY_DUNDERS``) and minus any global the
        function WRITES (``STORE_GLOBAL`` /
        ``DELETE_GLOBAL``). A written global is a side-effect accumulator (a
        ``global counter; counter += 1``) whose value drifts every call - folding
        it would make every call miss (the lesson, applied to globals).
        Modules / callables / classes are filtered per-call at fold time (a
        name's bound value can change). Cached per code object.

        Both bytecode-derived channels walk the NESTED scopes too:
        a global read only inside a genexp/lambda otherwise never invalidated
        (silent stale results), and — the reason the two must move together —
        the ``STORE_GLOBAL`` of a walrus accumulator inside a genexp lives in
        the genexp's own code object, so collecting nested reads without
        collecting nested writes would fold a drifting counter and miss
        forever. The in-place-mutation exclusion below needs no such change:
        it is AST-based, and ``ast.walk`` over the function's source already
        descends into comprehension and lambda bodies.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return ()
        cached = self._global_read_cache.get(code)
        if cached is not None:
            return cached
        g = getattr(func, "__globals__", {}) or {}
        import dis
        scopes = tuple(Cash._iter_code_scopes(code))
        written = {
            instr.argval for scope in scopes
            for instr in dis.get_instructions(scope)
            if instr.opname in ("STORE_GLOBAL", "DELETE_GLOBAL")
        }
        candidates = {
            n for scope in scopes for n in (scope.co_names or ())
            if n in g and n not in Cash._MACHINERY_DUNDERS and n not in written
        }
        # Also exclude globals the body mutates IN PLACE (``g['k'] += 1``,
        # ``g.append(...)``) - a STORE_GLOBAL-free accumulator that would
        # otherwise drift every call and cause a permanent miss.
        #
        # `hard` is that set: mutations visible in this function's own source.
        # `provisional` is the weaker case the argument rule used to lump in with
        # it - a name merely PASSED to a call. Those are folded (so a change
        # invalidates, CAS-270) and confirmed at runtime by
        # `_learn_mutating_captures`, which demotes any that the call is actually
        # observed to mutate.
        provisional: frozenset = frozenset()
        if candidates:
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
                hard = Cash._unsafe_uses_of(
                    tree, candidates, bare_args=False, mutating_methods_only=True,
                )
                provisional = Cash._unsafe_uses_of(tree, candidates) - hard
                candidates -= hard
            except SOURCE_RETRIEVAL_ERRORS:
                # No source: keep the old conservative answer. Without an AST
                # there is no way to tell a read from a mutation, and folding
                # blind would risk the permanent-miss trap with nothing to
                # learn from.
                candidates, provisional = set(), frozenset()
        names = tuple(sorted(candidates))
        if len(self._global_read_cache) < 4096:
            self._global_read_cache[code] = names
            # Kept in lockstep with the names cache so the two can never
            # disagree about a code object. A MISSING entry is not "nothing is
            # provisional" -- `_fold_read_globals` reads that as "watch every
            # folded name", which costs an extra hash per miss and is the safe
            # direction.
            self._provisional_global_cache[code] = provisional
        return names

    @staticmethod
    def _stabilize_for_global_hash(v: Any, hash_callable, _depth: int = 0) -> Any:
        """Rewrite *v* so callables (incl. lambdas held in containers) are
        replaced by their source hash, making a container of callables hashable
        and content-sensitive (dict-dispatch channel)."""
        if _depth > 8:
            return v
        if callable(v) and not isinstance(v, type):
            try:
                return ("__cash_callable__", hash_callable(v))
            except (OSError, TypeError, ValueError):
                return ("__cash_callable__", getattr(v, "__qualname__", repr(v)))
        if isinstance(v, dict):
            return {
                k: Cash._stabilize_for_global_hash(val, hash_callable, _depth + 1)
                for k, val in v.items()
            }
        if isinstance(v, (list, tuple)):
            return type(v)(
                Cash._stabilize_for_global_hash(x, hash_callable, _depth + 1) for x in v
            )
        return v

    def _code_identity(self, fn: Any) -> tuple:
        """The bytecode-level identity of a callable, or ``()`` if it has none.

        Bytecode rather than source because a class defined in a notebook cell
        has no retrievable source at all: ``inspect.getsource`` resolves a class
        through ``sys.modules[cls.__module__].__file__``, and a notebook
        ``__main__`` has none. A function escapes this via ``co_filename``,
        which is why ``_hash_callable_source`` works for helpers and not here.

        Comments and formatting are absent from bytecode, so they do not
        invalidate -- strictly better than source hashing. Docstrings live in
        ``co_consts`` and do.

        Instance method (not static) because defaults/kwdefaults go through
        ``_value_identity`` -> ``self._hash_arg_payload``: a default like
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
        every process. Measured: the same class's digest differed between two
        freshly started processes whenever a method held one of these,
        permanently missing the cache cross-process for any user class with a
        comprehension in a method. A nested code object carries no defaults of
        its own (those belong to the FUNCTION eventually built from it, not to
        the raw code object), so only co_code/co_consts/co_names apply here.

        The SAME disease reaches a plain (non-code) const too: ``x in
        {'alpha', 'beta'}`` compiles a ``frozenset`` straight into
        ``co_consts``, and ``repr()`` of a set/frozenset follows the table's
        internal (hash-order-dependent) iteration -- under Python's default
        per-process string-hash randomization, measured 2 distinct orderings
        across repeated fresh processes for a 2-element set. ``_value_identity``
        folds CONTENT instead, which is order-independent for a set/frozenset.
        """
        return (
            code.co_code,
            tuple(
                self._code_object_identity(k) if isinstance(k, types.CodeType)
                else self._value_identity(k)
                for k in code.co_consts
            ),
            tuple(code.co_names),
        )

    def _value_identity(self, v: Any) -> str:
        """Address-free identity for a value that is not itself a code object.

        ``_hash_arg_payload`` folds CONTENT and is the established,
        address-free tool used throughout this file for exactly this. What it
        cannot pickle used to fall back to ``repr()`` -- and ``repr()`` is a
        memory ADDRESS for the most ordinary unpicklable default there is.
        Measured in three fresh processes, ``def m(self, key=lambda r: r)``
        produced three different class digests, as did ``lock=threading.
        Lock()`` and the keyword-only spelling; the entry could therefore
        never hit again after a kernel restart, silently and permanently.

        ``_class_surface_parts`` already refuses a ``repr()`` fallback, on the
        grounds that it "would reintroduce the address leak this member-content
        fold exists to avoid". This makes the two agree.
        """
        try:
            return self._hash_arg_payload((v,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            return self._unpicklable_identity(v)

    def _unpicklable_identity(self, v: Any, _depth: int = 0) -> str:
        """Process-stable stand-in for a value ``_hash_arg_payload`` refused.

        Recursive because ``__defaults__`` is hashed as a WHOLE TUPLE: one
        unpicklable element poisons the entire tuple, so every element that
        CAN be folded still is, and only the residue is approximated.

        A lambda, function or class is approximated by its CODE SURFACE, which
        is strictly better than the old ``repr()`` on both counts -- stable
        across processes, and sensitive to an edit of the lambda's body, which
        an address never was.

        Everything else keeps its ``repr()`` UNLESS that repr carries a memory
        address. Only an address-bearing repr is the thing this method exists
        to remove; a value-based ``__repr__`` -- ``Config(n=1)`` -- is
        deterministic across processes and carries real information, and
        discarding it was measured to serve a stale result when the value
        changed. Collapsing to a type name is the last resort, for the case
        where the only thing distinguishing two objects was an address that
        changed every process: noise, never signal.
        """
        if _depth > 4:
            return "<deep>"
        if isinstance(v, (list, tuple, set, frozenset)):
            inner = [self._unpicklable_identity(x, _depth + 1) for x in v]
            if isinstance(v, (set, frozenset)):
                # Set iteration order follows the hash table, and string
                # hashing is randomized per process -- sort or reintroduce the
                # very instability this method exists to remove.
                inner.sort()
            return f"{type(v).__qualname__}[{'|'.join(inner)}]"
        if isinstance(v, dict):
            return "dict[" + "|".join(sorted(
                f"{self._unpicklable_identity(k, _depth + 1)}"
                f"={self._unpicklable_identity(val, _depth + 1)}"
                for k, val in v.items()
            )) + "]"
        try:
            return self._hash_arg_payload((v,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            pass
        surface = self._code_surface_hash(v)
        if surface is not None:
            return f"code:{surface}"
        try:
            text = repr(v)
        except Exception as e:  # noqa: BLE001 - a __repr__ may raise
            logger.debug("[CORE] repr() failed while identifying %s: %s", type(v), e)
            text = ""
        # ``0x`` is how CPython renders the address in every default repr
        # (``<object object at 0x...>``, ``<function <lambda> at 0x...>``,
        # ``functools.partial(<function f at 0x...>, 3)``), so its presence is
        # the test for "this repr is not reproducible".
        #
        # KNOWN RESIDUAL, and it is the UNSAFE direction -- do not read the
        # collapse below as conservative. A value-based repr that happens to
        # carry a hex literal (``Config(mask=0xff)``) is collapsed too, so
        # editing that value does NOT invalidate: measured, such a default
        # serves a STALE result. Accepted because the shape is narrow, not
        # because it is safe. Widening the test (e.g. ``0x`` only when preceded
        # by ``at ``) would shrink it further.
        if text and "0x" not in text:
            return text
        cls = type(v)
        return f"<{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', '?')}>"

    def _code_surface_own(self, obj: Any) -> str | None:
        """A digest of the user code *obj* itself carries, or ``None``.

        Its OWN surface only -- code merely referenced by that code is folded
        by :meth:`_code_surface_hash`, which combines these per-object digests.
        The split is what keeps the memo below honest: memoizing a digest that
        included a referenced class would serve a stale answer when only that
        OTHER class is redefined (a notebook cell re-run), because *obj* is
        still the same object and still hits its memo entry.

        ``None`` means "cannot determine" and the caller must fall back to
        today's by-reference key. This never raises: a hashing failure must not
        break a cached call.

        Memoized on the object itself, following ``_user_class_src_cache``:
        redefining a class produces a NEW object and therefore a distinct dict
        key, so the memo cannot serve a stale entry. Keying on ``id()`` would
        be a correctness bug, since CPython recycles addresses.
        """
        # A `functools.partial` is its function plus arguments. The arguments
        # already reach the key (a partial pickles them, by value); its code is
        # the wrapped function's, which pickle names only by reference -- so an
        # edit to that function's body kept the key, and the partial was
        # reported as uncomputable code instead (KEY-OPAQUE-CALLABLE).
        depth = 0
        while isinstance(obj, functools.partial) and depth < 8:
            obj = obj.func
            depth += 1
        try:
            # Dispatch FIRST, memo read second. Every argument to a cached
            # function passes through here (Task 4), and most are not a
            # class or callable at all -- a list, dict, set, numpy array,
            # DataFrame. Checking the dispatch before touching the memo means
            # those return None from a plain isinstance()/callable() check
            # (neither raises) instead of reaching a dict.get() that would
            # raise TypeError and get caught, on the common case rather than
            # the exception.
            is_type = isinstance(obj, type)
            if not (is_type or callable(obj)):
                return None
            if not self._is_user_code_object(obj):
                return None
            # The memo read must still be INSIDE the try: by this point *obj*
            # is a class or a callable, and while both are hashable in the
            # overwhelming common case, neither is guaranteed to be (a
            # __call__-implementing instance can set __hash__ = None) -- this
            # must not be the one path in this method that can still raise.
            cached = self._code_surface_cache.get(obj)
            if cached is not None:
                return cached
            if is_type:
                parts = self._class_surface_parts(obj)
            else:
                ident = self._code_identity(obj)
                if not ident:
                    return None
                parts = [(getattr(obj, "__qualname__", "?"), "", ident)]
            if not parts:
                return None
            digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
        except Exception as e:  # noqa: BLE001 - hashing must never break a call
            logger.debug("[CORE] code-surface hash failed for %r: %s", obj, e)
            return None
        try:
            if len(self._code_surface_cache) < 4096:
                self._code_surface_cache[obj] = digest
        except TypeError:
            pass  # unhashable object - skip the memo, keep the answer
        return digest

    # Bounds on the reference walk. Depth 4 and 64 targets are far past any
    # real object graph; they exist so a pathological one degrades into a
    # coarser digest rather than a hang. Exceeding them can only UNDER-fold,
    # which is the pre-existing behaviour, never a wrong-but-confident answer.
    _MAX_CODE_REF_DEPTH = 4
    _MAX_CODE_REF_TARGETS = 64

    @staticmethod
    def _walk_nested_code(code: types.CodeType, glb: dict, _depth: int = 0):
        """Yield *code* and the code objects nested in its constants.

        A comprehension, a lambda, or a nested ``def`` compiles to its own
        code object stored in ``co_consts``; the names IT references do not
        appear in the parent's ``co_names``. ``field(default_factory=lambda:
        B(0))`` is exactly that shape -- ``B`` is reachable only through the
        lambda -- so a walk that stopped at the top level would miss the case
        this exists for.
        """
        yield code, glb
        if _depth >= Cash._MAX_CODE_REF_DEPTH:
            return
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                yield from Cash._walk_nested_code(const, glb, _depth + 1)

    def _iter_code_and_globals(self, obj: Any):
        """Yield ``(code object, globals)`` for the code *obj* carries."""
        carriers: list[Any] = []
        if isinstance(obj, type):
            for base in obj.__mro__:
                if base is object or self._is_opaque(base):
                    continue
                if not self._is_user_code_object(base):
                    continue
                carriers.extend(vars(base).values())
                # Same blind spot as _class_surface_parts: a field declaring
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
                accessor = getattr(accessor, "__wrapped__", accessor)
                code = getattr(accessor, "__code__", None)
                glb = getattr(accessor, "__globals__", None)
                if isinstance(code, types.CodeType) and isinstance(glb, dict):
                    yield from self._walk_nested_code(code, glb)

    def _code_ref_targets(self, obj: Any) -> list[Any]:
        """User-code objects that *obj*'s code references by global name.

        Resolution happens on every call, deliberately. Only the (code,
        globals) pairs are memoized -- those are fixed for as long as *obj*
        exists -- because what a NAME is bound to can change underneath us,
        and that change is precisely what must invalidate.

        Names come from ``co_names``, i.e. what the code actually LOADS.
        Annotations are not consulted: ``__annotations__`` holds ``'B'`` as a
        string for a hint that never runs, so following them would invalidate
        on a change that cannot alter any result.
        """
        pairs = None
        try:
            pairs = self._code_refs_cache.get(obj)
        except TypeError:
            pass  # unhashable - recompute each time rather than fail
        if pairs is None:
            pairs = tuple(self._iter_code_and_globals(obj))
            try:
                if len(self._code_refs_cache) < 4096:
                    self._code_refs_cache[obj] = pairs
            except TypeError:
                pass

        targets: list[Any] = []
        seen_names: set[str] = set()
        for code, glb in pairs:
            for name in code.co_names:
                if name in seen_names:
                    continue
                seen_names.add(name)
                value = glb.get(name)
                if value is None or value is obj:
                    continue
                if not (isinstance(value, type) or callable(value)):
                    continue
                try:
                    if self._is_opaque(value) or not self._is_user_code_object(value):
                        continue
                except Exception:  # noqa: BLE001 - never break a call
                    continue
                targets.append(value)
        return targets

    def _code_surface_hash(self, obj: Any) -> str | None:
        """A digest of *obj*'s code AND the user code that code reaches.

        Folding only what an argument or global directly carries left a real
        hole: a class whose field factory constructs another class changes
        behaviour when THAT class is edited, and nothing in the first class's
        own surface moves. Measured -- the cache returned
        ``A(value=B(value=10))`` where a fresh call produced
        ``A(value=B(value=1000))``, a wrong answer rather than a stale one.

        Reachability is STATIC: names the code loads from its globals,
        transitively, bounded. Code selected at runtime (a class picked out of
        a dict) still cannot be followed, so this narrows the gap rather than
        closing it.
        """
        own = self._code_surface_own(obj)
        if own is None:
            return None
        try:
            reached = self._code_ref_closure(obj)
        except Exception as e:  # noqa: BLE001 - hashing must never break a call
            logger.debug("[CORE] code-ref closure failed for %r: %s", obj, e)
            return own
        if not reached:
            return own
        return hashlib.sha256(
            ":".join([own, *sorted(reached)]).encode("utf-8")
        ).hexdigest()

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
        depth = 0
        while frontier and depth < self._MAX_CODE_REF_DEPTH:
            following: list[Any] = []
            for source in frontier:
                for target in self._code_ref_targets(source):
                    if id(target) in seen:
                        continue
                    seen.add(id(target))
                    keep.append(target)
                    digest = self._code_surface_own(target)
                    if digest is not None:
                        digests.append(
                            f"{getattr(target, '__qualname__', '?')}:{digest}"
                        )
                    following.append(target)
                    if len(digests) >= self._MAX_CODE_REF_TARGETS:
                        return digests
            frontier = following
            depth += 1
        return digests

    def _dataclass_field_parts(self, base: type, field_map: dict) -> list[tuple]:
        """Fold a dataclass's field metadata, which nothing else reaches.

        ``@dataclass`` moves the per-field declaration off the class attribute
        and into ``__dataclass_fields__``. The attribute that remains is just
        the default value, so a field's TYPE and its ``metadata=`` never
        reached the digest -- and ``__dataclass_fields__`` itself cannot be
        pickled, because ``Field.metadata`` is a ``mappingproxy``. The generic
        fold below caught that ``TypeError``, set ``content = None``, and
        dropped the member in silence.

        Measured: a schema class with ``field(metadata={"desc": ...})``, passed
        as an argument, served an answer built from the OLD description after
        that description was rewritten. Which is the whole hazard, because a
        field description is prompt text in every structured-output library
        there is -- it is not decoration, it is the instruction.

        Only the parts nothing else covers are folded. The default value is
        already the class attribute and is hashed there; folding it again would
        change nothing and cost a hash.
        """
        parts: list[tuple] = []
        for fname, f in sorted(field_map.items()):
            try:
                meta = self._hash_arg_payload((dict(getattr(f, "metadata", {}) or {}),), {})
            except (TypeError, pickle.PicklingError, AttributeError,
                    OverflowError, ValueError):
                # An unpicklable metadata VALUE. Skip the metadata rather than
                # repr() it: a repr here would leak an object address and make
                # the digest differ between processes, which is worse than not
                # tracking it.
                meta = None
            # `str(f.type)` and not the object: an annotation may be a string
            # (`from __future__ import annotations`) or a class, and both spell
            # the same thing deterministically this way.
            parts.append((base.__qualname__, f"__dataclass_field__:{fname}",
                          (str(getattr(f, "type", "")), meta)))
        return parts

    def _pydantic_field_parts(self, cls: type) -> list[tuple]:
        """Fold a pydantic model's field declarations.

        The counterpart to :meth:`_dataclass_field_parts`. Pydantic keeps them
        in ``model_fields`` -- a property on the class, so ``vars(cls)`` never
        sees it -- and the only other place they appear is the compiled trio
        skipped above, whose digest is different in every process.

        So before this, a pydantic spec passed as an argument was tracked
        (through the compiled schema) but never cached across processes:
        measured 2 executions for 2 runs of an unedited model, against 1 for
        the equivalent dataclass. Safe and useless.

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
                default = self._hash_arg_payload((getattr(info, "default", None),), {})
            except (TypeError, pickle.PicklingError, AttributeError,
                    OverflowError, ValueError):
                default = None
            parts.append((cls.__qualname__, f"__pydantic_field__:{fname}", (
                str(getattr(info, "annotation", "")),
                getattr(info, "description", None),
                getattr(info, "alias", None),
                default,
            )))
        return parts

    def _class_surface_parts(self, cls: type, _depth: int = 0) -> list[tuple]:
        """Every user-code member of *cls* and its user base classes.

        Walked in reverse MRO so a subclass override lands after the base it
        replaces, and sorted within each class so dict ordering cannot change
        the digest.
        """
        parts: list[tuple] = self._pydantic_field_parts(cls)
        for base in reversed(cls.__mro__):
            # An OPAQUE base contributes nothing, so `mark_opaque(VendorBase)`
            # also stops a `Derived(VendorBase)` digest moving when the vendor
            # edits its own base. Without this the escape hatch worked only
            # when the opaque class was the one passed, which is not how
            # `docs/decorator.md` advertises it. Per-base and exact-match, so
            # opacity still does not inherit: marking a base does not make
            # `Derived` opaque, it only drops that base's own members.
            if base is object or self._is_opaque(base):
                continue
            if not self._is_user_code_object(base):
                continue
            for name, member in sorted(vars(base).items(), key=lambda kv: kv[0]):
                # __firstlineno__ (class attribute since Python 3.13, absent on
                # 3.10/3.11) records the class's first source line, which shifts
                # when a comment or blank line is added above it -- with no
                # code change at all. Skipping it is what keeps "comments do
                # not invalidate" (see _code_identity) true on 3.13+ too.
                if name in ("__dict__", "__weakref__", "__module__", "__firstlineno__"):
                    continue
                # Pydantic v2 compiles three Rust objects onto every model.
                # They are DERIVED from the field declarations, and their
                # digest differs in every process -- measured: the same
                # unedited model produced a different class digest on each
                # run, so a pydantic spec passed as an argument never hit
                # across processes. `model_fields` below carries the same
                # declarations and is stable, so this loses nothing.
                if name in _PYDANTIC_COMPILED:
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
                # (it exposes the underlying function as .func instead), so it
                # fell through to the data-attribute branch below and hashed
                # an unchanging descriptor repr. Measured: editing any of
                # these three wrapped method bodies left the digest unchanged
                # without this step.
                target = getattr(target, "__wrapped__", target)
                if not hasattr(target, "__code__"):
                    func_attr = getattr(target, "func", None)
                    if func_attr is not None and hasattr(func_attr, "__code__"):
                        target = func_attr
                ident = self._code_identity(target)
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
                                own = self._hash_arg_payload((member,), {})
                            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                                own = None
                            if own is not None:
                                parts.append((base.__qualname__, name, (ident, own)))
                                continue
                        parts.append((base.__qualname__, name, ident))
                        continue
                    # A callable member with NO reachable ``__code__``: a
                    # nested class (``class Outer: inner = Inner``), a
                    # ``functools.partial``, a callable instance. Dropping it
                    # meant the non-callable branch below folded content for
                    # an ordinary member while these three folded nothing at
                    # all -- measured: editing ``Inner.f`` left ``Outer``'s
                    # digest unchanged, and ``partial(scale, 3)`` collided
                    # with ``partial(scale, 4)``.
                    #
                    # The nested walk recurses into ``_class_surface_parts``
                    # DIRECTLY, not through the memoized ``_code_surface_hash``,
                    # and is bounded by DEPTH rather than by a cycle set. A
                    # cycle set would make the digest depend on which class
                    # happened to be hashed first (the memo would hold a cut
                    # result for one order and a full one for the other) --
                    # reintroducing exactly the cross-process instability
                    # ``_value_identity`` was just fixed for. A depth bound
                    # gives every process the same answer regardless of order.
                    nested = None
                    inner_cls = member if isinstance(member, type) else type(member)
                    if _depth < 2 and self._is_user_code_object(inner_cls):
                        sub_parts = self._class_surface_parts(inner_cls, _depth + 1)
                        if sub_parts:
                            nested = hashlib.sha256(
                                repr(sub_parts).encode("utf-8"),
                            ).hexdigest()
                    try:
                        content = self._hash_arg_payload((member,), {})
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
                    # just because that lookup succeeded (measured: a
                    # partialmethod's bound-argument edit, and an unrelated
                    # object's own attribute edit, both went undetected when
                    # `ident` alone short-circuited this). Fold `ident` TOO
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
                    # exists to avoid (a class attribute is exactly what
                    # Blocker 2 measured repr() leaking on). If content can't
                    # be folded and no `ident` was found either, dropping the
                    # member is strictly safer than a non-deterministic repr.
                    try:
                        content = self._hash_arg_payload((member,), {})
                    except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                        content = None
                    if ident or content is not None:
                        parts.append((base.__qualname__, name, (ident, content)))

        # Dataclass field factories. ``dataclasses`` DELETES the class
        # attribute when a field declares ``default_factory``, so the
        # ``vars()`` walk above cannot see it -- ``getattr_static`` raises
        # AttributeError for that name. The factory is nonetheless code that
        # decides what every instance holds: measured, editing
        # ``field(default_factory=lambda: B(0))`` to ``B(999)`` changed what
        # ``A()`` produced while leaving this digest byte-identical.
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

    def _user_class_source_hash(self, cls: type) -> str:
        """Memoized source hash of a USER class.

        A class's source cannot change within a running interpreter: editing the
        file and re-importing produces a NEW class object (a distinct dict key),
        so the hash is computed once per class object and reused on every
        subsequent call. The per-call cost of the instance channel below is then
        a cheap object-graph walk plus dict lookups -- never source I/O.

        Source-first, surface-as-fallback. Both of this method's callers
        (``_instance_class_source_parts``, directly and via
        ``_fold_read_globals``) gate on ``_is_user_class`` -> ``_is_user_module``,
        which requires ``__file__`` -- so every class actually reachable here
        already has retrievable source, and ``inspect.getsource`` succeeds. The
        class-aware surface (``_code_surface_hash``) only engages on
        ``SOURCE_RETRIEVAL_ERRORS`` -- a class truly without source, e.g. a
        notebook cell's ``__main__`` has no ``__file__`` -- or when this method
        is reached some other way in the future. Preferring it unconditionally
        was measured to regress every file-backed class whose method is wrapped
        by ``@functools.wraps``, ``@lru_cache``, or ``@singledispatchmethod``:
        ``_class_surface_parts`` walks the WRAPPER, not the wrapped function, so
        a body edit under one of those decorators stopped invalidating even
        though whole-class source hashing always saw it (source is just text).
        """
        cached = self._user_class_src_cache.get(cls)
        if cached is not None:
            return cached
        try:
            src = inspect.getsource(cls)
            h = hashlib.sha256(src.encode("utf-8")).hexdigest()
        except SOURCE_RETRIEVAL_ERRORS:
            # No source to hash (or it doesn't parse). _hash_callable_source
            # has no class branch of its own: a class has no __code__, so ITS
            # fallback chain falls through to type(fn).__qualname__ --
            # literally "type" for EVERY class, colliding all source-less
            # classes onto one hash. Try the class-aware surface first.
            h = self._code_surface_hash(cls) or self._hash_callable_source(cls)
        if len(self._user_class_src_cache) < 4096:
            self._user_class_src_cache[cls] = h
        return h

    @staticmethod
    def _iter_contained(obj: Any):
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

    def _instance_class_source_parts(
        self, value: Any, _seen: set | None = None, _depth: int = 0,
        own_pkg: str | None = None,
    ) -> list[tuple[str, str]]:
        """``(qualname, source-hash)`` for the user classes behind an INSTANCE.

        A cached function that reads a pre-built module-level object -- ``pre =
        MyTransformer()`` imported and dropped into a pipeline -- had that object
        only VALUE-hashed: its ``__dict__`` pickle carries no method source, so an
        edit to ``MyTransformer.transform`` left the key unchanged and served a
        stale result (found replaying a real repo's git history). Fold the source
        of the instance's class -- and, bounded, of the user-class instances it
        holds -- so a method-body edit invalidates.

        The walk recurses only into user-class instances: a third-party object
        (a fitted sklearn estimator, a numpy array) is not user-editable and its
        internals must not churn the key, and stopping there also bounds the cost
        on real pipelines. Consequence (documented limitation): a user class
        reachable only through a third-party container is not folded here.
        """
        if _depth > 4:
            return []
        if _seen is None:
            _seen = set()
        if id(value) in _seen:
            return []
        _seen.add(id(value))
        parts: list[tuple[str, str]] = []
        cls = type(value)
        if self._is_user_class(cls, own_pkg):
            try:
                parts.append((cls.__qualname__, self._user_class_source_hash(cls)))
            except SOURCE_RETRIEVAL_ERRORS:
                pass
        held = getattr(value, "__dict__", None)
        if isinstance(held, dict):
            for attr_val in held.values():
                for item in self._iter_contained(attr_val):
                    if self._is_user_class(type(item), own_pkg):
                        parts.extend(
                            self._instance_class_source_parts(
                                item, _seen, _depth + 1, own_pkg=own_pkg)
                        )
        return parts

    def _fold_read_globals(
        self,
        func: Callable,
        func_name: str,
        state_hash: str,
        owner_code: Any = None,
        seen: set | None = None,
        extra_names: tuple[str, ...] = (),
    ) -> str:
        """Fold module-level DATA globals the function reads into the key.

        ``owner_code`` is the code object the DRIFT GUARD is recorded under.
        It defaults to *func*'s own, which is right when *func* is the cached
        function. When folding a HELPER's globals it must be the CACHED
        function's code instead: `_learn_mutating_captures` records drift
        against the function whose call was observed, so looking it up under
        the helper's code would never find the entry, fold a drifting
        accumulator anyway, and miss forever.

        A cached function reading a mutable module global (a config constant, a
        dispatch dict of callables) returned stale results when that global
        changed, with no warning. Fold the content of read *data* globals so a
        change invalidates. Modules, plain callables (helpers - tracked via the
        purity analyzer / dependency graph), and classes are excluded; unhashable
        data globals warn once and are skipped.
        """
        names = self._read_global_data_names(func)
        if extra_names:
            names = tuple(dict.fromkeys(names + extra_names))
        g = getattr(func, "__globals__", None)
        if not isinstance(g, dict):
            return state_hash
        # NOTE: no early return on an empty ``names``. A body whose only global
        # reads are module attributes (``return conf.RATE``) has NO plain data
        # globals, so bailing here skipped the module-attribute channel in
        # exactly the case it exists for.
        parts: list[tuple[str, str]] = []
        own_pkg = self._own_package(func)
        code = getattr(func, "__code__", None)
        # A missing provisional entry means "unknown", not "none" -- watch every
        # folded name rather than fold one blind (see `_read_global_data_names`).
        provisional = self._provisional_global_cache.get(code)
        learned_mutating = self._mutating_globals.get(
            (owner_code if owner_code is not None else code, "global"), frozenset()
        )
        watch: dict[str, str] = {}
        for name in names:
            if name not in g:
                continue
            if seen is not None:
                pair = (id(g), name)
                if pair in seen:
                    continue
                seen.add(pair)
            if name in learned_mutating:
                # Observed to drift as a result of calling this function. Folding
                # it would key the entry on the function's own output and miss
                # forever, which is the trap the old argument rule guarded --
                # and the decorator has no perpetual-miss guard to catch it.
                continue
            v = g[name]
            # Skip modules, classes, and plain callables (helpers/deps handled
            # elsewhere). Containers of callables (dispatch dicts) ARE folded.
            if isinstance(v, types.ModuleType) or isinstance(v, type):
                continue
            if callable(v) and not isinstance(v, (dict, list, tuple, set)):
                continue
            try:
                stabilized = self._stabilize_for_global_hash(v, self._hash_callable_source)
                h = self._hash_arg_payload((stabilized,), {})
                parts.append((name, h))
                # Free: this is the hash the key already needed. Keeping it is
                # what makes the post-call check cost one hash instead of two.
                if provisional is None or name in provisional:
                    # `g`, not the decorated function's globals: this may be a
                    # helper's module (see `_fold_helper_read_globals`).
                    watch[name] = (h, "global", g)
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
                self._warn_once(
                    CashImpurityWarning,
                    func_name,
                    name,
                    f"@cash.cache on {func_name}: reads module global '{name}' whose "
                    f"value could not be hashed, so changes to it will NOT "
                    f"invalidate the cache.",
                    code="KEY-UNHASHABLE-GLOBAL",
                    fix=_UNHASHABLE_GLOBAL_FIX,
                )
                continue
            # A pre-built user-class INSTANCE (or a container of them) is only
            # value-hashed above -- its class's method SOURCE is invisible to the
            # pickle, so editing a method served stale. Fold the class-graph
            # source too (memoized per class; see _instance_class_source_parts).
            for item in self._iter_contained(v):
                if self._is_user_class(type(item), own_pkg):
                    for cname, chash in self._instance_class_source_parts(
                            item, own_pkg=own_pkg):
                        parts.append((f"{name}#cls:{cname}", chash))
        self._pending_capture_watch.update(watch)
        parts.extend(self._module_attr_parts(func, func_name, g))
        if not parts:
            return state_hash
        payload = ":".join(f"{n}={h}" for n, h in sorted(parts))
        return hashlib.sha256(f"{state_hash}:globals:{payload}".encode('utf-8')).hexdigest()

    def _fold_helper_read_globals(
        self, func: Callable, func_name: str, state_hash: str
    ) -> str:
        """Fold globals the transitive HELPERS read, not just *func*'s own.

        A helper reading module state made its caller stale with no warning::

            THRESHOLD = 5
            def helper(): return THRESHOLD

            @cash.cache
            def via_helper(n): return helper()   # THRESHOLD=6 -> HIT -> 5

        The helper's SOURCE was folded, so editing its body invalidated; the
        DATA it read was invisible. Same fold as the one-level case, applied
        to the helpers the purity analyzer already walks, so it inherits the
        written-global exclusion and the drift guard rather than re-deriving
        them.

        Helpers are re-resolved from ``sys.modules`` per call, matching
        ``SysModulesHelperResolver``: a redefined helper reads the redefined
        module's globals.
        """
        report = self._purity_reports.get(func_name)
        if report is None or not (report.helper_resolution_paths or report.helper_objects):
            return state_hash
        owner_code = getattr(func, "__code__", None)
        # Pre-seed with what the cached function itself already folded, so a
        # global it reads directly is not folded a second time on behalf of a
        # helper that also reads it.
        seen: set = set()
        own_globals = getattr(func, "__globals__", None)
        if isinstance(own_globals, dict):
            for name in self._read_global_data_names(func):
                seen.add((id(own_globals), name))
        return self._fold_paths_read_globals(
            report, func, func_name, state_hash, owner_code=owner_code, seen=seen,
        )

    def _fold_paths_read_globals(
        self,
        report: PurityReport,
        func: Callable,
        func_name: str,
        state_hash: str,
        *,
        owner_code: Any,
        seen: set,
    ) -> str:
        """Fold the read globals of every helper *report* resolved.

        Shared by the two callers that need it: a cached function's own helpers
        and the helpers of the cached functions it calls.
        """
        for qual in sorted(report.helper_resolution_paths):
            module_name, attr_chain = report.helper_resolution_paths[qual]
            target: Any = sys.modules.get(module_name)
            if target is None:
                continue
            for attr in attr_chain:
                target = getattr(target, attr, None)
                if target is None:
                    break
            if target is None or target is func or not callable(target):
                continue
            if getattr(target, "__globals__", None) is None:
                continue
            state_hash = self._fold_read_globals(
                target, func_name, state_hash, owner_code=owner_code, seen=seen
            )
        # A callable bound at a call site carries DATA besides its code: a
        # partial's arguments, a bound method's instance, a callable
        # instance's attributes. Its code is followed as a helper; this is the
        # rest (round 18: `F = partial(base, k=2)` -> `k=3`, and `F = S(2).f`,
        # were both served stale).
        carried: list[str] = []
        for module_name, chain, _ref in report.helper_bindings:
            live = resolve_binding(module_name, chain)
            if live is func:
                continue
            digest = self._carried_state_digest(live)
            if digest is not None:
                carried.append(f"{module_name}.{'.'.join(chain)}={digest}")
        if carried:
            state_hash = hashlib.sha256(
                f"{state_hash}:carried:{':'.join(sorted(carried))}".encode("utf-8")
            ).hexdigest()
        # Helpers with no path of their own -- the function inside a decorator,
        # a closure from a factory -- are held by reference. What THEY read
        # counts as much: `@add1 def h(x): return x * K` computes with K, and
        # the wrapper bound to the name `h` never mentions it.
        for qual in sorted(report.helper_objects):
            if qual in report.helper_resolution_paths:
                continue
            target = report.helper_objects[qual]()
            if target is None or target is func or getattr(target, "__globals__", None) is None:
                continue
            state_hash = self._fold_read_globals(
                target, func_name, state_hash, owner_code=owner_code, seen=seen,
                extra_names=self._decorator_global_names(target),
            )
        return state_hash

    def _carried_state_digest(self, value: Any) -> str | None:
        """Digest of the data a callable carries besides its code, or None.

        A partial's arguments; a bound method's instance (any class -- the
        same as that instance read as a global); a callable instance's own
        state, for USER classes only: a library's callable instance
        (`np.vectorize`) keeps lazy caches that change after its first call,
        which would make every call miss. Silent on failure: the code is
        still keyed, and a warning here would fire on every class-based
        decorator whose state is just the function it wraps.
        """
        if isinstance(value, functools.partial):
            payload: Any = (value.args, dict(value.keywords))
        elif isinstance(value, types.MethodType):
            owner = value.__self__
            if isinstance(owner, (type, types.ModuleType)):
                return None
            payload = owner
        elif (callable(value) and not isinstance(
                value, (types.FunctionType, types.BuiltinFunctionType, type, types.ModuleType))
              and not is_mock(value)
              and self._is_user_class(type(value), self._own_package(type(value)))):
            payload = value
        else:
            return None
        try:
            stabilized = self._stabilize_for_global_hash(payload, self._hash_callable_source)
            return self._hash_arg_payload((stabilized,), {})
        except Exception:  # noqa: BLE001 - never break a call over this
            return None

    def _decorator_global_names(self, fn: Callable) -> tuple[str, ...]:
        """Names the decorator expressions on *fn*'s ``def`` read from its module.

        ``@np.vectorize(otypes=OT)`` is evaluated once, at import, from the
        module's ``OT`` -- a name the function's body never mentions, so
        changing it changed nothing the key could see. Only the function a
        decorator wraps has these lines (its source starts at the first
        decorator); the values are folded like any other read global, so
        modules and callables among them are skipped there. Cached per code.
        """
        code = getattr(fn, "__code__", None)
        if code is None:
            return ()
        cached = self._decorator_names_cache.get(code)
        if cached is not None:
            return cached
        names: tuple[str, ...] = ()
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(code)))
            node = next((n for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            if node is not None and node.decorator_list:
                names = tuple(dict.fromkeys(
                    n.id for deco in node.decorator_list for n in ast.walk(deco)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                ))
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
            names = ()
        if len(self._decorator_names_cache) >= 4096:
            self._decorator_names_cache.clear()
        self._decorator_names_cache[code] = names
        return names

    def _fold_dependency_read_globals(
        self, func: Callable, func_name: str, state_hash: str
    ) -> str:
        """Fold the globals the CACHED functions this one calls read.

        The third of the three channels a global can reach a key through, and
        the one that was missing. A cached function's own globals are folded by
        ``_fold_read_globals``; its plain helpers' by
        ``_fold_helper_read_globals``; a cached CALLEE's were folded into that
        callee's key and nowhere else::

            # rules.py
            THRESHOLD = 20
            def keep(x): return (x % 100) < THRESHOLD

            # calc.py
            @cash.cache
            def inner(n): return sum(i for i in range(n) if keep(i))

            @cash.cache
            def outer(n): return inner(n)

        Edit THRESHOLD and ``inner`` recomputes -- its key moved -- while
        ``outer`` returns the answer computed under the old value, with zero
        executions and no warning. One process disagreeing with itself, which
        is what a round-15 tester reported after building exactly this shape as
        a library (config module, io module, build module).

        The SOURCE side of the same edge already worked: editing ``inner``'s
        body, or ``keep``'s, invalidates ``outer`` through the graph and the
        transitive helper hashes. Only the DATA those functions read was
        invisible.

        Transitive, because the chain is: ``outer`` -> ``inner`` -> a helper in
        a third module reading a global in a fourth. Each dependency's own
        helpers go through the same fold as if they were this function's.
        """
        owner_code = getattr(func, "__code__", None)
        # Pre-seed with what this function already folded for itself, so a
        # shared global is hashed once rather than once per reader.
        seen: set = set()
        own_globals = getattr(func, "__globals__", None)
        if isinstance(own_globals, dict):
            for name in self._read_global_data_names(func):
                seen.add((id(own_globals), name))
        visited = {func_name}
        stack = sorted(self.graph.get_dependencies(func_name))
        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)
            stack.extend(sorted(self.graph.get_dependencies(dep)))
            dep_func = self.functions.get(dep)
            if dep_func is None or getattr(dep_func, "__globals__", None) is None:
                continue
            state_hash = self._fold_read_globals(
                dep_func, func_name, state_hash, owner_code=owner_code, seen=seen,
            )
            dep_report = self._purity_reports.get(dep)
            if dep_report is not None and (dep_report.helper_resolution_paths or dep_report.helper_objects):
                state_hash = self._fold_paths_read_globals(
                    dep_report, func, func_name, state_hash,
                    owner_code=owner_code, seen=seen,
                )
        return state_hash

    @staticmethod
    def _own_package(func: Any) -> str | None:
        """The top-level package of the module that defines *func*."""
        top = (getattr(func, "__module__", None) or "").split(".")[0]
        return top or None

    @staticmethod
    def _in_own_package(module_name: str | None, own_pkg: str | None) -> bool:
        """Is *module_name* inside *own_pkg* (the cached function's package)?

        ``__main__`` never counts: a script is not a package, and everything
        it imports is judged on its own merits.
        """
        if not module_name or not own_pkg or own_pkg == "__main__":
            return False
        return module_name == own_pkg or module_name.startswith(own_pkg + ".")

    @staticmethod
    def _is_user_class(cls: Any, own_pkg: str | None = None) -> bool:
        """True for a class defined in user code (not stdlib / third-party).

        Used to fold ``ClassName.CONSTANT`` reads: editing a class-level config
        constant should invalidate, but ``np.float64.something`` or a library
        class's attributes should not churn the key.

        *own_pkg*: the cached function's top-level package, which counts as
        user code wherever it is installed -- see ``_is_user_module``.
        """
        import sys
        if Cash._in_own_package(getattr(cls, "__module__", None), own_pkg):
            return True
        mod = sys.modules.get(getattr(cls, "__module__", None) or "")
        return mod is not None and Cash._is_user_module(mod)

    @staticmethod
    def _is_user_module(mod: Any, own_pkg: str | None = None) -> bool:
        """True for a module the user is plausibly editing between runs.

        Third-party and stdlib modules are excluded deliberately: their
        contents are expected to be fixed for a given environment, and folding
        e.g. ``os.environ`` or numpy's internals would churn the key on every
        call. Editing your venv is not a case worth keying on.

        Except the cached function's OWN package (*own_pkg*), which is user code
        wherever it is installed. The path test alone put a user's own tool,
        once `pip install`ed, in the same bucket as numpy: `settings.FACTOR`
        in the tool's own `settings.py` stopped reaching the key, and a
        reinstall with a changed constant served the old report -- while
        `from settings import FACTOR`, a helper in a sibling module and a
        same-module global all still invalidated (CAS-111).
        """
        if Cash._in_own_package(getattr(mod, "__name__", None), own_pkg):
            return True
        path = getattr(mod, "__file__", None)
        if not path:
            return False  # builtin / namespace package - nothing to edit
        try:
            p = os.path.normcase(os.path.abspath(path))
        except (TypeError, ValueError):
            return False
        if "site-packages" in p or "dist-packages" in p:
            return False
        try:
            import sysconfig
            for key in ("stdlib", "platstdlib"):
                std = sysconfig.get_paths().get(key)
                if std and p.startswith(os.path.normcase(os.path.abspath(std))):
                    return False
        except (KeyError, OSError):
            pass
        return not p.startswith(os.path.normcase(os.path.dirname(os.path.abspath(__file__))))

    #: Fileless modules that are NOT the user's code. Everything else without a
    #: __file__ is a notebook cell, a REPL, or exec'd source -- i.e. something
    #: the user is plausibly editing between runs, which is the whole point.
    _FILELESS_NON_USER = frozenset(sys.builtin_module_names) | {
        "builtins", "__future__", "_frozen_importlib", "_frozen_importlib_external",
    }

    @staticmethod
    def _is_user_code_module(mod: Any) -> bool:
        """Like :meth:`_is_user_module`, but a module with no ``__file__``
        counts as user code rather than being disqualified.

        ``_is_user_module`` returns False for a fileless module ("nothing to
        edit"). That is right for its callers and wrong here: a class defined
        in a notebook cell lives in a ``__main__`` with no ``__file__``, and it
        is precisely the thing the user edits between runs.
        """
        name = getattr(mod, "__name__", "") or ""
        path = getattr(mod, "__file__", None)
        if path is None:
            return name not in Cash._FILELESS_NON_USER
        return Cash._is_user_module(mod)

    @staticmethod
    def _is_user_code_object(obj: Any) -> bool:
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
        user code (mirroring ``_is_user_code_module``'s fileless-module
        handling).
        """
        mod_name = getattr(obj, "__module__", None)
        mod = sys.modules.get(mod_name) if mod_name else None
        if mod is None:
            return True
        if not Cash._qualname_resolves_in(mod, obj):
            return True
        return Cash._is_user_code_module(mod)

    @staticmethod
    def _qualname_resolves_in(mod: Any, obj: Any) -> bool:
        """True if *obj* is actually reachable by walking its ``__qualname__``
        from *mod*, not merely claiming *mod* via ``__module__``.

        ``getattr(x, name, default)`` only swallows ``AttributeError`` -- a
        module implementing PEP 562 ``__getattr__`` (a real pattern for
        deprecation shims: raise a custom error for an old name instead of
        just returning it) can make this walk raise something else entirely.
        Task 1's original predicate, which this refines, could never raise;
        this must not become the first way ``_is_user_code_object`` can.
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
        except Exception:
            return False  # could not confirm reachability - do not trust it

    def _read_module_attr_pairs(self, func: Callable) -> tuple[tuple[str, str], ...]:
        """``(module_global, attribute)`` pairs the body reads, from bytecode.

        ``import conf; conf.RATE`` compiles to ``LOAD_GLOBAL conf`` followed by
        ``LOAD_ATTR RATE``. Only the *module* reaches ``_read_global_data_names``,
        and modules are filtered out at fold time, so the attribute was never
        keyed on: ``conf.RATE`` went permanently stale while the equivalent
        ``from conf import RATE`` invalidated correctly. Two spellings of one
        dependency, one of them silently wrong.

        Walks nested scopes for the same reason the sibling channel does
       : a read that happens only inside a genexp still counts.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return ()
        cached = self._module_attr_cache.get(code)
        if cached is not None:
            return cached
        import dis
        pairs: set[tuple[str, str]] = set()
        for scope in Cash._iter_code_scopes(code):
            instrs = list(dis.get_instructions(scope))
            for prev, nxt in zip(instrs, instrs[1:]):
                if prev.opname != "LOAD_GLOBAL":
                    continue
                if nxt.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    continue
                name, attr = prev.argval, nxt.argval
                if not isinstance(name, str) or not isinstance(attr, str):
                    continue
                if attr.startswith("__"):
                    continue
                pairs.add((name, attr))
        result = tuple(sorted(pairs))
        if len(self._module_attr_cache) < 4096:
            self._module_attr_cache[code] = result
        return result

    def _module_attr_parts(
        self, func: Callable, func_name: str, g: dict,
    ) -> list[tuple[str, str]]:
        """Key parts for ``module.ATTR`` data reads, one level of recursion deep.

        Two shapes are covered:

        * ``conf.RATE`` - fold the attribute's content.
        * ``conf.get_rate()`` - the callable itself is already tracked by the
          helper-source channel, but that only sees its *source*. A helper whose
          source never changes while the constant it returns does was stale, so
          fold the data globals the callee reads from its own module too.

        Callables, classes and nested modules are skipped as data (the first is
        handled by the helper channel, the others carry no editable value).
        """
        parts: list[tuple[str, str]] = []
        own_pkg = self._own_package(func)
        for mod_name, attr in self._read_module_attr_pairs(func):
            obj = g.get(mod_name)
            is_mod = isinstance(obj, types.ModuleType) and self._is_user_module(obj, own_pkg)
            # ``Cfg.LIMIT`` -- a class constant read through the class NAME -- is
            # the same bytecode shape (LOAD_GLOBAL Cfg; LOAD_ATTR LIMIT) but was
            # skipped because ``Cfg`` is a class, not a module, so editing the
            # constant served stale. Fold user-class attributes too.
            is_cls = isinstance(obj, type) and self._is_user_class(obj, own_pkg)
            if not (is_mod or is_cls):
                continue
            try:
                value = (inspect.getattr_static(obj, attr) if is_cls
                         else getattr(obj, attr))
            except (AttributeError, Exception):  # noqa: BLE001 - never break a call
                continue
            label = f"{mod_name}.{attr}"
            if isinstance(value, types.ModuleType) or isinstance(value, type):
                continue
            if callable(value) and not isinstance(value, (dict, list, tuple, set)):
                # A class method/staticmethod/classmethod is handled by the
                # helper-source / self-dep channels; only recurse into a
                # module-level helper's own constants here.
                if not is_mod:
                    continue
                # One level only: fold the constants the helper itself reads.
                # Deeper recursion would drag in whole transitive namespaces for
                # a diminishing chance of catching a real edit.
                helper_globals = getattr(value, "__globals__", None)
                if not isinstance(helper_globals, dict):
                    continue
                for inner in self._read_global_data_names(value):
                    if inner not in helper_globals:
                        continue
                    iv = helper_globals[inner]
                    if isinstance(iv, types.ModuleType) or isinstance(iv, type):
                        continue
                    if callable(iv) and not isinstance(iv, (dict, list, tuple, set)):
                        continue
                    h = self._safe_global_hash(iv, func_name, f"{label}.{inner}")
                    if h is not None:
                        parts.append((f"{label}.{inner}", h))
                continue
            h = self._safe_global_hash(value, func_name, label)
            if h is not None:
                parts.append((label, h))
        return parts

    def _safe_global_hash(self, value: Any, func_name: str, label: str) -> str | None:
        """Hash *value* for the key, warning once and skipping if it cannot be."""
        try:
            stabilized = self._stabilize_for_global_hash(value, self._hash_callable_source)
            return self._hash_arg_payload((stabilized,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError, ValueError):
            self._warn_once(
                CashImpurityWarning,
                func_name,
                label,
                f"@cash.cache on {func_name}: reads '{label}' whose value could not "
                f"be hashed, so changes to it will NOT invalidate the cache.",
                code="KEY-UNHASHABLE-GLOBAL",
                fix=_UNHASHABLE_GLOBAL_FIX,
            )
            return None

    def _normalize_call_args(
        self, func_name: str, args: tuple, kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Bind ``(args, kwargs)`` to the function signature and apply defaults.

        Collapses logically-identical calls written in different forms - ``f(1)``
        vs ``f(1, y=10)`` (the default) vs ``f(x=1, y=10)``, and kwargs in any
        order - to one canonical argument shape so they share a cache key
        instead of producing wasteful misses.

        Best-effort: any introspection or bind failure (builtins with no
        signature, ``*args`` calls that don't match, deliberately mismatched
        calls) returns the inputs unchanged, so behavior never regresses.
        """
        func = self.functions.get(func_name)
        cached = self._signatures.get(func_name)
        # Re-read the signature when the name has been rebound to a different
        # function object: a notebook cell re-run with an edited default keeps
        # the qualname but changes what `apply_defaults()` must fold.
        if cached is not None and cached[0] is func:
            sig = cached[1]
        else:
            try:
                sig = inspect.signature(func) if func is not None else None
            except (ValueError, TypeError):
                sig = None
            self._signatures[func_name] = (func, sig)
        if sig is None:
            return args, kwargs
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            # The call doesn't match the signature (the function itself would
            # raise when invoked). Leave the raw form untouched.
            return args, kwargs
        # ``bound.arguments`` is ordered by parameter definition, so the result
        # is canonical regardless of how the caller wrote the call. Re-express
        # named params as kwargs; keep *args positional; sort **kwargs so its
        # order doesn't leak into the key. (We only build a payload to hash, so
        # routing named params through kwargs is purely for determinism.)
        canon_args: list[Any] = []
        canon_kwargs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name not in bound.arguments:
                continue
            val = bound.arguments[name]
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                canon_args.extend(val)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                for k in sorted(val):
                    canon_kwargs[k] = val[k]
            else:
                canon_kwargs[name] = val
        return tuple(canon_args), canon_kwargs

    _ARG_HASH_MEMO_CAP = 1024

    def _memo_arg_hash(self, arg: Any, lineage: str, content_hash: str) -> None:
        """Record ``id(arg) -> (weakref, lineage, content_hash)`` for the session,
        bounded so a long session can't grow the memo without limit. When full,
        drop it wholesale: the memo is a pure speedup, so an occasional cold
        start just re-hashes. Values that cannot be weak-referenced are skipped
        (they simply keep full-hashing).
        """
        try:
            wref = weakref.ref(arg)
        except TypeError:
            return
        memo = self._arg_hash_memo
        if len(memo) >= self._ARG_HASH_MEMO_CAP:
            memo.clear()
        memo[id(arg)] = (wref, lineage, content_hash)

    _FRAME_MEMO_CAP = 256

    def _frozen_array_hash(self, arr: Any) -> str | None:
        """The content hash of a frozen function's numpy result, computed once.

        Valid while the array is still that object and still read-only; an
        array made writeable again (``a.flags.writeable = True``) is keyed by
        content from then on.
        """
        entry = self._frozen_arrays.get(id(arr))
        if entry is None:
            return None
        wref, _producer, content_hash = entry
        if wref() is not arr or getattr(arr, "flags", None) is None or arr.flags.writeable:
            self._frozen_arrays.pop(id(arr), None)
            return None
        if content_hash is None:
            content_hash = self._try_builtin_type_hash(arr)
            entry[2] = content_hash
        return content_hash

    def _audit_frozen(self, obj: Any) -> bool:
        """Is a frozen=True result still what it was? False once it is not.

        The declaration is trusted, and checked now and then: at the object's
        8th use as an argument and every 64th after that, and at every use
        under CASH_DEBUG. The first check records a baseline -- the content
        hash for a type cash content-hashes, a digest of the pickle otherwise
        -- and each later one compares. On a change, KEY-FROZEN-MUTATED names
        the producer, the object's tag stops being trusted, and it is keyed by
        its content from then on. An object that cannot be pickled cannot be
        audited, and stays trusted.
        """
        key = id(obj)
        entry = self._frozen_uses.get(key)
        if entry is None or entry[0]() is not obj:
            try:
                wref = weakref.ref(obj, lambda _r, k=key, m=self._frozen_uses: m.pop(k, None))
            except TypeError:
                return True
            entry = [wref, 0, None]
            if len(self._frozen_uses) >= 4096:
                self._frozen_uses.clear()
            self._frozen_uses[key] = entry
        entry[1] += 1
        uses = entry[1]
        due = (self.debug or os.environ.get("CASH_DEBUG")) or uses == _FROZEN_AUDIT_FIRST or (
            uses > _FROZEN_AUDIT_FIRST and uses % _FROZEN_AUDIT_EVERY == 0)
        if not due:
            return True
        try:
            digest = self._try_builtin_type_hash(obj)
            if digest is None:
                digest = hashlib.sha256(pickle.dumps(obj)).hexdigest()
        except Exception:  # noqa: BLE001 - cannot audit: the declaration stands
            return True
        if entry[2] is None:
            entry[2] = digest
            return True
        if entry[2] == digest:
            return True
        producer = getattr(obj, "_cash_lineage_producer", None) or "a frozen=True function"
        try:
            obj._cash_lineage_src = LINEAGE_SRC_DECORATOR
        except (AttributeError, TypeError):
            pass
        self._frozen_uses.pop(key, None)
        warn_diagnostic(
            CashImpurityWarning, "KEY-FROZEN-MUTATED",
            f"a {type(obj).__name__} returned by {producer}, which is declared "
            f"@cash.cache(frozen=True), has been modified since it was returned. "
            f"Calls that received it before the change may have been served "
            f"results for the unmodified object; from now on it is keyed by its "
            f"contents.",
            f"take frozen=True off {producer} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )
        return False

    @staticmethod
    def _frame_signature(obj: Any) -> tuple:
        """What must stay the same for a pandas object's content hash to hold.

        Under copy-on-write, a frame whose data another frame also references
        cannot be written in place: every write path (``loc``/``iloc``/``at``,
        column assignment, ``inplace=True`` methods, ``update``, ``insert``,
        ``pop``) first gives the written frame NEW block arrays, and writes
        through ``.values`` / ``to_numpy()`` raise (the arrays are read-only).
        So the identities of the block arrays, the manager and the axes are an
        exact change signal -- measured on 17 mutation forms, pandas 3.0.3. The
        axis NAMES are compared by value, because ``df.index.name = ...``
        renames the same Index object and the content hash includes them.
        """
        mgr = obj._mgr
        blocks = tuple(id(block.values) for block in mgr.blocks)
        if hasattr(obj, "columns"):
            return (id(mgr), blocks, id(obj.columns), tuple(obj.columns.names),
                    id(obj.index), tuple(obj.index.names))
        return (id(mgr), blocks, id(obj.index), tuple(obj.index.names), obj.name)

    def _frame_memo_lookup(self, obj: Any) -> str | None:
        """The content hash recorded for *obj*, if *obj* has not changed since."""
        entry = self._frame_memo.get(id(obj))
        if entry is None:
            return None
        wref, _held, signature, content_hash = entry
        try:
            if wref() is obj and self._frame_signature(obj) == signature:
                return content_hash
        except Exception:  # noqa: BLE001 - a pandas internals change: just re-hash
            pass
        self._frame_memo.pop(id(obj), None)
        return None

    def _frame_memo_store(self, obj: Any, content_hash: str) -> None:
        """Remember *obj*'s content hash, and hold a shallow copy of it.

        The shallow copy shares the data and is what makes the signature
        exact: while cash references the blocks, pandas must copy before any
        write. Cost: the first in-place write to each block afterwards copies
        that block, once. The entry, copy included, goes when *obj* is
        collected, or when the memo fills.
        """
        try:
            held = obj.copy(deep=False)
            signature = self._frame_signature(obj)
            memo = self._frame_memo
            key = id(obj)
            wref = weakref.ref(obj, lambda _ref, key=key, memo=memo: memo.pop(key, None))
        except Exception:  # noqa: BLE001 - the memo is a speedup; hash every time
            return
        if len(self._frame_memo) >= self._FRAME_MEMO_CAP:
            self._frame_memo.clear()
        self._frame_memo[key] = (wref, held, signature, content_hash)

    def _hash_arg_payload(self, args: tuple, kwargs: dict) -> str:
        """Hash one concrete ``(args, kwargs)`` form. May raise on unpicklable
        values; the caller decides whether to retry with a different form."""
        def get_arg_hash(arg):
            # Content-authoritative builtin hashers FIRST. pandas /
            # numpy / polars / pyarrow / modin / dask hash the argument's
            # *content*, which is byte-stable across processes and kernel
            # restarts. The notebook's in-memory ``_cash_lineage_hash`` (checked
            # next) is recomputed per session and is NOT reproducible across a
            # restart -- keying a persisted @cash.cache entry on it makes the
            # decorator miss after a restart even though the argument is
            # byte-identical (re-training the model the docs promise survives a
            # restart). A value that has a content hash must key on content so
            # the entry survives; the modest extra hashing cost is the price of
            # the flagship "restart-and-run-all in seconds" guarantee. Mirrors
            # principle: the reproducible signal, not the volatile
            # in-memory one, is authoritative.
            # Fast path: skip re-hashing a possibly-huge argument we already
            # content-hashed this session, when it is provably the SAME,
            # unmutated object. Keyed on ``id`` (NOT lineage): two *different*
            # objects that happen to share a lineage string must still be
            # distinguished by content -- an explicit invariant
            # (test_arg_hash_restart_stable) -- and distinct live objects have
            # distinct ids. The entry is validated on read by BOTH a weakref
            # identity check (guards id reuse after GC) AND the object's
            # ``_cash_lineage_hash`` being unchanged (cash's own mutation signal,
            # the same one it trusts to cache every notebook statement). The
            # stored value is still the reproducible content hash, so the cache
            # key is byte-identical and restart-safe; the memo is a pure
            # within-session speedup, empty after a restart.
            #
            # Trusted only where something KEEPS it current: the notebook's
            # statement layer re-tags a variable on every assignment and
            # mutation. The decorator also tags what it returns, and nothing
            # ever moves that tag -- in a script, `q.F = 0.03; run(q)` or
            # `df.loc[0, "a"] = 100` left it as it was, and both the memo below
            # and the tag-as-identity shortcut further down served the result
            # for the unmutated object (rounds 17-18).
            lineage = getattr(arg, '_cash_lineage_hash', None)
            if lineage is not None:
                src = getattr(arg, '_cash_lineage_src', None)
                if src == LINEAGE_SRC_FROZEN:
                    if not self._audit_frozen(arg):
                        lineage = None
                elif src != LINEAGE_SRC_STATEMENT:
                    lineage = None
            if self._frozen_arrays and id(arg) in self._frozen_arrays:
                frozen_hash = self._frozen_array_hash(arg)
                if frozen_hash is not None:
                    return frozen_hash
            if lineage is not None:
                entry = self._arg_hash_memo.get(id(arg))
                if entry is not None:
                    wref, memo_lineage, content_hash = entry
                    if memo_lineage == lineage and wref() is arg:
                        return content_hash
            # pandas >= 3 copy-on-write: an exact "has this frame changed?"
            # check instead of a trusted tag. See `_frame_memo_lookup`.
            frame_memo = lineage is None and _is_cow_pandas(arg)
            if frame_memo:
                content_hash = self._frame_memo_lookup(arg)
                if content_hash is not None:
                    return content_hash

            # Overriding hashers, ahead of everything cash would do itself.
            # The user has said their identity for this type beats content
            # hashing, which is the only way to stop re-reading a 800MB array
            # on every call. Guarded by the emptiness check so the ordinary
            # case pays one dict truth test, not a loop.
            if self._override_hashers:
                for type_, (hasher_fn, src_hash) in self._override_hashers.items():
                    if isinstance(arg, type_):
                        return f"{src_hash}:{hasher_fn(arg)}"

            builtin_hash = self._try_builtin_type_hash(arg)
            if builtin_hash is not None:
                if lineage is not None:
                    self._memo_arg_hash(arg, lineage, builtin_hash)
                elif frame_memo:
                    self._frame_memo_store(arg, builtin_hash)
                return builtin_hash
            # Notebook lineage hash: the authoritative, cheap identity for
            # values that carry NO content hasher (custom objects). Kept ahead
            # of registered hashers so a lineage-carrying object short-circuits
            # its (possibly expensive) registered hasher within a session
            # (test_hasher_priority_cash_hash_first).
            if lineage is not None:
                return lineage
            for type_, (hasher_fn, src_hash) in self._type_hashers.items():
                if isinstance(arg, type_):
                    # Embed the hasher source hash so that changing the
                    # hasher's body invalidates dependent cache entries
                    # even when the hasher's output coincidentally matches.
                    return f"{src_hash}:{hasher_fn(arg)}"
            return arg

        # Timed per argument -- two clock reads each -- so that a
        # CACHE-NET-LOSS verdict can name the argument that costs the time.
        costliest: tuple | None = None

        def timed(label: str, value: Any) -> Any:
            nonlocal costliest
            t0 = time.perf_counter()
            digest = get_arg_hash(value)
            seconds = time.perf_counter() - t0
            if costliest is None or seconds > costliest[1]:
                producer = getattr(value, "_cash_lineage_producer", None)
                if producer is None and self._frozen_arrays and id(value) in self._frozen_arrays:
                    producer = self._frozen_arrays[id(value)][1]
                old_pandas = (type(value).__name__ in ("DataFrame", "Series")
                              and (type(value).__module__ or "").startswith("pandas")
                              and not _is_cow_pandas(value))
                costliest = (label, seconds, type(value).__name__, producer, old_pandas)
            return digest

        hashed_args = tuple(timed(f"#{i}", a) for i, a in enumerate(args))
        hashed_kwargs = {k: timed(k, v) for k, v in kwargs.items()}
        _ARG_COST.last = costliest

        payload: Any = (hashed_args, hashed_kwargs)
        # A set/frozenset pickles in PYTHONHASHSEED-dependent iteration
        # order, so the same set argument hashes differently in every
        # process, silently breaking cross-process cache hits. Canonicalise
        # to a deterministic, order-independent form (recursing into objects
        # so a set inside a dataclass is covered too) - but only when a set
        # is actually present, so all other argument shapes keep
        # byte-identical keys. _stable_key_repr also canonicalises dict order.
        #
        # When there's no set, still canonicalise dict *ordering* so two dict
        # args equal but for insertion order share a key. This is
        # byte-identical for already-sorted dicts (the normalised top-level
        # kwargs), so only out-of-order dict values change their key.
        if _contains_set(payload):
            payload = _stable_key_repr(payload)
        else:
            payload = _canonicalize_dict_order(payload)
        args_bytes = pickle.dumps(payload)
        return hashlib.sha256(args_bytes).hexdigest()

    def _serialize_args(self, func_name: str, args: tuple, kwargs: dict,
                        normalized: tuple[tuple, dict] | None = None) -> str | None:
        """Hash the arguments, canonicalised.

        *normalized* lets a caller that has ALREADY canonicalised pass the
        result in rather than have it recomputed. That is not an optimisation:
        the code channel (`_fold_code_args`) and this value channel must key
        off the SAME bound arguments, or `f()` and `f(<the default>)` -- the
        same logical call -- disagree in one channel and split into two cache
        entries. One canonicalisation, shared, is the only way that invariant
        holds by construction rather than by two call sites staying in step.
        """
        if normalized is None:
            normalized = self._normalize_call_args(func_name, args, kwargs)
        try:
            return self._hash_arg_payload(*normalized)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            # Normalization can fold a default value into the payload (so
            # f(1) keys identically to f(1, y=<default>)). If that default is
            # unpicklable it must not make a call that hashed fine before stop
            # caching - retry with the raw, un-normalized form first.
            if normalized[0] is not args or normalized[1] is not kwargs:
                try:
                    return self._hash_arg_payload(args, kwargs)
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
                    pass
            # Pickle failure here is surfaced via CashCacheIneffectiveWarning in
            # _resolve_cache_key (which sees the None return). Keep this log at
            # debug level so it's available when explicitly enabled but doesn't
            # double-warn.
            logger.debug("Could not serialize arguments for %s: %s", func_name, e)
            return None

    @staticmethod
    def _try_hash_pandas(value: Any, type_name: str) -> str | None:
        """Hash a pandas DataFrame or Series over values AND schema.

        ``hash_pandas_object`` covers row values + index values but NOT the
        schema labels: column names, ``Series.name``, and index name(s) are
        invisible to it, so ``df.rename(columns=...)`` (or an empty frame of
        any shape) collided with the original and returned its cached result
       . Fold the labels in as a digest prefix.
        """
        try:
            import pandas as pd
            if type_name == 'DataFrame':
                schema = f"{list(value.columns)!r}:{list(value.index.names)!r}:"
            else:  # Series
                schema = f"{value.name!r}:{list(value.index.names)!r}:"
            h = hashlib.sha256(schema.encode('utf-8'))
            h.update(pd.util.hash_pandas_object(value).values.tobytes())
            return h.hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash pandas %s via hash_pandas_object", type_name)
            return None

    @staticmethod
    def _array_layout(value: Any) -> str:
        """The order *value*'s axes are laid out in memory: ``C``, ``F`` or ``K…``.

        The key used to fold in the raw strides (0fd2cb5), which separated C-
        from F-ordered arrays -- the point -- but also a strided VIEW from its
        contiguous copy. Those hold the same values in the same memory order,
        so no order-reading callee (``ravel(order='A'/'K')``, ``reshape``) tells
        them apart -- only ``.flags`` does, and a result computed FROM
        contiguity now shares an entry between the two, knowingly. What did
        tell them apart, on every run, was the cache itself. A function that
        returned ``arr[:, 0]`` handed its caller a view on the computing run and
        a contiguous copy on every restored one, so the caller's key changed
        between the two and its expensive step ran twice after every upstream
        edit (round 17, measured 1 then 1 then 0 executions).

        So: the axes of length > 1, ordered by |stride| from outermost in, with
        a broadcast (zero-stride) axis outermost. Identity is ``C``, reversed is
        ``F``; anything else spells the permutation. Stride MAGNITUDE and sign
        do not change what a memory-order read returns, so they stay out.

        Except for one flag. ``order='A'`` (``ravel``, ``reshape``, ``tobytes``,
        ``copy``) reads in Fortran order only when the array is F-CONTIGUOUS,
        and C order otherwise -- so an F-like strided view (``a.T[::2]``) and
        its F-contiguous copy read differently, and sharing ``F`` handed one the
        other's result (round 18, 8/8). ``Fs`` is the F-like array that is not
        F-contiguous. Nothing else needs the flag: with two or more axes longer
        than 1, only an F-like layout can be F-contiguous, and ``order='A'``
        reads everything else in C order, as ``C`` and ``K…`` already imply.

        One case still re-keys once: an F-like but non-contiguous view is stored
        by pickle as a C-ordered copy, and it genuinely ravels differently from
        one, so the restored value must key apart. Safe direction.
        """
        axes = [(axis, stride) for axis, (n, stride)
                in enumerate(zip(value.shape, value.strides)) if n > 1]
        if len(axes) <= 1:
            return "C"
        outer_first = sorted(
            axes, key=lambda a: (-abs(a[1]) if a[1] else float("-inf"), a[0]))
        perm = tuple(axis for axis, _ in outer_first)
        natural = tuple(axis for axis, _ in axes)
        if perm == natural:
            return "C"
        if perm == natural[::-1]:
            return "F" if value.flags.f_contiguous else "Fs"
        return "K" + ",".join(map(str, perm))

    @staticmethod
    def _try_hash_numpy(value: Any) -> str | None:
        """Hash a numpy ndarray over its FULL contents.

        Correctness requires hashing every byte, not a sample: two large
        arrays that differ only outside a sampled window would otherwise
        collide and return a wrong cached result (a silent data-corruption
        bug, especially for the large ML/data arrays caching targets). Shape
        and dtype are folded in so a reshape or retype of the same bytes does
        not collide. Uses a zero-copy ``memoryview`` for contiguous arrays and
        falls back to ``tobytes()`` (C-order copy) otherwise.

        The LAYOUT is folded in too -- the order the axes sit in memory, see
        `_array_layout` -- because the C-order fallback above erases it.
        Without it a C-ordered and an F-ordered array holding equal values hash
        identically, and a layout-sensitive callee is served the other one's
        result: measured, ``np.ravel(x, order='A')`` returned ``[0, 1, 2, …]``
        for an F-ordered input whose true answer is ``[0, 4, 8, 1, …]``.
        Normalising to C-order is right for value EQUALITY and wrong for a KEY.
        """
        try:
            h = hashlib.sha256(
                f"{value.shape}:{value.dtype}:{Cash._array_layout(value)}:".encode())
            if getattr(value.dtype, "hasobject", False):
                # object-dtype arrays: the buffer holds raw PyObject *pointers*,
                # not content, so tobytes() hashes memory addresses - identical
                # content in fresh objects never collides (permanent misses,
                # cross-process-unstable) and address reuse could alias distinct
                # content onto one key. Hash the elements' stable
                # representation instead (canonicalising nested sets/dicts so the
                # key is order- and PYTHONHASHSEED-independent).
                payload = _stable_key_repr(value.tolist())
                h.update(pickle.dumps(payload, protocol=4))
                return h.hexdigest()
            try:
                h.update(memoryview(value).cast("B"))   # no copy if C-contiguous
            except (TypeError, ValueError):
                h.update(value.tobytes())                # non-contiguous / odd layout
            return h.hexdigest()
        except (TypeError, ValueError, AttributeError, MemoryError, pickle.PicklingError):
            logger.debug("Failed to hash numpy ndarray")
            return None

    @staticmethod
    def _try_hash_polars(value: Any, type_name: str) -> str | None:
        """Hash a polars DataFrame, Series, or LazyFrame.

        A ``LazyFrame`` is identified by ``serialize()``, not ``explain()``.
        ``explain()`` renders the human-readable QUERY PLAN, and two frames
        over different in-memory data print identically -- both
        ``pl.DataFrame({"x": [1, 2, 3]}).lazy()`` and the same over
        ``[10, 20, 30]`` are ``DF ["x"]; PROJECT */1 COLUMNS``, so the second
        call was served the first's result. A wrong answer, reachable in three
        lines. ``serialize()`` carries the plan *and* the data the plan closes
        over, and is byte-identical across processes, so persisted entries
        still hit after a restart.

        KNOWN GAP: a plan that reads from an external source
        (``scan_csv``/``scan_parquet``/...) serializes the PATH, not the file's
        contents, so editing that file in place does not move the key. Closing
        that would mean collecting the frame to build a cache key, which
        defeats the point of a LazyFrame and can be arbitrarily expensive.
        Collect before passing, or name the file with ``file_depends_on=``.
        """
        try:
            import polars as pl
            if isinstance(value, pl.DataFrame):
                return hashlib.sha256(
                    value.hash_rows().to_list().__repr__().encode('utf-8')
                ).hexdigest()
            if isinstance(value, pl.Series):
                return hashlib.sha256(
                    value.hash().to_list().__repr__().encode('utf-8')
                ).hexdigest()
            if isinstance(value, pl.LazyFrame):
                try:
                    return hashlib.sha256(value.serialize()).hexdigest()
                except Exception:  # noqa: BLE001 - polars raises its own types
                    # A polars whose serialize() is missing or refuses this
                    # plan. The plan digest is unsound, but it is what the
                    # previous behaviour was; degrading to it beats declining
                    # to cache on a version difference.
                    logger.debug("polars LazyFrame serialize() unavailable; "
                                 "falling back to the plan digest")
                    return hashlib.sha256(
                        str(value.explain()).encode('utf-8')).hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash polars %s", type_name)
        return None

    @staticmethod
    def _try_hash_pyarrow(value: Any, type_name: str) -> str | None:
        """Hash a PyArrow Table or RecordBatch."""
        try:
            import pyarrow as pa
            if isinstance(value, (pa.Table, pa.RecordBatch)):
                # Hash the schema + every underlying buffer of every column.
                # The previous size-gated path hashed ONLY schema+row-count for
                # tables >=10 MB, so any two same-shape tables collided into a
                # wrong cache hit. Buffer hashing is zero-copy and total.
                h = hashlib.sha256(f"{value.schema}:{value.num_rows}:".encode())
                for col in value.columns:
                    chunks = col.chunks if hasattr(col, "chunks") else [col]
                    for chunk in chunks:
                        for buf in chunk.buffers():
                            if buf is not None:
                                h.update(memoryview(buf))
                return h.hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError, MemoryError):
            logger.debug("Failed to hash PyArrow %s", type_name)
        return None

    @staticmethod
    def _try_hash_modin(value: Any, type_name: str) -> str | None:
        """Hash a modin DataFrame or Series via pandas conversion."""
        try:
            pandas_val = value._to_pandas() if hasattr(value, '_to_pandas') else value
            import pandas as pd
            return hashlib.sha256(
                pd.util.hash_pandas_object(pandas_val).values.tobytes()
            ).hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash modin %s", type_name)
            return None

    @staticmethod
    def _try_hash_dask(value: Any) -> str | None:
        """Hash a dask object via its task graph key."""
        try:
            graph_key = str(value.__dask_keys__())
            return hashlib.sha256(graph_key.encode('utf-8')).hexdigest()
        except (TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash dask object via __dask_keys__")
            return None

    @staticmethod
    def builtin_hashed_family(type_: type) -> str | None:
        """Which built-in content hasher claims *type_*, or ``None``.

        The type-level counterpart of `_try_builtin_type_hash`, which can
        only answer the question about a value it already holds. Used to tell
        a user at ``register_hasher`` time that the hasher they just handed
        over would never be consulted -- the moment they can still do
        something about it.
        """
        return _builtin_hash_family(
            type_.__name__, getattr(type_, '__module__', '') or '')

    @staticmethod
    def _try_builtin_type_hash(value: Any) -> str | None:
        """Attempt to hash common types that may not pickle well.

        Returns a hex-digest string for recognised types (pandas, numpy,
        polars, PyArrow, modin, dask) or ``None`` when the value is not
        a supported type and should fall through to the default pickle
        path.

        The ``_try_`` prefix signals that ``None`` is a normal,
        expected return value - not an error.
        """
        type_name = type(value).__name__
        module = type(value).__module__ or ''
        family = _builtin_hash_family(type_name, module)

        if family == 'pandas':
            return Cash._try_hash_pandas(value, type_name)

        if family == 'numpy':
            return Cash._try_hash_numpy(value)

        if family == 'polars':
            return Cash._try_hash_polars(value, type_name)

        if family == 'pyarrow':
            return Cash._try_hash_pyarrow(value, type_name)

        if family == 'modin':
            return Cash._try_hash_modin(value, type_name)

        if family == 'dask':
            return Cash._try_hash_dask(value)

        # Generators / iterators - cannot hash
        if hasattr(value, '__next__') and hasattr(value, '__iter__'):
            return None

        return None

    def _compute_with_lock(
        self,
        cache_key: str,
        func_name: str,
        ttl: int | None,
        args_hash: str,
        call_start: float,
        compute_and_store: Callable[[], Any],
    ) -> Any:
        """Compute with double-checked locking; falls back to unlocked on error.

        Acquiring the lock is best-effort: if *any* backend raises while taking
        it (a Redis ``LockError`` on contention/timeout, a dropped connection,
        an OSError on a file lock), we degrade to an unlocked compute rather than
        crash the user's call. Acquisition, compute, and release are separated so
        a release failure can't re-run the compute, and a compute exception
        propagates normally (it is not mistaken for a lock failure)."""
        lock_cm = self.backend.lock(cache_key)
        try:
            lock_cm.__enter__()
        except Exception as e:  # noqa: BLE001 - any acquisition failure -> unlocked
            self._warn_lock_failed(func_name, e)
            return compute_and_store()
        try:
            raw_locked_metadata, locked_data = self.backend.get(cache_key)
            locked_metadata = (
                CacheMetadata.from_dict(raw_locked_metadata)
                if raw_locked_metadata is not None else None
            )
            # The SAME validity test as the unlocked path, by calling the same
            # function -- not a hand-rolled subset of it.
            #
            # This block used to re-implement the checks, and it kept losing
            # one. First ``_chunks_are_intact``: a chunked manifest can outlive
            # its chunks, and the reader terminates quietly on the first
            # missing one, so the re-read handed back a SHORT iterator -- 3 of
            # 10 items when a later chunk was gone, 0 when the first one was,
            # with no recompute, no error and no warning. Then, found by a
            # round-16 tester, ``_auto_file_deps_fresh``: under
            # ``use_locking=True`` a file the cached function reads could be
            # edited and the next process would serve the answer computed from
            # the old contents. The unlocked read at the top of the wrapper
            # caught the stale file and reported a miss -- and then this
            # re-read, which never asked, handed the entry back anyway. 5/5,
            # against 0/5 for the same edit without the flag.
            #
            # That is twice, so the duplication is the defect. One function
            # decides whether an entry may be served.
            hit = self._try_get_cached(
                cache_key, locked_metadata, locked_data, call_start,
                args_hash, func_name, ttl,
            )
            if hit is not _CACHE_MISS:
                return self._wrap_iterator_hit(cache_key, locked_metadata, hit)
            return compute_and_store()
        finally:
            try:
                lock_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - releasing failed; compute already done
                logger.debug("lock release failed for %s", func_name)

    def _compute_cache_key(self, func_name: str, state_hash: str, dynamic_hash: str, args_hash: str) -> str:
        return f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"

    def _validate_ttl(self, metadata: CacheMetadata | None, ttl: int | None) -> None:
        if ttl is not None and metadata:
            timestamp = metadata.timestamp or 0
            if time.time() - timestamp > ttl:
                raise CacheExpiredError("Cache expired")

    @staticmethod
    def _lineage_hash(cache_key: str, auto_file_deps: dict | None) -> str:
        """The lineage hash a result carries downstream.

        It is the producer's ``cache_key`` PLUS a fingerprint of the files the
        producer read. The cache key alone omits file state (files invalidate
        via a freshness re-stat, not via the key), so without this a downstream
        function keyed on the lineage hash would return a STALE result after an
        upstream file changed - the producer recomputes, but its new output
        carries the same lineage hash as the old one. Folding the file deps in
        gives a changed file a distinct lineage. No deps -> unchanged key.

        The fingerprint is built from the recorded content ``hash`` plus the
        size, NOT the mtime. Content is the authoritative freshness
        signal everywhere else, and mtime is the untrustworthy one: keying
        lineage on it would hand a touched-but-identical file a new lineage and
        needlessly recompute every downstream consumer, while a same-size edit
        under an indistinguishable mtime would reuse the old lineage and serve
        stale. A snapshot with no ``hash`` falls back to the mtime so the
        entry still keeps a stable lineage.
        """
        if not auto_file_deps:
            return cache_key
        fp = hashlib.sha256(
            repr(sorted(
                (p, d.get('hash') or d.get('mtime'), d.get('size'))
                for p, d in auto_file_deps.items()
            )).encode('utf-8')
        ).hexdigest()
        return f"{cache_key}:fdeps:{fp}"

    def _attach_lineage(self, result: Any, cache_key: str,
                        auto_file_deps: dict | None = None,
                        ttl: int | None = None,
                        func_name: str | None = None) -> None:
        """Attach lineage hash to result if it supports attribute setting.

        Works with pandas DataFrame/Series, polars DataFrame/Series, PyArrow
        Table, modin DataFrame, and any object that allows setting attributes.

        Skipped when the producer has a ``ttl``: a TTL'd value's identity is not
        captured by its cache key (the value changes over time while the key
        stays the same), so a downstream cached function keyed on the lineage
        hash would return a stale result after the upstream's TTL refresh. With
        no lineage hash, the downstream content-hashes the actual current value
        instead - correct, just without the large-value short-circuit.
        """
        if ttl is not None:
            return
        frozen = func_name is not None and func_name in self._frozen_funcs
        if frozen and type(result).__name__ == "ndarray" and \
                (type(result).__module__ or "").startswith("numpy"):
            # An array cannot carry a tag, and read-only is a promise numpy
            # enforces: a write raises instead of going stale.
            try:
                result.flags.writeable = False
                self._frozen_arrays[id(result)] = [
                    weakref.ref(result, lambda _r, k=id(result), m=self._frozen_arrays: m.pop(k, None)),
                    func_name, None]
            except (AttributeError, TypeError, ValueError):
                pass
            return
        lineage = self._lineage_hash(cache_key, auto_file_deps)
        try:
            # Say who wrote it: nothing will move this tag when the value is
            # mutated, so `_hash_arg_payload` must not take it for the content
            # -- unless the function was declared frozen=True.
            try:
                result._cash_lineage_src = LINEAGE_SRC_FROZEN if frozen else LINEAGE_SRC_DECORATOR
                if func_name is not None:
                    # Named in CACHE-NET-LOSS and KEY-FROZEN-MUTATED.
                    result._cash_lineage_producer = func_name
            except (AttributeError, TypeError):
                pass
            type_name = type(result).__name__
            module = type(result).__module__ or ''

            # pandas DataFrame / Series (has attrs dict)
            if module.startswith('pandas') and type_name in ('DataFrame', 'Series'):
                result._cash_lineage_hash = lineage
                return

            # polars DataFrame / Series
            if module.startswith('polars') and type_name in ('DataFrame', 'Series'):
                try:
                    result._cash_lineage_hash = lineage
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to polars %s", type_name)
                return

            # modin DataFrame / Series
            if module.startswith('modin') and type_name in ('DataFrame', 'Series'):
                try:
                    result._cash_lineage_hash = lineage
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to modin %s", type_name)
                return

            # PyArrow Table
            if module.startswith('pyarrow') and type_name in ('Table', 'RecordBatch'):
                try:
                    result._cash_lineage_hash = lineage
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to PyArrow %s", type_name)
                return

            # Generic: try setting on DataFrame-like objects with attrs
            if type_name == 'DataFrame' and hasattr(result, 'attrs'):
                result._cash_lineage_hash = lineage
                return

            # Generic custom objects: any instance that accepts attribute
            # assignment can carry the lineage hash, so a custom result short-
            # circuits downstream content-hashing the same way a DataFrame does.
            # Builtins (list/dict/tuple/str/numbers) and __slots__ objects with
            # no matching slot reject the assignment - caught below, harmless
            # skip - so those keep content-hashing (a hard Python limitation).
            try:
                result._cash_lineage_hash = lineage
            except (AttributeError, TypeError):
                logger.debug("Cannot attach _cash_lineage_hash to %s", type_name)

        except (AttributeError, TypeError):
            logger.debug("Failed to attach lineage hash to %s result", type(result).__name__)

    def _log_decorator_call(
        self,
        func_name: str,
        cache_hit: bool,
        execution_time: float,
        args_hash: str,
        cache_key: str,
        time_saved: float = 0.0,
        miss_detail: str = "",
    ) -> None:
        """Record a decorator call event for notebook integration.

        Thread-safe: uses a lock to protect concurrent appends.
        The notebook ``StatementProcessor`` drains this log after each
        statement execution to include decorator call metrics in the badge.

        ``execution_time`` is the wall-time of *this* operation - a lookup on a
        hit, the compute on a miss. ``time_saved`` is the compute a hit
        *avoided* (the originally-measured execution time stored with the
        cached entry), and 0.0 on a miss. They are distinct: a hit's
        ``execution_time`` is microseconds, but its ``time_saved`` is the full
        compute it stood in for. ``cache_info()['total_time_saved']`` sums the
        latter - summing ``execution_time`` (the old behaviour) under-reported
        savings by orders of magnitude.
        """
        entry = {
            'func_name': func_name,
            'cache_hit': cache_hit,
            'execution_time': execution_time,
            'time_saved': time_saved,
            'args_hash': args_hash,
            'cache_key': cache_key,
            'timestamp': time.time(),
        }
        outcome: dict[str, Any] = {}
        if not cache_hit:
            if args_hash == 'unhashable':
                reason = (MISS_UNHASHABLE, "an argument could not be hashed, so "
                                           "there is no key to look up")
            elif args_hash == 'error':
                reason = (MISS_KEY_FAILED, "building the key raised")
            elif args_hash == 'unkeyable':
                reason = (MISS_MOCKED, f"{miss_detail}, which has no code to key, "
                                       f"so the call ran uncached")
            else:
                reason = self._pending_miss.pop(cache_key, None) or (MISS_FIRST, "")
            entry['miss_reason'] = reason
            outcome = self._store_outcomes.get(cache_key) or {}
            # Only this call's own outcome. A streamed result is logged before
            # it is stored, and must not borrow the previous call's verdict.
            if outcome.get('at', 0) < entry['timestamp'] - execution_time:
                outcome = {}
            entry['not_persisted'] = outcome.get('not_persisted')
            entry['not_stored'] = outcome.get('not_stored')
        with self._decorator_call_log_lock:
            self._decorator_call_log.append(entry)
        # Asked for, not merely permitted: an application that turned the
        # `cash` logger up to INFO did not ask for a line per call.
        if (self.verbose or self.debug) and _calls_logger.isEnabledFor(logging.INFO):
            _calls_logger.info("%s", self._describe_call(entry))

    @staticmethod
    def _describe_call(entry: dict[str, Any]) -> str:
        """One line for the per-call log: what happened, and on a miss, why."""
        name = entry['func_name']
        if entry['cache_hit']:
            saved = entry.get('time_saved') or 0.0
            return f"HIT  {name}  (saved {saved:.2f}s)"
        kind, detail = entry.get('miss_reason') or (MISS_FIRST, "")
        line = f"MISS {name}  {kind}" + (f": {detail}" if detail else "")
        line += f"  (ran {entry['execution_time']:.2f}s"
        if entry.get('not_stored'):
            line += f"; not stored: {entry['not_stored']}"
        elif entry.get('not_persisted'):
            line += (f"; kept in RAM only -- {entry['not_persisted']} -- so "
                     f"another process will recompute it")
        return line + ")"

    def _warn_cache_if_raised(
        self, func_name: str, error: BaseException, *, stacklevel: int | None = None,
    ) -> None:
        """Surface a raised ``cache_if`` predicate as a user-visible warning.

        Previously this was a ``logger.debug`` - invisible to anyone not
        explicitly configuring logging. Promoted to a one-shot
        `CashCacheIneffectiveWarning` so a buggy predicate is
        diagnosed instead of silently disabling the cache.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "cache_if",
            f"@cash.cache on {func_name}: cache_if predicate raised "
            f"{type(error).__name__} ({error}), so the result is returned "
            f"un-cached and every later call recomputes.",
            code="CACHE-IF-RAISED",
            fix="make the predicate total -- it must handle every shape the "
                "result can take -- or drop cache_if= to restore caching.",
            stacklevel=stacklevel,
        )

    def _warn_metadata_invalid(
        self, func_name: str, error: BaseException, *, stacklevel: int | None = None,
    ) -> None:
        """Surface a malformed cache-metadata read as a user-visible warning.

        Happens when a backend returns a metadata dict missing the
        expected keys (e.g. a partially-written entry from an older
        cash version, or a corrupted file on disk). The call falls
        through to recompute - but the user should know.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "metadata_invalid",
            f"@cash.cache on {func_name}: a stored cache entry's metadata "
            f"could not be validated ({type(error).__name__}: {error}), so "
            f"cash treated the entry as absent and recomputed.",
            code="STORE-METADATA-INVALID",
            fix="nothing, for a one-off; if it keeps appearing, run "
                "f.cache_clear() so the unreadable records are replaced.",
            stacklevel=stacklevel,
        )

    def _warn_lock_failed(
        self, func_name: str, error: BaseException, *, stacklevel: int | None = None,
    ) -> None:
        """Surface a backend-locking failure as a user-visible warning.

        Previously this was ``logger.warning`` - visible to anyone who
        wired up logging.warning, but invisible to anyone running with
        default config. Promoted to a CashCacheIneffectiveWarning so
        the user notices the implicit race risk.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "lock_failed",
            f"@cash.cache on {func_name}: backend lock acquisition failed "
            f"({type(error).__name__}: {error}), so cash proceeded without the "
            f"lock and concurrent calls with the same args may compute "
            f"redundantly.",
            code="STORE-LOCK-FAILED",
            fix="investigate the backend the exception names -- a full disk, a "
                "stale lock file, or a cache_dir on a filesystem where locking "
                "does not work.",
            stacklevel=stacklevel,
        )

    def _inputs_moved_during_call(self, func_name: str, tracker: Any) -> bool:
        """Did a file this call read change before the call returned?

        The entry's file fingerprints are taken when it is STORED. A file
        rewritten after the body read it but before it returned was
        fingerprinted in its new state, so the entry matched the new file
        and served the old answer on every later call (CAS-109, round 17:
        a sync job overlapping a long pipeline; and, one level up, an outer
        aggregate re-fingerprinting a file its inner call had already read).
        The documented mitigation -- write to a temp file and rename -- did
        not help, because the rename lands before the store.

        The result is still returned: it is what the body computed. It is
        only not cached, because nothing can say which content it came from.
        """
        moved_fn = getattr(tracker, "inputs_changed_since_read", None)
        if moved_fn is None:
            return False
        try:
            moved = moved_fn()
        except Exception:  # noqa: BLE001 - never let the check break a call
            return False
        if not moved:
            return False
        shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
        self._warn_once(
            CashCacheStoreFailedWarning,
            func_name,
            "input_changed",
            f"@cash.cache on {func_name}: {shown} changed while the call was "
            f"running, after it had been read. The result was returned but not "
            f"cached, because it cannot be told which version of the file it "
            f"was computed from.",
            code="STORE-INPUT-CHANGED",
            fix="nothing, if something else writes these files while this runs "
                "-- the next call reads the settled file and caches normally. If "
                "the function writes a file it also reads, that is why: split the "
                "read and the write.",
        )
        return True

    def _warn_unseeded_randomness(
        self,
        func: Callable,
        func_name: str,
        allow_random: bool,
    ) -> None:
        """Warn once if *func*'s source draws from an unseeded RNG.

        The decorator used to be completely silent here while the notebook path
        warned, so ``@cash.cache`` would freeze a non-deterministic result
        forever with nothing on screen to say so. The two paths now share ONE
        detector — :class:`~cash.notebook.randomness.RandomnessDetector`, reused
        verbatim — so "what counts as unseeded" cannot drift between them.

        Runs at DECORATION time, once per function. The analysis is a pure
        function of the source, so there is no reason to pay for it per call,
        and ``cache()`` already reads the source anyway (``_register_func`` ->
        ``get_source_hash``), which warms ``linecache`` for us.

        A fresh detector is used per function rather than one shared across the
        instance. The detector's seed-tracking is *session*-scoped, which is
        right for a notebook (cells run top-to-bottom in one namespace) but
        wrong here: decoration order is not call order, so letting a
        ``np.random.seed(0)`` inside function A silence function B would be
        unsound. Per-function analysis keeps the verdict a property of the
        source we are actually looking at.

        Silent when:

        * ``allow_random=True``, or the notebook's ``# @cash:allow-random``
          appears in the function's own source (same directive vocabulary,
          parsed by the same ``parse_annotation_line``);
        * the RNG is seeded — the whole point, and the reason a seeded draw
          must not be flagged;
        * the source cannot be read (``exec``/REPL-defined functions). The
          purity analyzer has the identical blind spot and treats it the same
          way: no source, no claim.
        """
        if allow_random:
            return

        try:
            src_lines, first_lineno = inspect.getsourcelines(func)
        except SOURCE_RETRIEVAL_ERRORS:
            # No retrievable source (exec'd, REPL, C function). Staying silent
            # is the conservative choice: we cannot see a draw, so we cannot
            # honestly claim there is one.
            return
        src = textwrap.dedent("".join(src_lines))

        # Honour the notebook's in-source opt-out too. Users coming from
        # ``%cash_on`` reach for the comment, and the source is already in hand.
        for line in src.splitlines():
            ann = parse_annotation_line(line)
            if ann is not None and ann.allow_random:
                return

        # A seeding call fed by a PARAMETER -- `default_rng(seed)` -- counts as
        # seeded to the detector, but whether it is depends on the call:
        # `def simulate(params, seed=None)` draws from OS entropy whenever the
        # caller leaves the seed out, and R Monte Carlo replicates came back
        # identical with nothing said (CAS-116). Note which parameters, and
        # check their bound value per call.
        seed_params = _seed_parameters(src)
        if seed_params:
            self._seed_params[func_name] = seed_params

        try:
            unseeded, _messages, _has_seed = RandomnessDetector().analyze_code(src)
        except Exception:  # pragma: no cover - detector must never break caching
            logger.debug("randomness scan failed for %s", func_name, exc_info=True)
            return

        if not unseeded:
            return

        call = unseeded[0]
        extra = ""
        if len(unseeded) > 1:
            extra = f" (+{len(unseeded) - 1} more unseeded call(s) in this function)"
        # ``call.lineno`` is relative to the source we handed the detector, which
        # starts at the function's first line. Rebase it onto the file so the
        # number in the message matches what the user's editor shows.
        # ``getsourcelines`` returns 0 for sources it cannot place; keep the
        # relative number rather than reporting a nonsense negative line.
        abs_lineno = call.lineno + first_lineno - 1 if first_lineno else call.lineno

        # ASCII only: this lands in a terminal whose codepage may not be UTF-8.
        message = (
            f"@cash.cache on {func_name}: Unseeded randomness detected: "
            f"{describe_random_call(call)} at line {abs_lineno}{extra}. "
            f"The first call's result is cached and replayed on every later "
            f"call - the RNG is never consulted again, so the value is frozen "
            f"and not reproducible across a cleared cache."
        )
        # ``_warn_once`` keys on (category, func_name, "") -> one warning per
        # decorated function for the life of this Cash instance, and it also
        # files the message into ``f.cache_info()['warnings']`` so it stays
        # discoverable if the user missed the stderr emission.
        self._warn_once(
            CashRandomnessWarning, func_name, "", message,
            code="RANDOM-UNSEEDED",
            fix="seed the RNG to make the value reproducible, leave the "
                "function undecorated for a genuinely fresh draw, or pass "
                "@cash.cache(allow_random=True) to keep it frozen on purpose.",
            # 4, not 3: the chain from ``warnings.warn`` is
            # ``warn_diagnostic_message -> _warn_once ->
            # _warn_unseeded_randomness -> cache -> user``, so 3 blamed
            # ``cache`` itself and printed a line inside core.py. Measured
            # against a decoration on a known line; a reader whose warning
            # points into Cash cannot act on it, which is the whole point of
            # this diagnostic. Single caller, so the depth is fixed.,
        )

    def _warn_if_seed_is_none(self, func: Callable, func_name: str,
                              args: tuple, kwargs: dict) -> None:
        """RANDOM-UNSEEDED for a seed that is None in THIS call.

        The seed may be a parameter (CAS-116) or read from one or from a module
        global: ``default_rng(settings.seed)`` with the field None froze one
        draw across processes and said nothing (round 18), while the bare
        ``seed=None`` parameter warned.
        """
        bound = None
        g = getattr(func, "__globals__", None) or {}
        for expr, (call, root, is_param, path) in sorted(
                self._seed_params.get(func_name, {}).items()):
            if is_param:
                if bound is None:
                    try:
                        bound = inspect.signature(func).bind(*args, **kwargs)
                        bound.apply_defaults()
                    except (TypeError, ValueError):
                        return
                if root not in bound.arguments:
                    continue
                value = _read_seed(bound.arguments[root], path)
                origin = f"the parameter '{root}'" if not path else f"'{expr}'"
            else:
                if root not in g:
                    continue
                value = _read_seed(g[root], path)
                origin = f"'{expr}'"
            if value is not None:
                continue
            fix = (f"pass a seed: {root}=i per replicate keeps each one reproducible "
                   f"and cacheable." if is_param and not path else
                   f"set {expr} to an integer, or pass the seed as an argument.")
            self._warn_once(
                CashRandomnessWarning, func_name, f"seed-param:{expr}",
                f"@cash.cache on {func_name}: {call} is seeded from "
                f"{origin}, which is None in this call, so the RNG "
                f"draws from OS entropy. The first call's result is cached and "
                f"replayed on every later call with the same arguments -- "
                f"repeated calls return the same 'random' value, and it is "
                f"not reproducible across a cleared cache.",
                code="RANDOM-UNSEEDED",
                fix=f"{fix} Or @cash.cache(allow_random=True) to keep the value "
                    f"frozen on purpose.",
            )
            return

    def _warn_unseeded_estimator_result(
        self, func_name: str, result: Any, allow_random: bool,
    ) -> None:
        """Warn when a cached function RETURNS an unseeded fitted estimator.

        ``_warn_unseeded_randomness`` reads the source, and
        ``decorator.md`` is right that this hazard is invisible to it:
        randomness inside sklearn's compiled ``.fit()`` is not in any AST. The
        notebook's statement path solves that by asking the LIVE object
        (``get_params()['random_state'] is None``) rather than the source; the
        decorator path had no equivalent, so the recommended way to cache a fit
        was also the silent one.

        Reported in round 14: three runs returned the identical model (first
        tree's `random_state` 1200527474), no warning, no badge marker, using
        the docs' own recipe. The tester's words for the harm are the reason
        this exists -- "I would have written 'the model is completely stable
        across random seeds' in a report."

        Same verdict rule as ``_unseeded_estimator_fits``: unseeded iff
        ``get_params()`` HAS ``random_state`` and it is ``None``. A seed of any
        kind, or no such parameter at all (``LinearRegression``), is silent.
        Any failure is silent too -- an advisory must never break a call.
        """
        if allow_random:
            return
        # This runs on EVERY call, hits included, so it must stay cheap once it
        # has had its say. `_warn_once` would dedupe the emission but not the
        # `get_params()` that precedes it, and sklearn's `get_params` walks the
        # signature -- a per-hit cost on exactly the functions people cache to
        # avoid paying for a fit. Check the same key first and leave.
        if (CashRandomnessWarning, func_name, "_estimator_result") in \
                self._warning_keys_seen:
            return
        get_params = getattr(result, "get_params", None)
        if get_params is None or not callable(get_params):
            return
        try:
            params = get_params()
            if params.get("random_state", "absent") is not None:
                return
        except Exception:  # noqa: BLE001 - advisory only; never break a call
            return

        self._warn_once(
            CashRandomnessWarning, func_name, "_estimator_result",
            f"@cash.cache on {func_name}: returns a fitted estimator with "
            f"random_state=None. cash caches it, so every later call replays "
            f"that one fit - the model is frozen, not stable. Two genuine fits "
            f"would differ, and comparing runs cannot tell you otherwise.",
            code="RANDOM-UNSEEDED",
            fix="pass random_state=<int> to the estimator for a reproducible "
                "fit, leave the function undecorated for a genuinely fresh "
                "one, or pass @cash.cache(allow_random=True) to keep it frozen "
                "on purpose.",
        )

    def _warn_once(
        self,
        category: type[Warning],
        func_name: str,
        arg_type_name: str,
        message: str,
        *,
        code: str,
        fix: str,
        stacklevel: int | None = None,
    ) -> None:
        """Emit a coded diagnostic at most once per
        ``(category, func_name, arg_type_name)`` for this Cash instance.

        ``message`` is one sentence of *what happened*; ``fix`` is one
        imperative sentence; ``code`` is the diagnostic code from
        ``cash.diagnostics`` that names the section of ``docs/warnings.md``
        expanding both. The three are rendered together by
        :func:`~cash.diagnostics.format_diagnostic`, and the rendered text is
        what reaches BOTH stderr and ``cache_info()['warnings']`` -- the log
        and the terminal must not drift apart, since the log is where people
        look once the stderr line has scrolled away.

        ``arg_type_name`` is the empty string for warnings that do not
        attach to a specific arg type (e.g. store-failed). The seen-set
        key still distinguishes by func_name.

        **Do not pass ``stacklevel``.** The blamed frame is resolved at emit
        time by walking out to the nearest frame outside ``cash/`` -- see
        :func:`~cash.diagnostics._stacklevel_of_first_user_frame`. This used to
        be a per-caller constant, documented here as 5 by default with 6 and 3
        for the deeper and shallower chains, and four separate diagnostics
        shipped pointing at a line inside ``core.py`` anyway. A constant cannot
        be right for a helper reached at two different depths, and an over-deep
        one reports ``<sys>:0`` rather than clamping, so the failure was silent
        in both directions. The parameter survives only as an override for a
        site that needs one; none does.
        """
        rendered = format_diagnostic(code, message, fix)   # raises on a bad code
        key = (category, func_name, arg_type_name)
        with self._decorator_call_log_lock:
            if key in self._warning_keys_seen:
                return
            self._warning_keys_seen.add(key)
            # Also record in per-function rolling log so the warning is
            # discoverable after the fact via ``f.cache_info()['warnings']``
            # - even if the user missed the stderr emission. The code goes in
            # as its own field as well as inside the text, so a reader of the
            # log can branch on it the way a warning handler branches on
            # ``w.message.code``.
            entry = {
                'category': category.__name__,
                'code': code,
                'message': rendered,
                'timestamp': time.time(),
            }
            log = self._func_warnings.setdefault(func_name, [])
            log.append(entry)
            if len(log) > self._func_warnings_max:
                del log[: len(log) - self._func_warnings_max]
        warn_diagnostic_message(category, code, rendered, stacklevel=stacklevel)

    def drain_decorator_calls(self) -> list[dict[str, Any]]:
        """Return and clear all recorded decorator call events.

        Thread-safe: atomically copies and clears the log.
        Called by the notebook statement processor after executing a statement
        to collect decorator-level cache metrics for badge display.

        Returns:
            List of call event dicts, each with keys:
            ``func_name``, ``cache_hit``, ``execution_time``, ``time_saved``,
            ``args_hash``, ``cache_key``, ``timestamp``.

            ``execution_time`` is what this call cost; ``time_saved`` is the
            recorded cost of the original computation a hit avoided, so it is
            an estimate carried forward from the write, not a measurement of
            this call.
        """
        with self._decorator_call_log_lock:
            calls = list(self._decorator_call_log)
            self._decorator_call_log.clear()
        return calls

    def register_hasher(
        self,
        type_: type,
        hasher_fn: Callable[[Any], str],
        *,
        override: bool = False,
    ) -> None:
        """Register a custom hasher for a specific type.

        When ``_serialize_args`` encounters an argument of ``type_``, it will
        call ``hasher_fn(value)`` to produce a hash string instead of relying
        on ``pickle.dumps``.

        Args:
            type_: The Python type to register a hasher for.
            hasher_fn: A callable that takes a value of ``type_`` and returns
                a deterministic hash string.
            override: Take precedence over cash's own content hashers.
                Needed only for the types cash fingerprints itself -- numpy
                arrays, pandas / polars / PyArrow / modin frames, dask
                collections -- where those run first. A plain registration
                for one of those types is REJECTED with ``ValueError``: it
                could not have done anything, and saying so at setup beats
                leaving the user to discover that nothing got faster.

                Off by default because the built-ins read every byte, and a
                hasher that does not can return a *wrong* cached result
                rather than a slow one. Passing it says you accept that: what
                you return is the entire identity of the value, and two
                values sharing it share an entry. That is the right trade
                when you hold a version, a content id, or an immutable
                fingerprint the array itself does not carry -- and the wrong
                one for ``lambda a: a[0, 0]``.

                Overriding hashers are consulted before everything else,
                including a notebook value's lineage hash.

        Example:

            import pandas as pd
            from cash import Cash

            c = Cash()
            c.register_hasher(
                pd.DataFrame,
                lambda df: hashlib.sha256(
                    pd.util.hash_pandas_object(df).values.tobytes()
                ).hexdigest()
            )

            Note: when ``hasher_fn`` is a callable object (an instance
            with ``__call__``), the source hash is derived from the
            class's ``__call__.__code__`` - so two instances of the
            same callable class share a source hash, even if they hold
            different per-instance state. If your hasher's behavior
            depends on instance state, prefer a function or lambda
            that closes over the state explicitly.
        """
        # Rejected BEFORE anything is mutated, so a refused call leaves an
        # earlier good registration for this type exactly as it was.
        if not override:
            family = self.builtin_hashed_family(type_)
            if family is not None:
                raise ValueError(
                    f"cash.register_hasher({type_.__name__}): cash fingerprints "
                    f"{family} values itself and that runs first, so this hasher "
                    f"would never be called -- registering it does nothing at "
                    f"all.\n"
                    f"Either drop the registration (cash already hashes this "
                    f"type by content, correctly), or pass override=True to use "
                    f"yours instead.\n"
                    f"Be deliberate about override: what your hasher returns "
                    f"becomes the entire identity of the value, so any two "
                    f"values it hashes alike share one cache entry and the "
                    f"second call gets the first one's result. That is right "
                    f"when you hold a version or content id the value does not "
                    f"carry, and wrong for something like lambda a: a[0, 0]."
                )

        # Allowed -- a hasher that returns what the function captures is
        # correct -- but the one people write is keyed on the name, and that
        # hands one closure's cached result to the next.
        if isinstance(type_, type) and issubclass(type_, _CODE_VALUE_TYPES):
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "KEY-CALLABLE-HASHER",
                f"cash.register_hasher({type_.__name__}, ...) decides the "
                f"identity of every {type_.__name__} passed to a cached "
                f"function in this process. Closures one factory makes share a "
                f"name and a body, so a hasher that does not return what they "
                f"capture gives them one cache entry, and the second gets the "
                f"first one's result.",
                "prefer passing the captured values to the cached function as "
                "plain arguments, with a module-level function in the closure's "
                "place; if you keep this hasher, make it return the captured "
                "values too.",
            )
        src_hash = self._hash_callable_source(hasher_fn)
        # One type, one registration: re-registering must not leave the
        # previous entry behind in the other registry, still winning.
        self._type_hashers.pop(type_, None)
        self._override_hashers.pop(type_, None)
        if override:
            self._override_hashers[type_] = (hasher_fn, src_hash)
        else:
            self._type_hashers[type_] = (hasher_fn, src_hash)
        # A memoized hash was produced by whichever hasher was in effect
        # before this call; drop them so the new registration is not shadowed
        # for objects already seen.
        self._arg_hash_memo.clear()
        self._frame_memo.clear()

    #: Types whose code must not participate in any cache key. Process-wide,
    #: not per-instance: a marker is a property of the type, and a user who
    #: marks it once should not have to repeat it per Cash instance.
    #:
    #: Holds STRONG references deliberately, so a registered class can never
    #: be garbage collected. Considered and rejected a WeakSet: opaque types
    #: are registered by hand, at import time, in the tens at most for any
    #: real user -- not generated in volume -- so the leak this trades away
    #: has no realistic scale to bite at. A WeakSet would also silently
    #: un-register a type the moment nothing else references it, which is
    #: the opposite of "mark it once and forget about it."
    _OPAQUE_TYPES: set = set()

    @staticmethod
    def mark_opaque(*types_: type) -> None:
        """Exclude *types_* from code-surface hashing.

        For a class you cannot or should not edit -- third-party, generated, or
        simply not yours. For one you own, ``@cash.opaque`` is the same thing
        spelled declaratively.
        """
        Cash._OPAQUE_TYPES.update(types_)

    @staticmethod
    def _is_opaque(obj: Any) -> bool:
        """True when *obj* -- a class, or an instance of one -- must not have
        its code hashed into a cache key.

        Checks two independent marks, both EXACT-MATCH on the type itself,
        deliberately not inheritance-aware:

        * ``_OPAQUE_TYPES`` (``mark_opaque``) is a plain set. Registering a
          base class does not implicitly cover a subclass the caller never
          passed to ``mark_opaque`` -- sets have no notion of "and its
          descendants."
        * ``__cash_opaque__`` (``@cash.opaque``) is read from the target's
          OWN ``__dict__`` via ``vars()``, not via plain ``getattr``.
          ``getattr`` walks the MRO, so a subclass would inherit the mark
          from an opaque ancestor even though the subclass may carry its
          own freshly-written methods the user actively edits -- silently
          exempting THAT code from ever invalidating the cache, for a
          decision made about a different class entirely. Matching
          ``_OPAQUE_TYPES``'s exact-match semantics here also keeps the two
          spellings equivalent, as documented: "the same thing spelled
          declaratively" should behave the same, not diverge on
          inheritance because one happens to be implemented as a dunder
          attribute. A subclass that wants the same treatment marks
          itself; see ``test_a_subclass_of_an_opaque_class_does_not_
          inherit_opacity`` for the pinned case.

        Never raises. Measured, not assumed: a metaclass that defines
        ``__eq__`` without ``__hash__`` makes the CLASS ITSELF unhashable
        (Python's data-model default, not just its instances), so
        ``target in Cash._OPAQUE_TYPES`` can raise ``TypeError`` on a real,
        if unusual, class shape. An opacity check must not be the thing
        that breaks an otherwise-cacheable call.
        """
        try:
            if isinstance(obj, functools.partial):
                # A partial is the function it wraps plus arguments, both of
                # which are keyed now. `mark_opaque(functools.partial)` was the
                # old advice for silencing KEY-OPAQUE-CALLABLE, and it silenced
                # EVERY partial in the process, including ones over code the
                # user then edited (round 18).
                return False
            target = obj if isinstance(obj, type) else type(obj)
            if target in Cash._OPAQUE_TYPES:
                return True
            return bool(vars(target).get("__cash_opaque__", False))
        except Exception as e:  # noqa: BLE001 - opacity check must never break a call
            logger.debug("[CORE] opacity check failed for %r: %s", obj, e)
            return False

    def _learn_mutating_captures(self, func: Callable, func_name: str,
                                 watched: dict[str, tuple[str, str]]) -> None:
        """Demote any provisional global this call was OBSERVED to mutate.

        A global merely *passed to a call* (`sum(G)`, `model.predict(G)`) used to
        be dropped from the key outright, on the theory that the callee might
        mutate it. That silently served stale values forever (CAS-270). Those
        names are folded now, and confirmed here: hash them again once the body
        has run and compare against the hash the key already needed.

        Changed across the call => calling this function is what moves the value,
        so folding it would key the entry on the function's own output and miss
        forever. Stop folding that ONE name; the function keeps caching on
        everything else.

        Two things worth knowing:

        * The entry just written stays valid -- it is keyed on the PRE-call
          state, which is what produced it. The next call keys without this
          name, misses once, and thereafter behaves as it did before CAS-270.
        * A change *between* calls (`G = [...]` anywhere) is invisible to this
          window by construction, which is correct: that is precisely what
          folding is for, and it needs no detection.

        Only runs on a miss -- the body has to execute for there to be anything
        to observe -- so the cost is one hash on the path that just paid for a
        real computation.
        """
        if not watched:
            return
        code = getattr(func, "__code__", None)
        if code is None:
            return
        own_globals = getattr(func, "__globals__", None)
        cells = dict(zip(getattr(code, "co_freevars", ()) or (),
                         getattr(func, "__closure__", ()) or ()))
        for name, (before, scope, owner) in watched.items():
            try:
                if scope == "closure":
                    cell = cells.get(name)
                    if cell is None:
                        continue
                    after = self._hash_arg_payload((cell.cell_contents,), {})
                else:
                    # The mapping the BEFORE hash came from -- a helper's
                    # module, when this entry was folded on a helper's behalf.
                    g = owner if isinstance(owner, dict) else own_globals
                    if not isinstance(g, dict) or name not in g:
                        continue
                    after = self._hash_arg_payload(
                        (self._stabilize_for_global_hash(
                            g[name], self._hash_callable_source),), {})
            except Exception:  # noqa: BLE001 - unhashable NOW; treat as unchanged
                continue
            if after == before:
                continue
            self._mutating_globals.setdefault((code, scope), set()).add(name)
            where = ("variable it captures" if scope == "closure"
                     else "module global")
            self._warn_once(
                CashImpurityWarning,
                func_name,
                name,
                f"@cash.cache on {func_name}: calling it modifies the {where} "
                f"'{name}', so '{name}' can no longer be tracked for "
                f"invalidation and a cache hit will not repeat that change.",
                code="IMPURE-SCOPE-MUTATION",
                fix="pass the value in as an argument and return the new one, "
                    "instead of reaching out and rewriting it.",
            )

    def _refuses_identity_coupled(self, func_name: str, result: Any) -> bool:
        """True when *result* must never be stored, because storing it would
        detach a library's global registry from the object the caller holds.

        The statement path (``statement/processor.py``) and call interception
        (``call_unit.py``) have gated on ``identity_coupled_reason`` for a
        while; the decorator did not.  So ``@cash.cache`` on a function
        returning a ``Figure`` hijacked ``plt.gcf()`` -- on the FIRST call,
        during the *store*, because the RAM tier deep-copies and
        ``Figure.__setstate__`` re-registers the copy as pyplot's current
        figure.  The user then draws on their figure while ``plt.savefig()``
        writes the cache's private snapshot (CAS-245).

        Checked here rather than inside ``_store_in_cache`` so the refusal
        lands beside ``cache_if``, BEFORE ``_attach_lineage``: a value that is
        not stored must not carry a lineage hash pointing at an entry that was
        never written.

        KNOWN BOUNDARY: called at all four store sites (sync/async x
        non-iterator/single-chunk), which is every site where the value is in
        hand before anything is written.  A *multi*-chunk iterator is not
        covered -- ``_stream_and_store`` has already written earlier chunks by the
        time any item could be inspected, so gating there would mean aborting
        mid-write and reclaiming them.  Reaching it needs a generator yielding
        enough Figures to cross ``chunk_max_bytes`` (or a million of them),
        which no reported case comes near.  Widen this if one ever does.
        """
        # Local import: ``cacheability_decision`` pulls in the annotation and
        # AST-analysis modules, and this runs only on a miss's store path.
        # ``core`` -> ``cash.notebook`` is an established direction (see the
        # module-level CodeAnalyzer / parse_annotation_line imports), so no
        # shared module is needed for this.
        from cash.notebook.cacheability_decision import identity_coupled_reason

        # ``func_name`` is already in the message prefix, so name the slot
        # rather than repeating the qualified path inside the reason.
        reason = identity_coupled_reason("the returned value", result)
        if reason is None:
            return False
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: result not cached. {reason}",
            code="CACHE-IDENTITY-COUPLED",
            fix="split the function: cache the part that computes the numbers, "
                "and draw the figure from them in an uncached function.",
        )
        return True

    def _note_arg_cost(self, func_name: str) -> None:
        """Keep the costliest argument to hash seen for *func_name*.

        Only its description is kept -- parameter, type, seconds, the cached
        function that produced it -- never the value, which may be large.
        """
        cost = getattr(_ARG_COST, "last", None)
        _ARG_COST.last = None
        if cost is None:
            return
        label, seconds, type_name, producer, old_pandas = cost
        known = self._arg_costs.get(func_name)
        if known is not None and known[2] >= seconds:
            return
        if known is None and len(self._arg_costs) >= 1024:
            return
        self._arg_costs[func_name] = (label, type_name, seconds, producer, old_pandas)

    def _note_effectiveness(
        self,
        func_name: str,
        overhead_seconds: float,
        *,
        body_seconds: float | None,
        was_hit: bool,
    ) -> None:
        """Account for one call and warn if caching has become a net loss.

        Informational only: the decorator caches because the user asked it
        to, and deciding otherwise is the notebook cost model's job, not
        this one. See ``cash.effectiveness`` for when it speaks up.

        ``overhead_seconds`` covers cache-key construction and lookup -- the
        per-call costs. The one-off store is excluded, deliberately: it is
        paid once per key rather than per call, and the workload this exists
        to catch is one that pays a large key cost on every single call.
        """
        try:
            verdict = self._effectiveness.record(
                func_name,
                overhead_seconds=overhead_seconds,
                body_seconds=body_seconds,
                was_hit=was_hit,
                culprit=self._arg_costs.get(func_name),
            )
        except Exception:  # noqa: BLE001 - accounting must never break a call
            return
        # The warn is deliberately OUTSIDE that guard. Under ``-W error`` it
        # raises into the caller -- which is the shape this project spent
        # 8b47cc4 removing from the backend, so it is worth being explicit
        # that it is different here: there, cash raised on its own initiative
        # over a failure the user had not asked to hear about. Here the user
        # configured warnings-as-errors and is entitled to have that honoured.
        # Swallowing it would silently override their filter, which is worse.
        if verdict:
            what, fix = verdict
            warn_diagnostic(
                CashCacheIneffectiveWarning, "CACHE-NET-LOSS", what, fix,
            )

    def _store_in_cache(
        self,
        cache_key: str,
        func_name: str,
        result: Any,
        metadata: dict[str, Any] | None,
        ttl: int | None,
        state_hash: str,
        args_hash: str,
        execution_time: float = 0.0,
        auto_file_deps: dict[str, dict[str, float]] | None = None,
        body_seconds: float | None = None,
    ) -> None:
        try:
            serializer = get_serializer(result)

            meta = CacheMetadata(
                key=cache_key,
                func_name=func_name,
                timestamp=time.time(),
                # The decorator's measured wall-clock cost. ``TieredBackend``
                # reads this to decide whether the value is expensive enough
                # to promote past RAM (otherwise the smart-persistence
                # policy gates everything at the 0.1s floor, and script
                # runs that recompute the same cheap value forever).
                execution_time=execution_time,
                # The body's own cost, so a later HIT can tell what it
                # actually saved. execution_time cannot answer that: it is
                # measured from the top of the wrapper and includes the
                # key hashing whose worth is the question.
                body_seconds=body_seconds,
                serializer_cls=type(serializer),
                ttl=ttl,
                args_hash=args_hash,
                state_hash=state_hash,
                # Each entry: path -> {'mtime': float, 'size': int}.
                # Validated on subsequent get() via _auto_file_deps_fresh.
                auto_file_deps=auto_file_deps or None,
            )

            # Kept, not a temporary: TieredBackend writes back where the value
            # landed, and "RAM only" is the answer to the next process's miss.
            meta_dict = meta.to_dict()
            self.backend.set(cache_key, result, meta_dict, serializer=serializer)
            not_persisted = self._not_persisted_reason(meta_dict, execution_time)
            self._remember_outcome(cache_key, {
                "stored_at": time.time(),
                "ttl": ttl,
                "not_persisted": not_persisted,
            })
            if not_persisted is None:
                self._record_stored_key(func_name, cache_key, ttl)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            self._note_not_stored(cache_key, "the backend refused the write")
            backend_name = type(self.backend).__name__
            self._warn_once(
                CashCacheStoreFailedWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: backend {backend_name} failed to store "
                f"result ({type(e).__name__}: {e}). Compute succeeded, nothing was "
                f"stored, and the next call recomputes.",
                code="STORE-FAILED",
                fix=_STORE_FAILED_FIX,
            )

    def _warn_cache_if_bypassed(self, func_name: str, chunk_max_items: int,
                                chunk_max_bytes: int, stacklevel: int | None = None) -> None:
        """One-shot: the result outgrew a single chunk, so cache_if cannot run.

        Applying it would mean materializing every chunk back into memory,
        undoing the bound that chunking exists to provide.

        The fix line says **raise** the thresholds, and that direction is
        load-bearing. ``cache_if`` is consulted only in the ``chunk_index == 0``
        branch of ``_stream_and_store`` -- the whole result fit one chunk -- and
        this fires at ``chunk_index == 1``, once it did not. The message used to
        advise *lowering* the thresholds, which produces more chunks and so
        guarantees the very bypass it is warning about.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: the result exceeded a single chunk "
            f"(chunk_max_items={chunk_max_items}, "
            f"chunk_max_bytes={chunk_max_bytes}), so it was cached without "
            f"cache_if ever being consulted.",
            code="CACHE-IF-BYPASSED",
            fix="raise chunk_max_items / chunk_max_bytes above the size this "
                "result reaches, or return a list instead of an iterator, so "
                "the whole result arrives in one piece for the predicate to "
                "see.",
            stacklevel=stacklevel,
        )

    def _stream_and_store(
        self, source, *, cache_key, func_name, metadata, tracker, observer,
        rng_new, args, kwargs, args_hash, current_state_hash, ttl, cache_if,
        chunk_max_items, chunk_max_bytes, code_module=None,
    ):
        """Yield the producer's items as they come, and cache once it ends.

        Three things have to hold at once, and they are why this is not simply
        a `yield` added to a drain-then-store loop:

        **The tracker watches production, never consumption.** A generator's
        file reads happen while it is consumed, so the tracker has to be live
        across `next()`; leaving it live across the caller's `yield` would
        attribute the CALLER's reads to this function. It is entered once and
        suspended around each yield, which records the producer's lazy reads
        and nothing else.

        **Nothing is stored unless the producer ran to exhaustion.** A caller
        that breaks out, or a producer that raises, leaves no entry -- a
        truncated result under the full result's key is a wrong answer, not a
        slow one. Chunks already flushed are removed on the way out. The cost
        is real and accepted: draining eagerly used to leave a complete entry
        behind even when the caller took two items, and it no longer does.

        **Time is the producer's, not the wall clock.** Only the spans inside
        `next()` are summed, so a slow consumer cannot inflate the number the
        persistence decision reads.
        """
        from cash.notebook.object_hashing import estimate_object_size

        buffer: list[Any] = []
        buffer_bytes = 0
        chunk_index = 0
        total_items = 0
        produced_seconds = 0.0
        committed = False

        try:
            # Entered ONCE. Per item we only suspend around the `yield`, which
            # is a ContextVar swap; `__enter__` reinstalls patches and rechecks
            # the import hook, and paying that per item cost 5.1us each --
            # measured 294ms -> 1521ms on a 200k-item iterator before this.
            with tracker, observer:
                while True:
                    started = time.perf_counter()
                    try:
                        item = next(source)
                    except StopIteration:
                        produced_seconds += time.perf_counter() - started
                        break
                    produced_seconds += time.perf_counter() - started

                    buffer.append(item)
                    buffer_bytes += estimate_object_size(item)
                    total_items += 1
                    if len(buffer) >= chunk_max_items or buffer_bytes >= chunk_max_bytes:
                        if chunk_index == 1 and cache_if is not None:
                            self._warn_cache_if_bypassed(
                                func_name, chunk_max_items, chunk_max_bytes)
                        self._write_one_chunk(cache_key, chunk_index, buffer, ttl=ttl,
                                          execution_time=produced_seconds)
                        buffer = []
                        buffer_bytes = 0
                        chunk_index += 1

                    # The caller's own reads and effects are its own.
                    tracker_token = tracker.suspend()
                    observer_token = observer.suspend()
                    try:
                        yield item
                    finally:
                        observer.resume(observer_token)
                        tracker.resume(tracker_token)

            self._check_argument_mutation(func_name, args, kwargs, args_hash, observer)
            self._report_observed_effects(func_name, observer)
            auto_file_deps = self._snapshot_tracked_deps(tracker, code_module)

            if chunk_index == 0:
                # Everything fit in one chunk, so cache_if can still see the
                # whole result -- it gates STORAGE, never what the caller
                # already received.
                refusal = self._store_refusal(
                    None, func_name, buffer, rng_new, cache_if, tracker)
                if refusal is not None:
                    self._note_not_stored(cache_key, refusal)
                else:
                    if buffer:
                        self._write_one_chunk(cache_key, 0, buffer, ttl=ttl,
                                              execution_time=produced_seconds)
                    # An empty iterator still gets a zero-chunk manifest, so a
                    # hit returns empty instead of recomputing.
                    self._store_chunked_manifest(
                        cache_key, func_name,
                        {"n_chunks": 1 if buffer else 0, "total_items": total_items},
                        metadata, ttl, current_state_hash, args_hash,
                        produced_seconds, auto_file_deps,
                    )
            else:
                if buffer:
                    if chunk_index == 1 and cache_if is not None:
                        self._warn_cache_if_bypassed(
                            func_name, chunk_max_items, chunk_max_bytes)
                    self._write_one_chunk(cache_key, chunk_index, buffer, ttl=ttl,
                                          execution_time=produced_seconds)
                    chunk_index += 1
                self._store_chunked_manifest(
                    cache_key, func_name,
                    {"n_chunks": chunk_index, "total_items": total_items},
                    metadata, ttl, current_state_hash, args_hash,
                    produced_seconds, auto_file_deps,
                )

            committed = True
        finally:
            if not committed:
                # Abandoned or failed: the chunks written so far are
                # unreferenced (no manifest names them). Best effort -- a
                # killed process can still leave some behind.
                for index in range(chunk_index):
                    try:
                        self.backend.delete(f"{cache_key}:chunk_{index}")
                    except Exception:  # noqa: BLE001 - cleanup must not raise
                        logger.debug("[CORE] could not drop orphan chunk %d", index)

    def _write_one_chunk(
        self,
        cache_key: str,
        chunk_index: int,
        chunk_buffer: list[Any],
        ttl: int | None = None,
        execution_time: float = 0.0,
    ) -> None:
        """Write a single chunk to the backend.

        The chunk's metadata is minimal - the authoritative manifest
        lives at the canonical cache_key. We need *some* metadata for
        the serializer to round-trip correctly; the timestamp and the
        key are enough. We also propagate the manifest's ``ttl`` so
        ``Cash.cleanup()`` (without a ``max_age`` argument) can reclaim
        expired chunks alongside the expired manifest.
        """
        chunk_key = f"{cache_key}:chunk_{chunk_index}"
        serializer = get_serializer(chunk_buffer)
        chunk_metadata = CacheMetadata(
            key=chunk_key,
            timestamp=time.time(),
            serializer_cls=type(serializer),
            # The PRODUCER's time, not 0. A chunk is not an independent result
            # whose worth gets decided on its own -- it is the payload of an
            # entry whose durability was already decided. Writing 0 here sent
            # every chunk under the smart-persistence compute floor, so chunks
            # stayed RAM-only while the manifest (carrying the real time) went
            # to disk. A fresh process then found a manifest with no chunks
            # behind it and, because a missing chunk terminates iteration,
            # returned an EMPTY iterator. Silent data loss, and the cross-
            # process hit is the entire point of caching a generator.
            execution_time=execution_time,
            ttl=ttl,
        ).to_dict()
        try:
            self.backend.set(chunk_key, chunk_buffer, chunk_metadata, serializer=serializer)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            backend_name = type(self.backend).__name__
            self._warn_once(
                CashCacheStoreFailedWarning,
                f"{cache_key}:chunk_{chunk_index}",
                "",
                # NOT "you will get a truncated iterator". ``_chunks_are_intact``
                # probes every chunk and turns a manifest with a hole into a
                # MISS, on both read paths, so the cost is a permanent recompute
                # rather than a short answer. The locked re-read skipped that
                # guard until 2026-09-06; do not re-introduce the truncation
                # language here without first checking it is true again.
                f"@cash.cache: backend {backend_name} failed to store "
                f"chunk {chunk_index} of {cache_key} ({type(e).__name__}: {e}), "
                f"so the entry can never be read back and every later call "
                f"recomputes it.",
                code="STORE-CHUNK-FAILED",
                fix="clear the entry with f.cache_clear() before anything reads "
                    "it, then fix the write the exception names.",
            )

    def _store_chunked_manifest(
        self,
        cache_key: str,
        func_name: str,
        manifest_data: dict[str, Any],
        existing_metadata: dict[str, Any] | None,
        ttl: int | None,
        state_hash: str,
        args_hash: str,
        execution_time: float,
        auto_file_deps: dict[str, dict[str, float]] | None,
    ) -> None:
        """Write the manifest entry for a chunked iterator at *cache_key*.

        The value stored at the key is the manifest dict (``n_chunks``,
        ``total_items``). The metadata flags this entry as chunked so
        the hit path knows to use ``_ChunkedCachedIterator``.
        """
        try:
            serializer = get_serializer(manifest_data)
            metadata = CacheMetadata(
                key=cache_key,
                func_name=func_name,
                timestamp=time.time(),
                execution_time=execution_time,
                serializer_cls=type(serializer),
                ttl=ttl,
                args_hash=args_hash,
                state_hash=state_hash,
                iterator_storage="chunked",
                n_chunks=manifest_data["n_chunks"],
                auto_file_deps=auto_file_deps or None,
            ).to_dict()
            self.backend.set(cache_key, manifest_data, metadata, serializer=serializer)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            backend_name = type(self.backend).__name__
            self._warn_once(
                CashCacheStoreFailedWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: backend {backend_name} failed "
                f"to store chunked manifest ({type(e).__name__}: {e}). "
                f"Compute succeeded, nothing was stored, and the next call "
                f"recomputes.",
                code="STORE-FAILED",
                fix=_STORE_FAILED_FIX,
            )

    def cleanup(self, max_age: int | None = None) -> int:
        """Remove expired items from the cache.

        Args:
            max_age: If provided, remove items older than *max_age* seconds,
                regardless of their stored TTL.

        Returns:
            Number of entries removed.
        """
        now = time.time()

        def is_expired(raw_metadata):
            try:
                metadata = CacheMetadata.from_dict(raw_metadata)
                timestamp = metadata.timestamp or 0
                age = now - timestamp

                if max_age is not None and age > max_age:
                    return True

                stored_ttl = metadata.ttl
                return bool(stored_ttl is not None and age > stored_ttl)
            except (AttributeError, TypeError, ValueError):
                return True

        return self.backend.cleanup_expired(is_expired)

    def explorer(self) -> CacheExplorer:
        """Return a `CacheExplorer` instance for interactive cache browsing."""
        from .ui.explorer import CacheExplorer
        return CacheExplorer(self)

    def run_summary(self) -> str:
        """A per-function hit/miss table for this process, or ``""``.

        Empty when no cached function was ever called, so a caller can print
        this unconditionally without emitting a header over nothing.
        """
        rows = [(name, s) for name, s in self._function_stats.items()
                if s['hits'] or s['misses']]
        bypassed = sum(s.get('bypassed', 0) for s in self._function_stats.values())
        disabled_line = (
            f"cash: caching disabled (disable=True / CASH_DISABLE) -- {bypassed} "
            f"call{'' if bypassed == 1 else 's'} ran uncached" if bypassed else "")
        if not rows:
            return disabled_line
        rows.sort(key=lambda r: r[1]['total_time_saved'], reverse=True)

        hits = sum(s['hits'] for _, s in rows)
        calls = hits + sum(s['misses'] for _, s in rows)
        saved = sum(s['total_time_saved'] for _, s in rows)
        width = min(44, max(len(name) for name, _ in rows))

        def _fit(name: str) -> str:
            # Keep the TAIL. A name is ``module.qualname``, so for anything
            # nested -- a closure, a method, a test helper -- the part that
            # identifies it is at the end, and truncating from the right threw
            # away the function name and kept the package path.
            return name if len(name) <= width else "..." + name[-(width - 3):]

        lines = [f"cash: {hits} of {calls} calls restored, {saved:.1f}s saved"]
        where = self._summary_cache_dir()
        if where:
            # Which directory this ran against. A script user has no badge and
            # no other place to see it, and every question that starts "why is
            # nothing cached" is answered or excluded by this one line: a
            # scheduled job's cwd-relative cache, a path typed with one
            # backslash too few, a container volume that is not the one they
            # meant. A tester spent a round on a cache directory that was not
            # the one they thought they had set.
            lines.append(f"  cache: {where}")
        for name, stat in rows:
            # Pad the whole "N hits," token, not the word: padding the word
            # puts the space before the comma ("1 hit ,").
            hit_col = f"{stat['hits']} {'hit' if stat['hits'] == 1 else 'hits'},"
            miss_col = f"{stat['misses']} {'miss' if stat['misses'] == 1 else 'misses'}"
            saved_col = (f"{stat['total_time_saved']:.1f}s saved"
                         if stat['total_time_saved'] else "-")
            lines.append(f"  {_fit(name):<{width}}  {hit_col:<10}"
                         f"{miss_col:<12}{saved_col}")
            lines.extend(self._summary_reasons(stat))
        if disabled_line:
            lines.append("  " + disabled_line)
        return "\n".join(lines)

    @staticmethod
    def _summary_reasons(stat: dict[str, Any]) -> list[str]:
        """The indented lines under a summary row: why it missed, what stayed.

        "1 miss" was the whole story before, and it hid the common surprise:
        a result computed in 0.05 s is never written to disk, so every new
        process misses it. The run that CAUSES that is the one that can say so.
        """
        out = []
        reasons = stat.get('miss_reasons') or {}
        if reasons:
            out.append("      missed: " + ", ".join(
                f"{n} {kind}" for kind, n in sorted(reasons.items(), key=lambda r: -r[1])))
        for why, n in (stat.get('not_stored') or {}).items():
            out.append(f"      not stored ({n}x): {why}")
        for why, n in (stat.get('not_persisted') or {}).items():
            out.append(f"      kept in RAM only ({n}x): {why}; "
                       f"a new process recomputes it")
        return out

    def _summary_cache_dir(self) -> str | None:
        """The cache directory this instance is using, for the summary header.

        Reads the ALREADY-BUILT backend when there is one and falls back to the
        configured path otherwise: the summary must never be the thing that
        creates a cache directory, and it runs from an ``atexit`` handler where
        building one is worse than saying nothing.
        """
        backend = self._backend
        for candidate in (backend, *getattr(backend, "backends", ())):
            path = getattr(candidate, "cache_dir", None)
            if isinstance(path, str) and path:
                return path
        configured = getattr(self.config, "cache_dir", None)
        return configured if isinstance(configured, str) and configured else None

    def _print_run_summary(self) -> None:
        """``atexit`` hook for ``summary=True``. Must never raise.

        Interpreter shutdown tears modules down underneath handlers, so a
        diagnostic that explodes here would turn a finished run into a
        traceback the user cannot act on.
        """
        try:
            text = self.run_summary()
            if text:
                # stderr: stdout is the program's output -- a report, a pipe, a
                # JSON response -- and a summary landing in it broke all three
                # for round-17 testers.
                print(text, file=sys.stderr)
        except Exception:  # noqa: BLE001 - a summary must not fail a finished run
            pass

    def show_stats(self) -> None:
        """Display the interactive analytics dashboard.

        Requires IPython/Jupyter and ipywidgets. In script environments,
        prints the same per-function table ``summary=True`` prints at exit.
        """
        from .ui.dashboard import HAS_WIDGETS, show_analytics_dashboard
        # Asking the dashboard whether it CAN run, rather than calling it and
        # catching. It prints "ipywidgets is required" and returns normally, so
        # the except-ImportError fallback this replaces was unreachable: the
        # documented script behaviour never once happened.
        if HAS_WIDGETS:
            try:
                show_analytics_dashboard()
                return
            except (ImportError, RuntimeError):
                pass
        text = self.run_summary()
        print(text if text else "cash: no cached function has been called yet.")

    def register_magic(self) -> None:
        """Register IPython magic commands (``%cash_on``, ``%%cash``, etc.)."""
        try:
            from IPython import get_ipython
        except ImportError:
            logger.debug("IPython not available. Magic commands not registered.")
            return

        ip = get_ipython()
        if ip is None:
            logger.debug("No active IPython session found. Magic commands not registered.")
            return

        # Internal import - must always succeed when IPython is present.
        # Kept outside the ImportError guard above so a broken import path
        # surfaces loudly instead of masquerading as "IPython not available".
        from .notebook.ipython.magics import CashMagics

        magics = CashMagics(ip, self)
        ip.register_magics(magics)

    def clear_all(self) -> None:
        """Clear cached results for every function registered with this instance.

        Equivalent to calling ``f.cache_clear()`` on every ``@cash.cache``-decorated
        function. Resets hit/miss statistics and removes all backend entries.
        """
        for wrapped in self._wrapped_funcs.values():
            wrapped.cache_clear()

    def _analyze_dependencies(self, func: Callable[..., Any]) -> None:
        """Populate analysis for *func* + its transitive cached-dependency
        closure, then surface *func*'s own purity issues.

        Populating the WHOLE closure (not just *func*) before the first cache
        key is computed is what keeps the key stable from the very first call.
        The state hash folds in each dependency's purity-report
        ``helper_source_hashes``; those used to be filled lazily on each
        dependency's own first call, so the key deepened only after the chain
        warmed - and a fresh process therefore missed the first call to every
        cached function even though a valid entry was on disk (finding #7).

        Surfacing stays per-function: each dependency warns/raises on its OWN
        first direct call, not here, so eager population doesn't change which
        warnings fire or when.
        """
        self._ensure_closure_analyzed(func)
        func_name = self._get_func_key(func)
        report = self._purity_reports.get(func_name) or PurityReport()
        mode = self._purity_modes.get(func_name, "warn")
        self._surface_purity(func_name, report, mode)

    def _ensure_closure_analyzed(self, func: Callable[..., Any]) -> None:
        """Populate graph edges + purity reports for *func* and every cached
        function transitively reachable from it, WITHOUT surfacing warnings.

        Idempotent per source version (guarded by ``self._populated``). Always
        traverses the dependency edges - even when the root is already
        populated - so a dependency invalidated by a source edit gets
        re-populated. The local ``seen`` set bounds cyclic graphs.

        Under ``self._analysis_lock`` because the CACHE KEY is built from what
        this populates. Concurrent first calls otherwise resolved two different
        keys for one call -- the threads that arrived mid-population, and the
        one doing it -- which is what made ``use_locking=True`` look like it
        admitted exactly two threads at every thread count. Both paths into
        this must hold the lock: the wrapper's one-time analysis AND
        ``explain()``, which populates the closure directly and would otherwise
        race it back apart.
        """
        with self._analysis_lock:
            self._ensure_closure_analyzed_locked(func)

    def _ensure_closure_analyzed_locked(self, func: Callable[..., Any]) -> None:
        """The body of ``_ensure_closure_analyzed``, with the lock already held."""
        stack = [func]
        seen: set[str] = set()
        while stack:
            f = stack.pop()
            fname = self._get_func_key(f)
            if fname in seen:
                continue
            seen.add(fname)
            if fname not in self._populated:
                self._populate_analysis(f, fname)
            for dep in self.graph.get_dependencies(fname):
                dep_func = self.functions.get(dep)
                if dep_func is not None:
                    stack.append(dep_func)

    def _refresh_helper_bindings(self, func: Callable[..., Any], func_name: str) -> str | None:
        """Re-analyse any report whose call-site bindings moved; name a mock.

        Walks *func* and its cached-dependency closure. A report is built once
        per function, from whatever its call sites' names held at that moment,
        so a helper patched before the first call stayed "the helper" after
        the real one came back, and the helpers below a patched one were never
        walked. When a binding no longer holds the object analysed, that
        function is analysed again from the current bindings (the analyzer's
        own cache applies the same check).

        Returns a description when the tree reaches a mock -- there is no code
        to key and its answer is whatever the test configured, so the caller
        runs this call uncached -- and None otherwise. Never raises: a failure
        here leaves the reports as they were.

        Per call: one ``sys.modules`` lookup, an attribute chain and an
        identity test per binding, no hashing.
        """
        try:
            reason: str | None = None
            stack: list[tuple[Callable[..., Any], str]] = [(func, func_name)]
            seen: set[str] = set()
            while stack:
                f, name = stack.pop()
                if name in seen:
                    continue
                seen.add(name)
                report = self._purity_reports.get(name)
                if report is not None and report.helper_bindings and bindings_changed(report):
                    with self._analysis_lock:
                        self._populate_analysis(f, name)
                    report = self._purity_reports.get(name)
                if report is not None and report.unkeyable and reason is None:
                    reason = report.unkeyable[0]
                for dep in self.graph.get_dependencies(name):
                    dep_func = self.functions.get(dep)
                    if dep_func is not None:
                        stack.append((dep_func, dep))
            return reason
        except Exception:  # noqa: BLE001 - never break a call over this
            logger.debug("[CORE] binding refresh failed for %s", func_name, exc_info=True)
            return None

    def _populate_analysis(self, func: Callable[..., Any], func_name: str) -> None:
        """Record *func*'s cached-call graph edges and purity report (no
        surfacing). The analyzer caches by source hash globally, so this is
        cheap on repeated registrations.
        """
        self._populated.add(func_name)
        called_names = CodeAnalyzer.find_called_functions(func, self.functions)
        for called in called_names:
            if called != func_name:
                self.graph.add_dependency(func_name, called)
        try:
            report = get_analyzer().analyze(func)
        except (OSError, TypeError, SyntaxError, RecursionError) as e:
            # Analyzer must never break caching. On error, treat as clean -
            # the user's compute still runs.
            logger.debug("Purity analyzer failed for %s: %s", func_name, e)
            report = PurityReport()
        self._purity_reports[func_name] = report

    #: A re-hash costing more than this marks the function as not worth
    #: verifying again. Measured: ~7.4 ms for a 16 MB ndarray, so this is
    #: roughly a 100 MB argument. The first miss still gets checked -- the
    #: budget only stops a large argument from being re-hashed on every
    #: subsequent miss.
    _MUTATION_CHECK_BUDGET_S = 0.05

    def _check_argument_mutation(
        self, func_name: str, args: tuple, kwargs: dict,
        args_hash: str | None, observer: Any,
    ) -> None:
        """Did the body change the arguments it was handed?

        The static analyzer sees `rows.append(x)` written in the function, but
        not `vendor.normalize(rows)` sorting the list inside a library it does
        not walk into. Measured: of four state mutations that reached past the
        analyzer, three left an observable change in the arguments -- so the
        cheapest way to find them is to look.

        The argument hash is already computed to build the cache key, so this
        re-runs exactly that and compares. `_serialize_args` canonicalises
        (kwargs order included) and is deterministic on unchanged input, which
        is what makes a difference mean *mutation* rather than noise.

        Runs only on a miss, and only while it stays cheap: a re-hash over the
        budget above retires the check for that function rather than taxing
        every later miss.
        """
        if args_hash is None or observer is None:
            return
        if func_name in self._mutation_check_too_costly:
            return
        started = time.perf_counter()
        try:
            after = self._serialize_args(func_name, args, kwargs)
        except Exception:                                    # noqa: BLE001
            # Hashing is best-effort here. An argument that hashed once and
            # not twice (a generator drained by the body, say) is not evidence
            # of mutation, and must not be reported as such.
            return
        if time.perf_counter() - started > self._MUTATION_CHECK_BUDGET_S:
            self._mutation_check_too_costly.add(func_name)
        if after is not None and after != args_hash:
            observer.record(
                "argument mutation",
                "the arguments differ after the call than before it",
            )

    def _make_effect_observer(self) -> Any:
        """An :class:`EffectObserver` scoped to this instance's cache dir.

        Excluding the cache directory is load-bearing: cash writes the entry
        it is computing, and without the exclusion every cached function would
        be observed writing a file and every one of them would warn.
        """
        from .effect_observer import EffectObserver
        cache_dir = getattr(self.config, "cache_dir", None)
        return EffectObserver(exclude_under=cache_dir)

    def _report_observed_effects(self, func_name: str, observer: Any) -> None:
        """Warn once when the first call did something a hit will not do.

        Silent in three cases, each for its own reason:

        * ``assume_safe=True`` -- the user audited this function and said so.
        * the static analyzer already flagged it -- they have been told; a
          second warning about one function is noise, not information.
        * nothing was observed -- which is *not* proof of purity. Only the
          path this call took was watched, so an effect behind a branch that
          did not run is unobserved. That is why this supplements the static
          pass rather than replacing it.
        """
        summary = observer.summary() if observer is not None else None
        if summary is None:
            return
        if self._purity_modes.get(func_name, "warn") == "silent":
            return
        if func_name in self._purity_static_flagged:
            return
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "observed_effect",
            f"@cash.cache on {func_name}: the first call had effects that "
            f"static analysis did not see -- they happen inside library code "
            f"cash does not walk into. Every cache HIT from here on returns "
            f"the stored value WITHOUT repeating them.\n{summary}",
            code="IMPURE-OBSERVED-EFFECTS",
            fix="split the function if an effect is part of the result -- an "
                "'argument mutation' line means an object the CALLER still "
                "holds stops being changed -- or record it as incidental with "
                "@cash.cache(assume_safe=True).",
        )

    def _surface_purity(
        self, func_name: str, report: PurityReport, mode: str,
    ) -> None:
        """Turn a `PurityReport` into warnings or an exception.

        Called once per function on first call (after first
        ``_analyze_dependencies``).

        * ``warn`` (default): one-shot `CashImpurityWarning`
          summarising issues; also recorded in
          ``cache_info()['warnings']``.
        * ``silent`` (``assume_safe=True``): the user has audited
          this; suppress the warning. The report is still stored
          so helper source hashes invalidate correctly.
        * ``strict``: raise `CashImpureFunctionError`. Opaque
          callees count as issues in this mode (paranoid).
        """
        issues = list(report.issues)
        if mode == "strict" and report.opaque_callees:
            opaque_list = ", ".join(report.opaque_callees[:5])
            if len(report.opaque_callees) > 5:
                opaque_list += f", ... +{len(report.opaque_callees) - 5} more"
            issues.append(_make_opaque_issue(func_name, opaque_list))

        if not issues:
            return
        if mode == "silent":
            return

        summary = _format_issues_summary(func_name, issues)

        # Untrackable-dependency patterns (eval/exec/compile, getattr(obj,name)()
        # dynamic dispatch, importlib.import_module) RAISE by default, even in
        # the ordinary "warn" mode: cash cannot see an edit to a dependency it
        # resolves from a runtime value, so a cached result can go silently
        # stale, and caching correctness can no longer be guaranteed. The user
        # must acknowledge the risk with assume_safe=True (the ``silent`` mode
        # handled above) to cache anyway.
        from .purity_analyzer import ISSUE_UNTRACKABLE_DEP
        untrackable = [i for i in issues if getattr(i, "kind", None) == ISSUE_UNTRACKABLE_DEP]
        if untrackable and mode != "strict":
            untrackable_summary = _format_issues_summary(func_name, untrackable)
            raise CashImpureFunctionError(
                f"@cash.cache on {func_name}: a dependency is resolved from a "
                f"runtime value, so cash cannot tell when it changes and a cached "
                f"result could be silently stale. Caching correctness cannot be "
                f"guaranteed for this function.\nPut `# @cash:assume-safe` on "
                f"the line named below to accept the risk for that statement "
                f"alone, pass @cash.cache(assume_safe=True) to waive the whole "
                f"function, or refactor to a statically-named "
                f"call.\n{untrackable_summary}"
            )

        # Ambient reads get their own warning, not the side-effects one. The
        # hazard is the opposite shape -- nothing is skipped, a hidden INPUT is
        # frozen -- and so is the fix: pass the value in as an argument, where
        # it reaches the key. Filing them under "likely side effects" told the
        # user to audit for writes that are not there, and left the actual
        # failure (a nightly job whose `date.today()` is the night it first
        # ran) unnamed.
        from .purity_analyzer import ISSUE_AMBIENT_READ
        # strict=True keeps them in the one exception it raises: there, every
        # issue is a hard stop and splitting the report would hide half of it.
        ambient = [i for i in issues if getattr(i, "kind", None) == ISSUE_AMBIENT_READ]
        if ambient and mode != "strict":
            issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_AMBIENT_READ]
            self._purity_static_flagged.add(func_name)
            self._warn_once(
                CashImpurityWarning,
                func_name,
                "ambient",
                f"@cash.cache on {func_name}: the body reads ambient state "
                f"(the clock, the environment, the working directory, a fresh "
                f"UUID). That value is not part of the cache key, so the first "
                f"call's answer is what every later call gets back -- in this "
                f"process and in every process "
                f"after it.\n{_format_issues_summary(func_name, ambient)}",
                code="KEY-AMBIENT-READ",
                fix="pass the value in as an argument -- `f(now=datetime.now())` "
                    "-- so it reaches the cache key and a new value means a new "
                    "entry. If freezing it is what you want, say so with "
                    "`# @cash:assume-safe` on that line.",
            )
        if not issues:
            return
        summary = _format_issues_summary(func_name, issues)

        self._purity_static_flagged.add(func_name)

        if mode == "strict":
            raise CashImpureFunctionError(
                f"@cash.cache(strict=True) on {func_name}: purity issues "
                f"detected. Fix the function, mark callees with "
                f"@pure / @stateful, put `# @cash:assume-safe` on the lines you "
                f"have audited, or relax to assume_safe=True.\n{summary}"
            )
        # mode == "warn"
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "purity",
            f"@cash.cache on {func_name}: reading the source found likely "
            f"side effects or scope mutations, so cached results may not "
            f"reflect what the body does.\n{summary}",
            code="IMPURE-SIDE-EFFECTS",
            fix=(("for a line that changes an argument in place, return a "
                  "modified copy instead -- the caller keeps its object "
                  "whether the call hits or misses. " if "changes the argument" in summary
                  else "")
                 + "go down the list and put `# @cash:assume-safe` on each line "
                "you have audited, or refactor; @cash.cache(assume_safe=True) "
                "waives the whole function instead, including anything added "
                "to it later. The first annotation changes the function's key "
                "once: @cash: directives are part of its source identity."),
        )

    def register_file_handler(self, module_name: str, func_name: str, handler_factory: Callable[..., Any]) -> None:
        """Register a custom file-dependency handler.

        Cash already intercepts the popular reader functions
        (``pd.read_csv``, ``np.load``, ``open``, ``json.load``,
        etc.) so any cached function that uses them gets automatic
        file-dep tracking. Use this method when your code reads
        files via a custom or vendored reader that Cash doesn't
        know about yet.

        The handler is a closure-style factory: Cash gives it the
        original function and a ``track_callback(path)``; it returns
        a replacement function that calls ``track_callback`` for
        each file path it touches and then forwards to the original.
        The wrapper is installed on the target module so all callers
        - yours and any library code - get tracking transparently.

        Args:
            module_name: The module that owns the reader function
                (e.g. ``"my_lib"``, ``"my_lib.io"``). Use a dotted
                path for nested modules.
            func_name: The reader function's name in that module
                (e.g. ``"read_data"``). Supports glob wildcards like
                ``"read_*"`` to track several readers at once.
            handler_factory: Factory that produces the wrapper. Must
                accept two arguments -
                ``(original_function, track_callback)`` - and return
                a callable with the same signature as the original.
                See *Example* below for the exact shape.

        Example:

            ```python
            import cash

            c = cash.Cash()

            # my_lib.read_data(path) reads a custom binary format.
            # Make any cached function calling it invalidate when
            # the file on disk changes.
            def custom_reader_handler(original_func, track_callback):
                def wrapper(path, *args, **kwargs):
                    track_callback(path)              # record the dep
                    return original_func(path, *args, **kwargs)
                return wrapper

            c.register_file_handler("my_lib", "read_data", custom_reader_handler)

            @c.cache
            def load_features():
                import my_lib
                return my_lib.read_data("/data/features.bin")
                # ^ when /data/features.bin changes, cache invalidates
            ```

            For multiple reader names in one go:

            ```python
            c.register_file_handler("my_lib", "read_*", custom_reader_handler)
            # Catches read_data, read_metadata, read_index, ...
            ```

        Notes:
            * The wrapper replaces the attribute on the live module
              object - so existing imports
              (``from my_lib import read_data``) still see the
              original unwrapped version. Track callers that go
              through the module namespace
              (``my_lib.read_data(...)``).
            * Inside the wrapper, call ``track_callback(path)`` with
              the **absolute or resolvable** path you want recorded.
              Relative paths are resolved against ``os.getcwd()`` at
              tracking time.
            * Tracking is on the file's ``(mtime, size)``; downstream
              cache-key computation is automatic.
        """
        from .notebook.file_tracker import FileDependencyRegistry
        registry = FileDependencyRegistry()
        registry.register(module_name, func_name, handler_factory)

    def shutdown(self) -> None:
        """Cleanup resources (e.g. wait for async writes).

        Guards on the *private* ``_backend`` rather than the ``backend``
        property: this runs from an ``atexit`` handler, and touching the
        lazy property during interpreter teardown would *build* a backend
        (spawning a ThreadPoolExecutor that calls
        ``threading._register_atexit``), raising "can't register atexit
        after shutdown". If the backend was never materialised there is
        nothing to drain, so we no-op.
        """
        backend = getattr(self, '_backend', None)
        if backend is not None:
            backend.shutdown()
