"""Decorator-side purity analyzer.

Walks the body of a ``@cash.cache``-decorated function plus any
*module-bounded* helpers it calls, and reports:

* **Known-impure calls** - ``requests.post``, ``os.system``,
  ``logging.info``, file-write methods (``to_csv``, ``write``, ...).
* **Explicit dynamism** - ``eval``/``exec``/``compile``,
  ``getattr(obj, name)(...)`` where *name* is not a constant,
  calling a *parameter* directly (``def f(cb): cb(x)``).
* **Discarded-return calls** - ``f(x)`` as a statement (return
  value thrown away). Strong syntactic signal of "I'm calling this
  for the side effect". Skipped for callees in
  :data:`KNOWN_PURE_BUILTINS` (where discarding is just dead code).
* **Scope mutations** - ``global``/``nonlocal``, assignment to
  ``Attribute`` or ``Subscript`` targets, augmented-assign to same.

As a side benefit, the same walk captures source hashes of every
analyzed user-code helper. Those hashes become part of the cache
key, so editing a helper invalidates the parent cache automatically
on the next run.

**Boundary rule** - recursion follows callees that are *user code*:
the callable's source file is outside stdlib / site-packages
(``is_local_module``) **or** the callable shares the cached
function's top-level package. Everything else is treated as opaque
and optimistic (not flagged). Users who want a library call flagged
call ``cash.stateful(library_func)`` on it.

The analyzer is pure: no side effects of its own, no warnings
emitted from here. The decorator layer turns the report into
warnings / exceptions / cache-key components.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import logging
import sys
import textwrap
import threading
import types
import weakref
from collections.abc import Callable
from typing import Any

from .._annotation_refs import annotation_referents
from .._memo import PURITY_REPORTS, LruMemo
from ..diagnostics import warn_diagnostic
from ..effects import (
    EffectKind,
)
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashCacheIneffectiveWarning
from ..purity import (
    is_pure,
    is_stateful,
)
from ..source_norm import (
    callable_identity,
    compiled_identity,
    extension_file_digest,
    getsource,
    getsourcelines,
    own_source,
)
from .ambient_reads import log_helper_names, log_only_ambient_reads, method_namespace
from .annotations import assume_safe_block_lines, audited_lines
from .ast_util import bytecode_global_refs, resolve_callee
from .callee_effects import scope_locals
from .helper_bindings import (
    binding_path,
    bindings_changed,
    build_namespace,
    called_names_in_tree,
    callee_chain,
    held_ref,
    local_import_map,
    resolve_binding,
    resolve_callee_chain,
    resolve_in_class_namespaces,
    resolve_local_import,
)
from .helper_code import UnwalkableLayers, callable_layers, is_mock, is_user_code, own_code_is_user, qualname_of
from .mutable_globals import mutable_global_reads
from .purity_flow import (
    fresh_name_nodes,
)
from .purity_report import (
    ISSUE_AMBIENT_READ,
    ISSUE_IMPURE_CALL,
    PurityIssue,
    PurityReport,
)
from .purity_visitor import PurityVisitor
from .static_dispatch import MODULE_NAME_PREFIX, spell_static_dispatch

logger = logging.getLogger(__name__)

__all__ = ["PurityAnalyzer"]


class PurityAnalyzer:
    """Walks a callable's body + its module-bounded helpers and
    returns a :class:`PurityReport`.

    Results are cached by the analyzed callable's source hash.
    Multiple :class:`Cash` instances share a single process-wide
    analyzer via :func:`get_analyzer`.
    """

    #: How many callables one walk may visit. Not a depth cap: every helper
    #: the cached function reaches, however deep, is keyed, and the visited
    #: set ends cycles. Code that is finite never reaches this. What can is
    #: code that makes a NEW function on every read (a module ``__getattr__``
    #: or a class building a closure per attribute access, reached from the
    #: function it builds): the walk would never end. Stopping there silently
    #: would leave the rest out of the key, so the function runs uncached
    #: instead (``PurityReport.unwalkable``), with a warning.
    _WALK_LIMIT = 5_000

    def __init__(self) -> None:
        # memo key -> (report, the function it was built from); see `analyze`
        self._cache: LruMemo[str, tuple[PurityReport, weakref.ref | None]] = LruMemo(PURITY_REPORTS)
        self._cache_lock = threading.Lock()

    def analyze(self, func: Callable[..., Any]) -> PurityReport:
        """Return a :class:`PurityReport` for *func*.

        Idempotent and cached by source hash. A function marked ``@pure`` or
        ``@stateful`` is walked for the cache key like any other, but its body
        is not audited: ``@pure`` reports nothing, ``@stateful`` reports the
        function itself.
        """
        owner: weakref.ref | None = None
        target = getattr(func, "__func__", func)  # a bound method is made anew per access
        closure = bool(getattr(func, "__closure__", None))
        source_hash = _try_source_hash(func)
        if source_hash is not None:
            # Keyed by the namespace the names resolve in as well as the text:
            # `def run(): return step()` written identically in two modules
            # calls two different `step`s, and sharing one report handed the
            # second module the first one's helpers -- editing its own `step`
            # then changed nothing its key could see.
            source_hash = f"{source_hash}:{id(getattr(func, '__globals__', None))}"
            # An id outlives nothing: a module dropped from `sys.modules` frees
            # its namespace, and a new module with the same text can be given
            # the same address. Its function was then handed the dead one's
            # report, bindings and all -- and a binding into a module that has
            # gone proves nothing (`bindings_changed`), so a helper patched
            # with a mock was never seen and the call was served from the
            # cache. So an entry holds the function it was built from, and
            # serves only while that function is alive in the same namespace.
            try:
                owner = weakref.ref(target)
            except TypeError:
                source_hash = None
            else:
                # A closure's names also resolve in its cells: two closures
                # with the same text in one module (one factory called twice,
                # or two factories) can capture different helpers, and sharing
                # a report keyed the second by the first one's helpers. A
                # report of a closure belongs to that function object alone.
                if closure:
                    source_hash = f"{source_hash}:{id(func)}"
        if source_hash is not None:
            with self._cache_lock:
                entry = self._cache.get(source_hash)
            cached = None
            if entry is not None:
                cached, cached_owner = entry
                built_from = cached_owner() if cached_owner is not None else None
                if built_from is None:
                    cached = None
                elif closure and built_from is not target:
                    cached = None
                elif getattr(built_from, "__globals__", None) is not getattr(func, "__globals__", None):
                    cached = None
            # The source is the same, but a name it calls through may hold a
            # different object now (a patched helper, or a real one restored):
            # the tree below that binding is not the one this report walked.
            if cached is not None and not bindings_changed(cached):
                return cached

        report = self._analyze_uncached(func)
        if report.unwalkable:
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "KEY-HELPERS-UNWALKABLE",
                f"cash cannot key {qualname_of(func)}: {report.unwalkable}. It runs uncached.",
                "Name the helpers it reaches with depends_on=[...] instead of creating them on every read.",
            )
        if is_stateful(func):
            # The user has spoken: one finding for the function itself.
            report = dataclasses.replace(
                report,
                issues=(
                    PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description="explicitly marked @stateful",
                        where=qualname_of(func),
                        line=0,
                    ),
                ),
            )

        if source_hash is not None:
            with self._cache_lock:
                self._cache[source_hash] = (report, owner)
        return report

    def _analyze_uncached(self, root_func: Callable[..., Any]) -> PurityReport:
        root_module = getattr(root_func, "__module__", None)

        all_issues: list[PurityIssue] = []
        helper_hashes: dict[str, str] = {}
        helper_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        helper_objects: dict[str, Any] = {}
        opaque: list[str] = []
        visited: set[str] = set()
        bindings: list[tuple[str, tuple[str, ...], Any]] = []
        seen_bindings: set[tuple[str, tuple[str, ...]]] = set()
        unkeyable: list[str] = []
        # id(callee) -> the first call-site binding that reached it
        caller_paths: dict[int, tuple[str, tuple[str, ...]]] = {}
        waived_paths: set[tuple[str, tuple[str, ...]]] = set()
        unwaived_paths: set[tuple[str, tuple[str, ...]]] = set()
        environment_reads: set[tuple[str, str]] = set()
        cached_callees: list[Any] = []
        cached_seen: set[int] = set()

        def _note_cached(callee: Any) -> None:
            if id(callee) not in cached_seen:
                cached_seen.add(id(callee))
                cached_callees.append(held_ref(callee))

        #: Clock helpers judged at a call site: their own read is not reported.
        judged_helpers: set[Any] = set()

        def _note_binding(callee: Any, path: tuple[str, tuple[str, ...]] | None) -> None:
            if path is None or path in seen_bindings:
                return
            seen_bindings.add(path)
            bindings.append((path[0], path[1], held_ref(callee)))

        def _record_resolution_path(func: Callable[..., Any], qualname: str) -> None:
            """Note where to re-resolve *func* from ``sys.modules`` per call.

            Lets the decorator pick up in-process redefinitions (notebook
            cells, REPL) and rebinding (``monkeypatch``, ``mock.patch``).
            Preferably through the name the CALLER uses; otherwise the
            helper's own ``__qualname__`` split on '.', so methods like
            ``Klass.method`` resolve. The root function is skipped -- it is
            not a "helper" and the decorator holds its own reference.
            """
            path = caller_paths.get(id(func))
            if func is not root_func and path is not None and resolve_binding(*path) is func:
                helper_paths[qualname] = path
                return
            helper_module = getattr(func, "__module__", None)
            helper_inner_qualname = getattr(func, "__qualname__", None)
            home = (
                (helper_module, tuple(helper_inner_qualname.split(".")))
                if helper_module and helper_inner_qualname and "<locals>" not in helper_inner_qualname
                else None
            )
            # The home must lead back to THIS object. A function wrapped by a
            # decorator does not: its module name now holds the wrapper, so a
            # per-call re-resolution hashed the wrapper in its place and the
            # wrapped function's own edits never reached the key.
            if home is not None and resolve_binding(*home) is not func:
                home = None
            if func is not root_func and home is not None:
                helper_paths[qualname] = home
            elif func is not root_func:
                try:
                    helper_objects[qualname] = weakref.ref(func)
                except TypeError:  # not weak-referenceable
                    pass

        # The third element is HASH_ONLY: fold this callable's code into the
        # cache key, but do not analyze it for purity. Set for classes reached
        # from another class's body -- they are followed for correctness, not
        # audited, and analyzing them reports every ``self.x = x`` in an
        # ordinary __init__ as a scope mutation.
        def _queue_hash_only(target: Any, owner: Any, depth: int) -> None:
            if target is None or target is owner or not callable(target):
                return
            if getattr(target, "_cash_cached", False) is True:
                if not is_mock(target):
                    _note_cached(target)
                return
            if not is_user_code(target, root_module):
                return
            # ``reported`` is the entry being walked when this runs.
            stack.append((target, depth + 1, True, reported))

        def _queue_annotation_refs(obj: Any, depth: int) -> None:
            """Queue, hash-only, the user classes and functions *obj*'s
            annotations name (see ``cash._annotation_refs``): pydantic runs a
            field type's validators, a ``get_type_hints`` builder constructs it."""
            for target in annotation_referents(obj, lambda o: is_user_code(o, root_module)):
                _queue_hash_only(target, obj, depth)

        def _queue_class_refs(cls: Any, tree: ast.AST, depth: int) -> None:
            """Queue user-code objects a CLASS body constructs, hash-only."""
            if not isinstance(cls, type):
                return
            for called in called_names_in_tree(tree):
                _queue_hash_only(resolve_in_class_namespaces(cls, called), cls, depth)
            _queue_annotation_refs(cls, depth)

        # Each entry is (callable, depth, hash_only, reported). Every entry is
        # walked for the cache key. ``reported`` is False below a callable marked
        # ``@pure`` or ``@stateful``: the marker settles what it and everything
        # it calls may do, so their findings are not reported -- but their code
        # still decides the result, so it is keyed like any other helper's.
        root_reported = not (is_pure(root_func) or is_stateful(root_func))
        stack: list[tuple[Callable[..., Any], int, bool, bool]] = [(root_func, 0, False, root_reported)]
        # Why `callable_layers` could not find every function one callable
        # runs: the walk stops and the function runs uncached.
        layer_failures: list[str] = []
        if (
            isinstance(root_func, types.FunctionType)
            and hasattr(root_func, "__wrapped__")
            and not own_code_is_user(root_func, root_module)
        ):
            # `@cash.cache` over a LIBRARY decorator (`@retry(...)`,
            # `@torch.no_grad()`): the wrapper's own body is someone else's
            # code, so start from the user functions it runs instead.
            try:
                root_layers = callable_layers(root_func)
            except UnwalkableLayers as e:
                root_layers = []
                layer_failures.append(str(e))
            starts = [(layer, 0, False, root_reported) for layer in root_layers if own_code_is_user(layer, root_module)]
            if starts:
                stack = starts
        # id -> whether that walk reported findings, and the name it took.
        visited_ids: dict[Any, bool] = {}
        walked_names: dict[Any, str] = {}
        # Every object walked stays alive until the walk ends: `id()` is
        # unique only among live objects, and a bound method is made anew on
        # each attribute read, so a collected one could hand its id to a
        # different method, which was then skipped as already walked.
        keep_alive: list[Any] = []
        unwalkable = ""
        while stack:
            func, depth, hash_only, reported = stack.pop()
            reported = reported and not (is_pure(func) or is_stateful(func))
            # A bound method is visited as its function and the object it is
            # bound to: `Model.run` read twice gives two method objects, one
            # method.
            if isinstance(func, types.MethodType):
                walk_id: Any = (id(func.__func__), id(func.__self__))
            else:
                walk_id = id(func)
            walked_reported = visited_ids.get(walk_id)
            if walked_reported is not None and (walked_reported or not reported):
                continue
            if layer_failures:
                unwalkable = layer_failures[0]
                break
            if len(keep_alive) >= self._WALK_LIMIT:
                unwalkable = (
                    f"the helpers {qualname_of(root_func)} reaches do not end (over {self._WALK_LIMIT} "
                    "functions walked; code that makes a new function on every read can cause this)"
                )
                break
            keep_alive.append(func)
            visited_ids[walk_id] = reported
            if walked_reported is not None:
                # Walked below a marker first, reached now from an unmarked
                # caller too: walk it again so its findings are reported. Its
                # key part is the same, under the same name.
                qualname = walked_names[walk_id]
            else:
                qualname = qualname_of(func)
                # Visited by OBJECT: a library wrapper can copy the name of the
                # function it wraps (`toolz.curry`, `np.vectorize`), and visiting
                # by name walked only whichever of the two came first. A second
                # object under a name already taken gets a numbered one, in walk
                # order, which is the same in every process.
                if qualname in visited:
                    n = 2
                    while f"{qualname}#{n}" in visited:
                        n += 1
                    qualname = f"{qualname}#{n}"
                visited.add(qualname)
                walked_names[walk_id] = qualname

            # Read source. Failure -> opaque leaf for PURITY: we cannot see
            # what it does, so we decline to judge it.
            #
            # It must NOT become invisible to the CACHE KEY as well, which is
            # what dropping it outright used to do. A helper whose source
            # cannot be read contributed nothing to its callers' state hash,
            # so ANY edit to it went unnoticed -- not merely a constant.
            # Measured with an exec-defined helper under a filename absent
            # from linecache: replacing its entire body still served the
            # stale result. Fall back to the compiled identity, which is
            # exactly what ``hash_callable_source`` recomputes live for
            # the same object, so the snapshot and the per-call value agree
            # instead of disagreeing forever.
            try:
                src = own_source(func)
            except SOURCE_RETRIEVAL_ERRORS:
                if reported:
                    opaque.append(qualname)
                helper_hashes[qualname] = compiled_identity(func)
                _record_resolution_path(func, qualname)
                # Its helpers still decide the result. Without source there
                # is no AST to find them in, so the bytecode's global names
                # stand in: a cached function run from `python - <<EOF` or
                # `python -c` keyed its own bytecode and nothing it called,
                # and an edited helper was served the old result.
                if isinstance(func, types.FunctionType):
                    for chain in bytecode_global_refs(func):
                        callee = resolve_callee_chain(func.__globals__, chain)
                        if not isinstance(callee, types.FunctionType) or callee is func:
                            continue
                        if is_mock(callee):
                            continue
                        if getattr(callee, "_cash_cached", False) is True:
                            _note_cached(callee)
                            continue
                        if not own_code_is_user(callee, root_module):
                            continue
                        path = binding_path(func, chain)
                        _note_binding(callee, path)
                        if path is not None:
                            caller_paths.setdefault(id(callee), path)
                        stack.append((callee, depth + 1, False, reported))
                continue
            src = textwrap.dedent(src)

            # Hash the NORMALIZED source for cache-key invalidation of
            # helpers. Root function's hash is captured separately by the
            # decorator via hash_callable_source - we record all walked
            # callables here so the decorator can fold them into the state
            # hash uniformly. Normalizing means a comment or reformat in a
            # helper no longer invalidates its callers, which was the more
            # surprising half of the old behaviour: users expect editing a
            # function to recompute it, not editing something it calls.
            # `callable_identity`, the digest the live check recomputes: for a
            # wrapper it folds in what it wraps, which the text alone does not.
            helper_hashes[qualname] = callable_identity(func)

            _record_resolution_path(func, qualname)

            try:
                tree = ast.parse(src)
            except SyntaxError:
                if reported:
                    opaque.append(qualname)
                continue

            if hash_only:
                # Followed for the cache key, not audited: reporting every
                # ``self.x = x`` in an ordinary __init__ as a scope mutation
                # would bury the real findings. Keep walking what IT builds,
                # so the closure stays transitive.
                if reported:
                    opaque.append(qualname)
                _queue_class_refs(func, tree, depth)
                continue

            func_def = _find_first_function_def(tree)
            if func_def is None:
                # A CLASS, not a function: a ClassDef has no FunctionDef at
                # the top, so there is no body to analyze for purity. Bailing
                # here also ended the WALK, which lost the code the class
                # reaches. A dataclass field like
                # ``field(default_factory=lambda: B(0))`` constructs B, so
                # editing B changes what every instance holds -- measured, the
                # cache returned ``A(value=B(value=10))`` where a fresh call
                # produced ``A(value=B(value=1000))``. A wrong answer, not a
                # stale one.
                #
                # Queue what the class body CALLS. Names only, from actual
                # Call nodes: annotations are deliberately not consulted,
                # since ``value: B`` never runs and following it would
                # invalidate on a type hint.
                if reported:
                    opaque.append(qualname)
                _queue_class_refs(func, tree, depth)
                continue

            # Drop the decorator expressions: ``inspect.getsource`` includes the
            # ``@c.cache(...)`` / ``@get_cash().cache`` lines, and analyzing them
            # as if they were body statements walks into the decorator factory's
            # source (cash's own internals), flagging its mutations as the
            # user's. We only want to analyze the function body.
            func_def.decorator_list = []

            param_names = frozenset(
                arg.arg for arg in (func_def.args.args + func_def.args.posonlyargs + func_def.args.kwonlyargs)
            )
            if func_def.args.vararg:
                param_names = param_names | {func_def.args.vararg.arg}
            if func_def.args.kwarg:
                param_names = param_names | {func_def.args.kwarg.arg}

            own_issues_from = len(all_issues)
            # What the body's names are bound to: the callee resolution below,
            # and aliased ambient reads, both need it. Imports written inside
            # the body bind locals the module's globals never see; a local
            # shadows a global of the same name.
            namespace = build_namespace(func)
            local_imports = local_import_map(func_def, func)
            for _local, (_mod, _prefix) in local_imports.items():
                _obj = resolve_local_import(_mod, _prefix, root_module)
                if _obj is not None:
                    namespace[_local] = _obj
            dispatch_issues = spell_static_dispatch(func_def, namespace, qualname)
            ambient_namespace = method_namespace(func, func_def, namespace)
            visitor = PurityVisitor(
                qualname=qualname,
                param_names=param_names,
                fresh_nodes=fresh_name_nodes(func_def),
                log_only=log_only_ambient_reads(func_def, func, ambient_namespace),
                namespace=namespace,
                log_helpers=log_helper_names(func_def, func),
                ambient_namespace=ambient_namespace,
                func_def=func_def,
            )
            visitor.visit(func_def)
            visitor.finalize_taint()
            visitor.issues.extend(dispatch_issues)
            if visitor.opens_tracked_database:
                visitor.issues = [i for i in visitor.issues if i.effect_kind is not EffectKind.DB_READ]
            if depth > 0 and getattr(func, "__code__", None) in judged_helpers:
                # Judged where it is called (`clock_helper_read`). Reached
                # any other way (``fn = now; fn()``), it reports its own read.
                visitor.issues = [i for i in visitor.issues if i.kind != ISSUE_AMBIENT_READ]
            judged_helpers |= visitor.judged_helpers
            all_issues.extend(visitor.issues)
            environment_reads |= visitor.environment_reads

            # Flag reads of module globals that are reassigned/mutated somewhere
            # in the module - a silent staleness footgun (the cached result won't
            # change when the global does). Constants (never written) are not
            # flagged, so this stays quiet on dispatch tables / lookup maps.
            all_issues.extend(mutable_global_reads(func, func_def, qualname, visitor.read_names))

            # Drop what THIS function's source says it has already audited.
            # Filtered per function, against that function's own source, so a
            # waiver written in a helper covers the helper and nothing else.
            audited, function_scope = _waived_lines(src, tree, func)
            _drop_audited(all_issues, own_issues_from, audited, function_scope)
            # Only now, after the waivers matched against the function's own
            # source: report lines as the FILE numbers them. Relative to the
            # decorator line, "line 4" sent users to the wrong line.
            _anchor_issue_lines(all_issues, own_issues_from, func)
            if not reported:
                # Under ``@pure`` / ``@stateful``: walked for the key only.
                del all_issues[own_issues_from:]

            # Resolve callees and queue user-code helpers. The merged
            # namespace (built above) includes closure cells so nested-function
            # helpers (defined inside another function) are visible for
            # recursion.

            def _call_site_path(chain: tuple[str, ...] | None) -> tuple[str, tuple[str, ...]] | None:
                if chain and chain[0].startswith(MODULE_NAME_PREFIX):  # sys.modules["mod"]
                    return (chain[0][len(MODULE_NAME_PREFIX) : -1], chain[1:]) if len(chain) > 1 else None
                if chain and chain[0] in local_imports:  # loop var, used within iteration
                    module_name, prefix = local_imports[chain[0]]
                    return (module_name, prefix + chain[1:]) if module_name in sys.modules else None
                return binding_path(func, chain)

            def _queue_helper(callee: Any, line: int, path: tuple[str, tuple[str, ...]] | None = None) -> None:
                if callee is None or not callable(callee):
                    return
                # First, before any attribute read: a mock answers every
                # attribute truthily, so it would pass for a cached function
                # or a @pure one below. It has no code to key and returns
                # whatever the test configured -- the call runs uncached.
                if is_mock(callee):
                    _note_binding(callee, path)
                    where = f"{path[0]}.{'.'.join(path[1])}" if path else "a callee"
                    unkeyable.append(f"{where} is a {type(callee).__name__}")
                    return
                # A call to another @cash.cache-decorated function is a
                # dependency-graph edge, not a helper to walk: its own source
                # hash and purity are tracked as a separate node. Recursing
                # would read cash's wrapper machinery (which ``functools.wraps``
                # makes look like same-package user code) and flag cash's own
                # internal mutations as the user's.
                #
                # Its binding is still noted: rebinding the name the caller
                # calls it by (``app.inner = fake``) replaces the edge.
                if getattr(callee, "_cash_cached", False) is True:
                    _note_binding(callee, path)
                    _note_cached(callee)
                    return
                # A classmethod reached as `module.Model.run`, or bound to a
                # name (`run = Model.run`), is a method bound to the CLASS.
                # Its function is walked below; the class it reads through
                # `cls` is not named anywhere in the body, so its constants
                # (`factor = 2`) reached no channel and an edit was served
                # the old result. Key the class, as a class read by name is.
                if isinstance(callee, types.MethodType) and isinstance(callee.__self__, type):
                    _queue_hash_only(callee.__self__, func, depth)  # loop var, used within iteration
                # A ``@pure`` or ``@stateful`` callee settles what the helper
                # may DO, not what it computes: it is walked below like any
                # helper, so an edit to it moves its callers' keys, and only
                # its findings are left out (``reported`` in the walk).
                if is_stateful(callee) and reported:  # loop var, called within iteration
                    all_issues.append(
                        PurityIssue(
                            kind=ISSUE_IMPURE_CALL,
                            description=f"calls @stateful {qualname_of(callee)}()",
                            where=qualname,  # loop var, called within iteration
                            line=line,
                        )
                    )
                # The functions it runs besides its own code: the other half
                # of a decorated helper, the user function inside a library
                # wrapper (np.vectorize, toolz.curry, lru_cache), a
                # singledispatch implementation. Each user-code one is walked
                # in its own right, under its own name and namespace.
                layers = []
                try:
                    callee_layers = callable_layers(callee)
                except UnwalkableLayers as e:
                    layer_failures.append(str(e))
                    return
                for layer in callee_layers:
                    if getattr(layer, "_cash_cached", False) is True and not is_mock(layer):
                        _note_cached(layer)
                    elif own_code_is_user(layer, root_module):
                        layers.append(layer)
                # A function of a compiled extension built in the project
                # (`build_ext --inplace`) is no Python source, but it is the
                # user's code: keyed by its built file (`compiled_identity`).
                own = is_user_code(callee, root_module) or extension_file_digest(callee) is not None
                if not own and not layers:
                    # A library function the call site names by a module
                    # attribute (`requests.get`, `pd.read_csv`): not walked,
                    # but its binding is noted, so `mock.patch("requests.get")`
                    # -- or the whole module swapped for a MagicMock -- is seen
                    # on the next call and runs it uncached. It was not: the
                    # fake answer was stored under the real key and served to
                    # every later, unpatched run.
                    #
                    # Plain and builtin functions only, reached back through
                    # their path: a bound method is a new object on every
                    # attribute read and would look rebound on every call.
                    if (
                        path is not None
                        and isinstance(callee, (types.FunctionType, types.BuiltinFunctionType))
                        and resolve_binding(*path) is callee
                    ):
                        _note_binding(callee, path)
                    return
                _note_binding(callee, path)
                if own:
                    if path is not None:
                        caller_paths.setdefault(id(callee), path)
                    stack.append((callee, depth + 1, False, reported))
                for layer in layers:
                    stack.append((layer, depth + 1, False, reported))

            for call_node in visitor.called_callable_nodes + visitor.impure_call_nodes:
                site_path = _call_site_path(callee_chain(call_node.func))
                if site_path is not None:
                    start = getattr(call_node, "lineno", 0)
                    end = getattr(call_node, "end_lineno", None) or start
                    on_waived = any(n in audited for n in range(start, end + 1))
                    (waived_paths if on_waived else unwaived_paths).add(site_path)
                _queue_helper(
                    resolve_callee(call_node.func, namespace, modules_only=False),
                    getattr(call_node, "lineno", 0),
                    site_path,
                )

            # A helper referenced by NAME but reached through a value -- not in
            # call position -- was never walked, so an edit to it silently
            # served a stale result: ``fn = helper; fn(x)``, ``_apply(helper,
            # x)``, storing a function in a local then calling it. Any read name
            # that resolves to a user FUNCTION is a source dependency; walk it
            # too. Restricted to functions/methods so modules,
            # classes, and arbitrary attribute reads are not folded in, and the
            # existing user-code / purity / cached-node filters still apply. The
            # direction is safe: at worst it over-invalidates (a referenced-but-
            # unused function recomputes), never serves stale.
            for _name in visitor.read_names:
                _val = namespace.get(_name)
                if inspect.isfunction(_val) or inspect.ismethod(_val):
                    _queue_helper(_val, 0, _call_site_path((_name,)))
                elif isinstance(_val, type):
                    # A user class named but not called by a user path:
                    # `A.model_validate(d)` (a library method) or `build(A, d)`.
                    # Its code shapes the result all the same; followed for the
                    # key, not audited.
                    _queue_hash_only(_val, func, depth)
            # The same through a module or a class: `map(helper.g, xs)`,
            # `fn = helper.g`, `df.apply(features.row)`. Only the call-position
            # spelling was followed, so an edit to `g` served the old result.
            # Resolved statically (no property runs), from a module or class
            # the name is bound to, not a local or parameter that shadows it.
            shadowed = (param_names | scope_locals(func_def)) - local_imports.keys()
            for _node in visitor.read_attributes:
                _chain = callee_chain(_node)
                if _chain is None or _chain[0] in shadowed:
                    continue
                if not isinstance(namespace.get(_chain[0]), (types.ModuleType, type)):
                    continue
                _val = resolve_callee(_node, namespace, modules_only=False)
                if _val is None or is_mock(_val):
                    continue
                if getattr(_val, "_cash_cached", False) is True or (
                    (inspect.isfunction(_val) or inspect.ismethod(_val)) and is_user_code(_val, root_module)
                ):
                    _queue_helper(_val, getattr(_node, "lineno", 0), _call_site_path(_chain))
                elif isinstance(_val, type):
                    _queue_hash_only(_val, func, depth)
            _queue_annotation_refs(func, depth)
        if layer_failures and not unwalkable:
            unwalkable = layer_failures[0]

        # Stable order: by where (insertion) then line then kind.
        all_issues_sorted = tuple(
            sorted(
                all_issues,
                key=lambda i: (i.where, i.line, i.kind, i.description),
            )
        )
        return PurityReport(
            issues=all_issues_sorted,
            helper_source_hashes=helper_hashes,
            helper_objects=helper_objects,
            helper_resolution_paths=helper_paths,
            opaque_callees=tuple(sorted(set(opaque))),
            helper_bindings=tuple(bindings),
            waived_bindings=frozenset(waived_paths - unwaived_paths),
            unkeyable=tuple(unkeyable),
            environment_reads=frozenset(environment_reads),
            cached_callees=tuple(cached_callees),
            unwalkable=unwalkable,
        )


_global_analyzer: PurityAnalyzer | None = None
_global_analyzer_lock = threading.Lock()


def get_analyzer() -> PurityAnalyzer:
    """Return the process-wide :class:`PurityAnalyzer` singleton.

    Multiple :class:`Cash` instances share one analyzer cache so
    redundant AST walks are avoided across instances.
    """
    global _global_analyzer
    if _global_analyzer is not None:
        return _global_analyzer
    with _global_analyzer_lock:
        if _global_analyzer is None:
            _global_analyzer = PurityAnalyzer()
        return _global_analyzer


def _waived_lines(src: str, tree: ast.AST, func: Any) -> tuple[frozenset[int], bool]:
    """The lines of *src* a waiver covers, and whether one covers the function.

    ``# @cash:assume-safe`` (`audited_lines`) and the lines inside a ``with
    cash.assume_safe():`` block (`assume_safe_block_lines`), both numbered
    as *src* and *tree* are. Only the comment on the ``def`` line waives the
    function-scoped findings: a block is lines, and they carry none.
    """
    # Substring test before the line scan: almost no function carries a
    # comment waiver, and this runs for every function the analyzer walks.
    marked, function_scope = audited_lines(src) if "@cash:" in src else (frozenset(), False)
    if "with" in src:
        marked = marked | assume_safe_block_lines(tree, func)
    return marked, function_scope


def _drop_audited(issues: list[PurityIssue], start: int, audited: frozenset[int], function_scope: bool) -> None:
    """Remove issues in ``issues[start:]`` that the waivers cover, in place."""
    if not audited and not function_scope:
        return
    kept = [issue for issue in issues[start:] if not (issue.line in audited if issue.line else function_scope)]
    del issues[start:]
    issues.extend(kept)


def _anchor_issue_lines(issues: list[PurityIssue], start: int, func: Any) -> None:
    """Rewrite ``issues[start:]`` in the defining file's line numbers, in place.

    Anchored on the same object `own_source` read: for a ``functools.wraps``
    wrapper, its own code. ``getsourcelines`` unwraps, so a print in the
    wrapper was numbered from the WRAPPED function's first line, in another
    file -- "line 46" of an 11-line deco.py.
    """
    try:
        target = func.__code__ if isinstance(func, types.FunctionType) and hasattr(func, "__wrapped__") else func
        first = getsourcelines(target)[1]
        filename = inspect.getsourcefile(target) or ""
    except SOURCE_RETRIEVAL_ERRORS:
        return
    for index in range(start, len(issues)):
        issue = issues[index]
        issues[index] = dataclasses.replace(
            issue,
            line=issue.line + first - 1 if issue.line else 0,
            filename=filename,
        )


def _try_source_hash(func: Callable[..., Any]) -> str | None:
    """Memo key for the analyzer's own report cache -- NOT a cache key.

    Deliberately the raw text, unlike every channel that goes through
    ``source_identity_digest``. Nothing downstream keys on this, so a comment
    or a reformat only means a function is analyzed again. The normalized
    form drops what the key ignores and the report does not: a waiver, the
    ``# @cash:assume-safe`` comment or a ``with cash.assume_safe():`` block.
    Two methods named alike in one module, one of them waived, shared one
    report, and the second got the first one's findings.
    """
    try:
        src = getsource(func)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _find_first_function_def(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None
