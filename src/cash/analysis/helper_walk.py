"""The walk from a ``@cash.cache`` function through every helper it reaches.

`HelperWalk` visits the cached function and, transitively, each user-code
callable it calls, names as a value, or constructs. For every one it does two
things:

* **Keys it**: the callable's source digest goes into
  ``helper_source_hashes``, and the binding its caller reaches it through
  into ``helper_resolution_paths`` / ``helper_bindings``, so the decorator
  re-resolves and re-hashes it on each call and an edit or a rebinding
  reaches the key.
* **Audits it**: a function's body goes through `PurityVisitor`, whose
  findings become the report's issues, minus what a waiver in that
  function's own source covers.

**Boundary rule** - the walk follows callees that are *user code*: the
callable's source file is outside stdlib / site-packages
(``is_local_module``) **or** the callable shares the cached function's
top-level package. Everything else is treated as opaque and optimistic (not
flagged). Users who want a library call flagged call
``cash.stateful(library_func)`` on it.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import dataclasses
import inspect
import os
import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

from .. import _weak
from .._annotation_refs import annotation_referents
from ..code_digest import callable_identity, compiled_identity, extension_file_digest
from ..effects import EffectKind
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..install_paths import is_user_code_file, top_package
from ..jit_functions import jit_python_function
from ..loaded_code import code_objects_of
from ..purity import is_pure, is_stateful
from ..source_reading import getsourcelines, own_source
from . import fact_store, tree_memo
from .ambient_reads import log_only_ambient_reads, method_namespace
from .annotations import assume_safe_block_lines, audited_lines
from .ast_util import bytecode_global_refs, resolve_callee
from .callee_effects import scope_locals, source_global_mutations
from .check_targets import decides_checks_itself, named_only_as_check_target
from .helper_bindings import (
    binding_path,
    build_namespace,
    called_names_in_tree,
    callee_chain,
    dotted_imports,
    held_ref,
    local_import_map,
    resolve_binding,
    resolve_callee_chain,
    resolve_in_class_namespaces,
    resolve_local_import,
    unloaded_local_imports,
)
from .helper_code import (
    UnwalkableLayers,
    callable_layers,
    is_exec_built,
    is_mock,
    is_user_code,
    own_code_is_user,
    qualname_of,
)
from .mutable_globals import mutable_global_reads
from .purity_flow import (
    kept_fresh_call_candidates,
    kept_fresh_flow_nodes,
    kept_names,
    resolve_fresh_call_candidates,
    returns_memo,
)
from .purity_report import (
    ANSWERED_AT_RUN_TIME,
    ISSUE_AMBIENT_READ,
    ISSUE_IMPURE_CALL,
    ISSUE_UNTRACKABLE_DEP,
    UNREAD_MODULE,
    PurityIssue,
    PurityReport,
)
from .purity_visitor import PurityVisitor
from .static_dispatch import MODULE_NAME_PREFIX, spell_static_dispatch
from .unloaded_modules import source_digest

#: ``(module_name, attribute chain)``: where a binding is looked up again.
BindingPath = tuple[str, tuple[str, ...]]

#: The changes each callable's walk made, per walk slot (`HelperWalk._visit`),
#: for the walks of one key build (`shared_nodes`); None outside one.
_SHARED_NODES: contextvars.ContextVar[dict[tuple, tuple[Any, list]] | None] = contextvars.ContextVar(
    "cash_shared_walk_nodes", default=None
)


@contextlib.contextmanager
def shared_nodes() -> Iterator[None]:
    """Walk each callable once in the walks started inside: one key build.

    A key build asks for the report of every function of every class an
    argument holds, and each report is a walk of its own through the
    helpers below it; the helpers they share were read, parsed and audited
    again for each. Inside, a callable walked again for the same walk slot
    makes the changes its first walk made, without reading it again
    (`HelperWalk._visit`). Kept only for the build: an edit between two
    builds is read again. Nested builds share the outer one's.
    """
    if _SHARED_NODES.get() is not None:
        yield
        return
    token = _SHARED_NODES.set({})
    try:
        yield
    finally:
        _SHARED_NODES.reset(token)


class _Entry(NamedTuple):
    """One callable waiting to be walked.

    Every entry is walked for the cache key. ``hash_only``: fold its code into
    the key, but do not audit it -- set for classes reached from another
    class's body, which are followed for correctness, and analysing them
    reports every ``self.x = x`` in an ordinary ``__init__`` as a scope
    mutation. ``reported`` is False below a callable marked ``@pure`` or
    ``@stateful``: the marker settles what it and everything it calls may do,
    so their findings are not reported -- but their code still decides the
    result, so it is keyed like any other helper's.
    """

    func: Any
    depth: int
    hash_only: bool
    reported: bool


@dataclasses.dataclass
class _Body:
    """A function body the walk has audited, and what its callees are
    resolved against."""

    func: Any
    func_def: ast.FunctionDef | ast.AsyncFunctionDef
    qualname: str
    depth: int
    reported: bool
    param_names: frozenset[str]
    #: What the body's names are bound to, imports written in the body included.
    namespace: dict[str, Any]
    #: ``local name -> (module, attribute prefix)`` for imports in the body.
    local_imports: dict[str, BindingPath]
    visitor: PurityVisitor
    #: Lines a waiver in this function's own source covers.
    audited: frozenset[int]


class HelperWalk:
    """One walk from *root_func*; `run` returns its `PurityReport`."""

    #: How many callables one walk may visit. Not a depth cap: every helper
    #: the cached function reaches, however deep, is keyed, and the visited
    #: set ends cycles. Code that is finite never reaches this. What can is
    #: code that makes a NEW function on every read (a module ``__getattr__``
    #: or a class building a closure per attribute access, reached from the
    #: function it builds): the walk would never end. Stopping there silently
    #: would leave the rest out of the key, so the function runs uncached
    #: instead (``PurityReport.unwalkable``), with a warning.
    WALK_LIMIT = 5_000

    def __init__(self, root_func: Callable[..., Any]) -> None:
        self._root_func = root_func
        self._root_module: str | None = getattr(root_func, "__module__", None)
        self._issues: list[PurityIssue] = []
        self._helper_hashes: dict[str, str] = {}
        self._helper_paths: dict[str, BindingPath] = {}
        self._helper_objects: dict[str, Any] = {}
        self._walked_code: list[tuple[Any, tuple]] = []
        self._opaque: list[str] = []
        #: Names already given to a walked callable.
        self._names: set[str] = set()
        self._bindings: list[tuple[str, tuple[str, ...], Any]] = []
        self._seen_bindings: set[BindingPath] = set()
        self._unbound: dict[tuple[str, str], None] = {}
        self._unkeyable: list[str] = []
        self._unloaded: dict[str, None] = {}
        self._unvouched: dict[str, str] = {}
        #: id(callee) -> the first call-site binding that reached it
        self._caller_paths: dict[int, BindingPath] = {}
        self._waived_paths: set[BindingPath] = set()
        self._unwaived_paths: set[BindingPath] = set()
        self._environment_reads: set[tuple[str, str]] = set()
        self._cached_callees: list[Any] = []
        self._cached_seen: set[int] = set()
        #: Walk ids of the classes a body names only to test objects
        #: against (`check_targets`), and of those reached any other way.
        self._check_classes: set[Any] = set()
        self._other_classes: set[Any] = set()
        #: Clock helpers judged at a call site: their own read is not reported.
        self._judged_helpers: set[Any] = set()
        #: Why `callable_layers` could not find every function one callable
        #: runs: the walk stops and the function runs uncached.
        self._layer_failures: list[str] = []
        #: walk id -> whether that walk reported findings, and the name it took.
        self._visited_ids: dict[Any, bool] = {}
        self._walked_names: dict[Any, str] = {}
        #: Every object walked stays alive until the walk ends: `id()` is
        #: unique only among live objects, and a bound method is made anew on
        #: each attribute read, so a collected one could hand its id to a
        #: different method, which was then skipped as already walked.
        self._keep_alive: list[Any] = []
        self._unwalkable = ""
        #: `top_package` of the root's module: all of the root a callable's
        #: own walk depends on (`is_user_code` and the unloaded-import rules
        #: read nothing else of it).
        self._own_package = top_package(self._root_module)
        #: The depth of the callable being walked, which its queued callees
        #: are walked one below.
        self._depth = 0
        #: The changes the callable being walked makes to this walk, in
        #: order, while they are recorded for the next walk (`_visit`).
        self._record: list[tuple[str, tuple]] | None = None
        #: (id(body), id(callee)) for each exec()/eval()-built callee judged.
        self._exec_built_sites: set[tuple[int, int]] = set()
        #: ``(module, attribute prefix, dotted submodules)`` of each import
        #: written in an audited body: what it binds depends on what is
        #: loaded (`report_store` checks it again).
        self._local_imports: dict[tuple[str, tuple[str, ...], tuple[str, ...]], None] = {}
        #: Walk ids walked for their body, not only for the key (hash-only).
        self._bodies_walked: set[Any] = set()
        #: Functions audited with their findings reported, in walk order.
        self._audited: list[Any] = []
        #: False once something in the walk depended on where it started
        #: (`PurityReport.audited_whole`).
        self.composable = True
        self._stack: list[_Entry] = self._start()

    # --- what a stored report is checked against (`report_store`) ---

    def visited(self) -> list[Any]:
        """Every callable the walk visited, in walk order."""
        return list(self._keep_alive)

    def visited_names(self) -> dict[int, str]:
        """``id`` of each visited callable -> the name its key part went under."""
        return {id(f): self._walked_names[_walk_id(f)] for f in self._keep_alive if _walk_id(f) in self._walked_names}

    def local_imports_read(self) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
        """The imports written in the audited bodies (`_put_local_import`)."""
        return list(self._local_imports)

    # --- the walk ---

    def run(self) -> PurityReport:
        # One walk binds nothing anew: a helper's `returns_fresh` answer,
        # asked of every body that calls it, is worked out once.
        try:
            with returns_memo():
                self._run()
        finally:
            fact_store.flush()
        if self._layer_failures and not self._unwalkable:
            self._unwalkable = self._layer_failures[0]
        return self._report()

    def _run(self) -> None:
        while self._stack:
            func, depth, hash_only, reported = self._stack.pop()
            reported = reported and not (is_pure(func) or is_stateful(func))
            walk_id = _walk_id(func)
            walked_reported = self._visited_ids.get(walk_id)
            if walked_reported is not None and (walked_reported or not reported):
                if not hash_only and walk_id not in self._bodies_walked:
                    # Met first as a class's helper, for its key only: a walk
                    # from a function that calls it audits its body.
                    self.composable = False
                continue
            if self._must_stop():
                self.composable = False
                break
            if not hash_only:
                self._bodies_walked.add(walk_id)
            qualname = self._claim(func, walk_id, walked_reported, reported)
            self._visit(_Entry(func, depth, hash_only, reported), qualname)

    def _start(self) -> list[_Entry]:
        """The first entries: the root function, or, for ``@cash.cache`` over
        a LIBRARY decorator (``@retry(...)``, ``@torch.no_grad()``), the user
        functions its wrapper runs -- the wrapper's own body is someone else's
        code."""
        root = self._root_func
        root_reported = not (is_pure(root) or is_stateful(root))
        stack = [_Entry(root, 0, False, root_reported)]
        if (
            type(root) is types.FunctionType
            and hasattr(root, "__wrapped__")
            and not own_code_is_user(root, self._root_module)
        ):
            try:
                root_layers = callable_layers(root)
            except UnwalkableLayers as e:
                root_layers = []
                self._layer_failures.append(str(e))
            starts = [
                _Entry(layer, 0, False, root_reported)
                for layer in root_layers
                if own_code_is_user(layer, self._root_module)
            ]
            if starts:
                stack = starts
        return stack

    def _must_stop(self) -> bool:
        """Has the walk met something that means it cannot reach every
        helper? Then it records why, and the function runs uncached."""
        if self._layer_failures:
            self._unwalkable = self._layer_failures[0]
            return True
        if len(self._keep_alive) >= self.WALK_LIMIT:
            self._unwalkable = (
                f"the helpers {qualname_of(self._root_func)} reaches do not end (over {self.WALK_LIMIT} "
                "functions walked; code that makes a new function on every read can cause this)"
            )
            return True
        return False

    def _claim(self, func: Any, walk_id: Any, walked_reported: bool | None, reported: bool) -> str:
        """Mark *func* walked and return the name its key part goes under."""
        self._keep_alive.append(func)
        self._visited_ids[walk_id] = reported
        if walked_reported is not None:
            # Walked below a marker first, reached now from an unmarked
            # caller too: walk it again so its findings are reported. Its
            # key part is the same, under the same name.
            return self._walked_names[walk_id]
        qualname = qualname_of(func)
        # Visited by OBJECT: a library wrapper can copy the name of the
        # function it wraps (`toolz.curry`, `np.vectorize`), and visiting
        # by name walked only whichever of the two came first. A second
        # object under a name already taken gets a numbered one, in walk
        # order, which is the same in every process.
        if qualname in self._names:
            self.composable = False  # the findings' names depend on walk order
            n = 2
            while f"{qualname}#{n}" in self._names:
                n += 1
            qualname = f"{qualname}#{n}"
        self._names.add(qualname)
        self._walked_names[walk_id] = qualname
        return qualname

    def _visit(self, entry: _Entry, qualname: str) -> None:
        """`_walk_one`, or, inside `shared_nodes`, the changes the same
        callable made to an earlier walk of the same key build, made again.

        All a callable's walk reads of the walk it is in is what the entry
        and its name say and the root's package; the rest of what depends
        on the walk (which caller's binding reached it first, what is
        already noted or queued, the clock helpers judged above it) is
        decided when a change is made (the ``_put_`` methods), and that is
        done again. So the report is the one walking it again would build,
        and the callable is read, parsed and audited once per key build
        instead of once per function of a class that reaches it."""
        self._depth = entry.depth
        memo = _SHARED_NODES.get()
        func = entry.func
        if memo is None or type(func) is types.MethodType:
            # A bound method is made anew on every read: kept in the memo, it
            # would outlive the walk, and a weak reference the report holds
            # to it would stay alive where the walk's own dies.
            self._walk_one(entry, qualname)
            return
        slot = (
            id(func),
            qualname,
            entry.hash_only,
            entry.reported,
            entry.depth == 0,
            self._own_package,
            len(sys.modules),
        )
        done = memo.get(slot)
        if done is not None and done[0] is func:
            made: dict[int, Any] = {}
            for name, args in done[1]:
                getattr(self, name)(*(_made_anew(a, made) for a in args))
            return
        record: list[tuple[str, tuple]] = []
        self._record = record
        try:
            self._walk_one(entry, qualname)
        finally:
            self._record = None
        memo[slot] = (func, record)  # holds func, so its id is not reused

    def _emit(self, name: str, *args: Any) -> None:
        """Make one change to this walk (the ``_put_`` method *name*), and
        record it for `_visit` when it is recording."""
        if self._record is not None:
            self._record.append((name, tuple(map(_recipe, args))))
        getattr(self, name)(*args)

    # --- the changes one callable's walk makes (see `_visit`) ---

    def _put_opaque(self, qualname: str) -> None:
        self._opaque.append(qualname)

    def _put_hash(self, qualname: str, digest: str) -> None:
        self._helper_hashes[qualname] = digest

    def _put_resolution(self, func: Any, qualname: str) -> None:
        self._record_resolution_path(func, qualname)

    def _put_walked_code(self, item: tuple[Any, tuple]) -> None:
        self._walked_code.append(item)

    def _put_audited(self, func: Any) -> None:
        self._audited.append(func)

    def _put_push(self, callee: Any, hash_only: bool, reported: bool) -> None:
        if isinstance(callee, type):
            self._other_classes.add(_walk_id(callee))
        self._stack.append(_Entry(callee, self._depth + 1, hash_only, reported))

    def _put_check_push(self, callee: Any, reported: bool) -> None:
        """A class a body names only to test objects against: walked for
        its code like any class read by name (`_queue_hash_only`)."""
        self._check_classes.add(_walk_id(callee))
        self._stack.append(_Entry(callee, self._depth + 1, True, reported))

    def _put_caller_path(self, callee: Any, path: BindingPath) -> None:
        self._caller_paths.setdefault(id(callee), path)

    def _put_issues(self, issues: list[PurityIssue]) -> None:
        self._issues += issues

    def _put_body_issues(
        self,
        code: Any,
        below_root: bool,
        own: list[PurityIssue],
        more: list[PurityIssue],
        judged: frozenset,
        environment_reads: frozenset,
        reported: bool,
    ) -> None:
        """A body's findings (*own*: its rules', *more*: the rest), and what
        it judged and read."""
        if below_root and code in self._judged_helpers:
            # Judged where it is called (`clock_helper_read`). Reached any
            # other way (``fn = now; fn()``), it reports its own read.
            own = [i for i in own if i.kind != ISSUE_AMBIENT_READ]
            self.composable = False
        self._judged_helpers |= judged
        self._environment_reads |= environment_reads
        if reported:
            self._issues += own + more

    def _put_cached(self, callee: Any) -> None:
        if id(callee) not in self._cached_seen:
            self._cached_seen.add(id(callee))
            self._cached_callees.append(held_ref(callee))

    def _put_binding(self, callee: Any, path: BindingPath) -> None:
        if path in self._seen_bindings:
            return
        self._seen_bindings.add(path)
        self._bindings.append((path[0], path[1], held_ref(callee)))

    def _put_unbound(self, key: tuple[str, str]) -> None:
        self._unbound.setdefault(key, None)

    def _put_unloaded(self, module_name: str) -> None:
        self._unloaded.setdefault(module_name, None)

    def _put_site_path(self, path: BindingPath, waived: bool) -> None:
        (self._waived_paths if waived else self._unwaived_paths).add(path)

    def _put_local_import(self, read: tuple[str, tuple[str, ...], tuple[str, ...]]) -> None:
        self._local_imports.setdefault(read, None)

    def _put_unkeyable(self, why: str) -> None:
        self._unkeyable.append(why)

    def _put_layer_failure(self, why: str) -> None:
        self._layer_failures.append(why)

    def _put_exec_built(self, site: tuple[int, int], issues: list[PurityIssue]) -> None:
        """An exec()-built callee met at *site*: reported once per body."""
        if site in self._exec_built_sites:
            return
        self._exec_built_sites.add(site)
        self._issues += issues

    def _walk_one(self, entry: _Entry, qualname: str) -> None:
        """Key one callable, audit it if it is a function body, and queue
        what it reaches.

        A JIT-compiled function (numba's ``@njit``) is keyed as itself --
        the function it compiles plus its compile options -- and its body is
        the function it compiles."""
        func = entry.func
        code_fn = jit_python_function(func)
        if code_fn is None:
            code_fn = func
        try:
            raw_src = own_source(code_fn)
        except SOURCE_RETRIEVAL_ERRORS:
            self._walk_without_source(entry, qualname)
            return
        src = textwrap.dedent(raw_src)

        # Hash the NORMALIZED source for cache-key invalidation of helpers,
        # so a comment or reformat in a helper does not invalidate its
        # callers. The root function's hash is captured separately by the
        # decorator via hash_callable_source; every walked callable is
        # recorded here so the decorator can fold them into the state hash
        # uniformly. `callable_identity` is the digest the live check
        # recomputes: for a wrapper it folds in what it wraps, which the text
        # alone does not.
        self._emit("_put_hash", qualname, callable_identity(func))
        self._emit("_put_resolution", func, qualname)
        self._record_walked_code(func)

        try:
            tree = tree_memo.parse(src, getattr(getattr(code_fn, "__code__", None), "co_filename", None))
        except SyntaxError:
            if entry.reported:
                self._emit("_put_opaque", qualname)
            return

        func_def = None if entry.hash_only else _find_first_function_def(tree)
        if func_def is None and not entry.hash_only and _is_lambda(func):
            # A lambda's source is the statement it sits in: its body is
            # audited as the function it is (`_lambda_as_def`), so
            # `STEPS = [lambda x: x + os.environ["MODE"]]` keys MODE as
            # the same read in a def does.
            func_def = _lambda_as_def(func, tree, _dedent_width(raw_src, src))
            if func_def is not None:
                self._queue_bytecode_refs(entry)
        if func_def is None:
            # Followed for the cache key, not audited. Either a hash-only
            # entry, where reporting every ``self.x = x`` in an ordinary
            # __init__ as a scope mutation would bury the real findings; or a
            # CLASS, whose body has no function to audit but still constructs
            # code that shapes the result: a dataclass field like
            # ``field(default_factory=lambda: B(0))`` builds B, so editing B
            # changes what every instance holds. Keep walking what it builds,
            # so the closure stays transitive.
            if entry.reported:
                self._emit("_put_opaque", qualname)
            self._queue_class_refs(func, tree, entry.reported)
            if type(func) is types.FunctionType and func.__code__.co_name == "<lambda>":
                # A lambda: its source is the statement it sits in, with no
                # def to audit. What it calls still computes its result
                # (`STEPS = [lambda x: price(x)]` runs `price`, which reads
                # RATE): found by name in its bytecode, as for code with no
                # source.
                self._queue_bytecode_refs(entry)
            return

        body = self._audit(entry._replace(func=code_fn), qualname, func_def, src, tree)
        if entry.reported:
            self._emit("_put_audited", func)
        self._queue_called(body)
        self._queue_named_values(body)
        self._queue_attribute_values(body)
        self._queue_annotation_refs(code_fn, entry.reported)

    def _record_walked_code(self, func: Any) -> None:
        """Remember the code *func* ran when walked (`PurityReport.walked_code`)."""
        owner = getattr(func, "__func__", func)  # a bound method is made anew per read
        codes = code_objects_of(owner)
        if not codes:
            return
        try:
            item = (_weak.ref(owner), codes)
        except TypeError:  # not weak-referenceable
            return
        self._emit("_put_walked_code", item)

    def _walk_without_source(self, entry: _Entry, qualname: str) -> None:
        """A callable whose source cannot be read: an opaque leaf for PURITY,
        since what it does cannot be seen.

        It must not become invisible to the CACHE KEY as well: a helper that
        contributed nothing to its callers' state hash let any edit to it go
        unnoticed. It is keyed by its compiled identity, exactly what
        ``hash_callable_source`` recomputes live for the same object, so the
        snapshot and the per-call value agree. Its helpers still decide the
        result; without source there is no AST to find them in, so the
        bytecode's global names stand in (a cached function run from
        ``python -c`` otherwise keyed nothing it called).
        """
        func = entry.func
        if entry.reported:
            self._emit("_put_opaque", qualname)
        self._emit("_put_hash", qualname, compiled_identity(func))
        self._emit("_put_resolution", func, qualname)
        self._record_walked_code(func)
        code_fn = jit_python_function(func) or func
        if type(code_fn) is types.FunctionType:
            self._queue_bytecode_refs(entry._replace(func=code_fn))

    def _queue_bytecode_refs(self, entry: _Entry) -> None:
        """Queue the user functions *entry*'s bytecode calls by global name:
        its helpers when there is no AST to find them in."""
        func = entry.func
        for chain in bytecode_global_refs(func):
            callee = resolve_callee_chain(func.__globals__, chain)
            if not _is_function(callee) or callee is func:
                continue
            if is_mock(callee):
                continue
            if getattr(callee, "_cash_cached", False) is True:
                self._note_cached(callee)
                continue
            if not own_code_is_user(callee, self._root_module):
                continue
            path = binding_path(func, chain)
            self._note_binding(callee, path)
            if path is not None:
                self._emit("_put_caller_path", callee, path)
            self._emit("_put_push", callee, False, entry.reported)

    # --- auditing one body ---

    def _audit(
        self,
        entry: _Entry,
        qualname: str,
        func_def: ast.FunctionDef | ast.AsyncFunctionDef,
        src: str,
        tree: ast.AST,
    ) -> _Body:
        """Run the body rules over *func_def* and keep the findings its own
        waivers do not cover."""
        func = entry.func
        # Drop the decorator expressions: ``inspect.getsource`` includes the
        # ``@c.cache(...)`` / ``@get_cash().cache`` lines, and analyzing them
        # as if they were body statements walks into the decorator factory's
        # source (cash's own internals), flagging its mutations as the
        # user's. Only the function body is analysed.
        # A copy: the tree is shared with every other walk that met this
        # source (`tree_memo`).
        func_def = tree_memo.stripped_def(func_def)
        param_names = _param_names(func_def)
        namespace, local_imports = self._body_namespace(func, func_def)
        # A new tree when a lookup was spelled: the shared one stays what
        # its text parses to.
        func_def, dispatch_issues = spell_static_dispatch(func_def, namespace, qualname, library=_library_code(func))
        ambient_namespace = method_namespace(func, func_def, namespace)
        visitor = PurityVisitor(
            qualname=qualname,
            param_names=param_names,
            fresh_nodes=_fresh_name_nodes(func_def, namespace),
            log_only=log_only_ambient_reads(func_def, func, ambient_namespace),
            namespace=namespace,
            ambient_namespace=ambient_namespace,
            func_def=func_def,
        )
        visitor.visit(func_def)
        visitor.finalize_taint()
        own = visitor.issues + dispatch_issues
        if visitor.opens_tracked_database:
            own = [i for i in own if i.effect_kind is not EffectKind.DB_READ]
        # Reads of module globals that are reassigned or mutated somewhere in
        # the module: the cached result won't change when the global does.
        more = mutable_global_reads(func, func_def, qualname, visitor.read_names, _scope_locals(func_def))
        if self._unvouched:
            more += self._unvouched_import_issues(func_def, qualname)

        # Drop what THIS function's source says it has already audited.
        # Filtered per function, against that function's own source, so a
        # waiver written in a helper covers the helper and nothing else.
        audited, function_scope = _waived_lines(src, tree, func)
        own = _drop_audited(own, audited, function_scope)
        more = _drop_audited(more, audited, function_scope)
        if entry.reported:
            # Only now, after the waivers matched against the function's own
            # source: report lines as the FILE numbers them.
            own = _anchor_issue_lines(own, func)
            more = _anchor_issue_lines(more, func)
        self._emit(
            "_put_body_issues",
            getattr(func, "__code__", None),
            entry.depth > 0,
            own,
            more,
            frozenset(visitor.judged_helpers),
            frozenset(visitor.environment_reads),
            entry.reported,
        )
        return _Body(
            func=func,
            func_def=func_def,
            qualname=qualname,
            depth=entry.depth,
            reported=entry.reported,
            param_names=param_names,
            namespace=namespace,
            local_imports=local_imports,
            visitor=visitor,
            audited=audited,
        )

    def _body_namespace(self, func: Any, func_def: ast.AST) -> tuple[dict[str, Any], dict[str, BindingPath]]:
        """What the body's names are bound to, and the imports written in it.

        Closure cells are included, so nested-function helpers are visible.
        Imports written inside the body bind locals the module's globals never
        see; a local shadows a global of the same name.
        """
        namespace = build_namespace(func)
        local_imports = local_import_map(func_def, func)
        dotted = dotted_imports(func_def)
        self._unvouched = {}
        for local, (module_name, prefix) in local_imports.items():
            self._emit("_put_local_import", (module_name, prefix, tuple(dotted.get(local, ()))))
            obj = resolve_local_import(module_name, prefix)
            if obj is not None:
                namespace[local] = obj
            for unloaded in unloaded_local_imports(module_name, prefix, dotted.get(local, ()), self._root_module):
                # Walked once it is loaded (`bindings_changed`); keyed by
                # its source until then (`ModuleAttrFold.local_binding_parts`).
                self._emit("_put_unloaded", unloaded)
                if source_digest(unloaded, self._root_module) is None:
                    self._unvouched.setdefault(local, unloaded)
        return namespace, local_imports

    def _unvouched_import_issues(self, func_def: ast.AST, qualname: str) -> list[PurityIssue]:
        """An import in the body of a user module not loaded yet whose source
        cannot vouch for what it binds (`unloaded_modules.source_digest`):
        the walk does not import it, as that would run its code, so the key
        cannot capture its helpers or values and the call runs uncached."""
        issues = []
        for node in ast.walk(func_def):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for alias in node.names:
                local = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                module = self._unvouched.get(local)
                if module is not None:
                    issues.append(
                        PurityIssue(
                            kind=ISSUE_UNTRACKABLE_DEP,
                            description=f"import {module} - {UNREAD_MODULE}",
                            where=qualname,
                            line=node.lineno,
                        )
                    )
                    break
        return issues

    # --- queueing what a body reaches ---

    def _queue_called(self, body: _Body) -> None:
        """Every call the visitor recorded, judged or not: resolved, its
        binding noted, and the callee queued when it is user code."""
        shadowed: frozenset[str] | None = None
        for call_node in body.visitor.called_callable_nodes + body.visitor.impure_call_nodes:
            chain = callee_chain(call_node.func)
            site_path = _call_site_path(body, chain)
            if site_path is not None and len(site_path[1]) == 1 and site_path[1] == chain:
                if shadowed is None:
                    shadowed = body.param_names | _scope_locals(body.func_def)
                self._note_if_unbound(body, site_path, shadowed)
            if site_path is not None:
                start = getattr(call_node, "lineno", 0)
                end = getattr(call_node, "end_lineno", None) or start
                on_waived = any(n in body.audited for n in range(start, end + 1))
                self._emit("_put_site_path", site_path, on_waived)
            self._queue_helper(
                body,
                resolve_callee(call_node.func, body.namespace, modules_only=False),
                getattr(call_node, "lineno", 0),
                site_path,
            )

    def _queue_named_values(self, body: _Body) -> None:
        """A helper referenced by NAME but reached through a value, not in
        call position: ``fn = helper; fn(x)``, ``_apply(helper, x)``.

        Any read name that resolves to a user FUNCTION is a source dependency.
        Restricted to functions and methods, so modules and arbitrary values
        are not folded in. The direction is safe: at worst it over-invalidates
        (a referenced-but-unused function recomputes), never serves stale.
        """
        for name in body.visitor.read_names:
            value = body.namespace.get(name)
            if _is_function(value) or inspect.ismethod(value):
                self._queue_helper(body, value, 0, _call_site_path(body, (name,)))
            elif isinstance(value, type):
                # A user class named but not called by a user path:
                # `A.model_validate(d)` (a library method) or `build(A, d)`.
                # Its code shapes the result all the same; followed for the
                # key, not audited.
                self._note_decided_at_run_time(body, name, value)
                check_only = not decides_checks_itself(value) and named_only_as_check_target(
                    body.func_def, name, body.namespace
                )
                self._queue_hash_only(value, body.func, body.reported, check_only=check_only)
            elif value is not None:
                self._queue_special_functions(body, value)

    def _note_decided_at_run_time(self, body: _Body, name: str, cls: type) -> None:
        """``isinstance(x, Kind)`` where Kind's metaclass is the user's and
        answers it itself (``__instancecheck__``, ``__subclasscheck__``):
        what that reads -- a set its ``register()`` fills -- is state the
        class keeps, which no key holds. On 3.12+ the trace of what the call
        ran checks it (`cash.decorator.code_trace`); before, the call runs
        uncached, with the warning that says why."""
        if not body.reported or not untraced():
            return
        decider = next(
            (
                f"{meta.__qualname__}.{d}"
                for meta in type(cls).__mro__
                if meta not in (type, object) and is_user_code(meta, self._root_module)
                for d in _DECIDERS
                if d in vars(meta)
            ),
            None,
        )
        if decider is None:
            return
        for node in ast.walk(body.func_def):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _CHECKS
                and len(node.args) == 2
                and any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.args[1]))
            ):
                continue
            if node.lineno in body.audited:
                continue
            issue = PurityIssue(
                kind=ISSUE_UNTRACKABLE_DEP,
                description=f"{node.func.id}(..., {name}) - {ANSWERED_AT_RUN_TIME} {decider}",
                where=body.qualname,
                line=node.lineno,
            )
            self._emit("_put_issues", _anchor_issue_lines([issue], body.func))
            return

    def _queue_special_functions(self, body: _Body, value: Any) -> None:
        """An object of the user's class read by name: ``BAG['k']``,
        ``BAG.value``, ``with TIMER:`` run its special methods and
        properties. The ones that change a global are audited as helpers, so
        the change is reported as their own would be."""
        from .handed_callables import is_users_instance, special_functions

        if not is_users_instance(value):
            return
        for fn in special_functions(type(value)):
            if not is_user_code(fn, self._root_module):
                continue
            try:
                source = textwrap.dedent(own_source(fn))
            except SOURCE_RETRIEVAL_ERRORS:
                continue
            if source_global_mutations(source):
                self._queue_helper(body, fn, 0, None)

    def _queue_attribute_values(self, body: _Body) -> None:
        """The same through a module or a class: ``map(helper.g, xs)``,
        ``fn = helper.g``, ``df.apply(features.row)``.

        Resolved statically (no property runs), from a module or class the
        name is bound to, not a local or parameter that shadows it.
        """
        shadowed = (body.param_names | _scope_locals(body.func_def)) - body.local_imports.keys()
        for node in body.visitor.read_attributes:
            chain = callee_chain(node)
            if chain is None or chain[0] in shadowed:
                continue
            if not issubclass(type(body.namespace.get(chain[0])), (types.ModuleType, type)):
                continue
            value = resolve_callee(node, body.namespace, modules_only=False)
            if value is None or is_mock(value):
                continue
            if getattr(value, "_cash_cached", False) is True or (
                (_is_function(value) or inspect.ismethod(value)) and is_user_code(value, self._root_module)
            ):
                self._queue_helper(body, value, getattr(node, "lineno", 0), _call_site_path(body, chain))
            elif isinstance(value, type):
                self._queue_hash_only(value, body.func, body.reported)

    def _queue_helper(self, body: _Body, callee: Any, line: int, path: BindingPath | None) -> None:
        """Queue *callee*, reached from *body* through *path*, if it is user
        code; note its binding either way."""
        if callee is None or not callable(callee):
            return
        # First, before any attribute read: a mock answers every attribute
        # truthily, so it would pass for a cached function or a @pure one
        # below. It has no code to key and returns whatever the test
        # configured -- the call runs uncached.
        if is_mock(callee):
            self._note_binding(callee, path)
            where = f"{path[0]}.{'.'.join(path[1])}" if path else "a callee"
            self._emit("_put_unkeyable", f"{where} is a {type(callee).__name__}")
            return
        # A call to another @cash.cache-decorated function is a
        # dependency-graph edge, not a helper to walk: its own source hash and
        # purity are tracked as a separate node. Recursing would read cash's
        # wrapper machinery (which ``functools.wraps`` makes look like
        # same-package user code) and flag cash's own internal mutations as
        # the user's. Its binding is still noted: rebinding the name the
        # caller calls it by (``app.inner = fake``) replaces the edge.
        if getattr(callee, "_cash_cached", False) is True:
            self._note_binding(callee, path)
            self._note_cached(callee)
            return
        # A classmethod reached as `module.Model.run`, or bound to a name
        # (`run = Model.run`), is a method bound to the CLASS. Its function is
        # walked below; the class it reads through `cls` is not named anywhere
        # in the body, so its constants (`factor = 2`) reach the key only if
        # the class is keyed, as a class read by name is.
        if type(callee) is types.MethodType and isinstance(callee.__self__, type):
            self._queue_hash_only(callee.__self__, body.func, body.reported)
        # A ``@pure`` or ``@stateful`` callee settles what the helper may DO,
        # not what it computes: it is walked below like any helper, so an
        # edit to it moves its callers' keys, and only its findings are left
        # out (``reported`` in the walk).
        if is_stateful(callee) and body.reported:
            self._emit(
                "_put_issues",
                [
                    PurityIssue(
                        kind=ISSUE_IMPURE_CALL,
                        description=f"calls @stateful {qualname_of(callee)}()",
                        where=body.qualname,
                        line=line,
                    )
                ],
            )
        if is_exec_built(callee) and not _captured_by_root(body, callee):
            # Its source is a string the program built or read: as
            # untrackable as `exec` / `eval` written in the body. Reported
            # once per body, at the first site: a call site (which a waiver
            # on its line covers) comes before the bare read of the name.
            self._note_library_binding(callee, path)
            issues = []
            if body.reported and line not in body.audited:
                issue = PurityIssue(
                    kind=ISSUE_UNTRACKABLE_DEP,
                    description=f"{'.'.join(path[1]) if path else callee.__qualname__}(...) - built by "
                    "exec()/eval() from a string, so an edit to that string is not seen",
                    where=body.qualname,
                    line=line,
                )
                issues = _anchor_issue_lines([issue], body.func)
            self._emit("_put_exec_built", (id(body.func), id(callee)), issues)
            return
        layers = self._user_layers(callee)
        if layers is None:
            return
        # A function of a compiled extension built in the project
        # (`build_ext --inplace`) is no Python source, but it is the user's
        # code: keyed by its built file (`compiled_identity`).
        own = is_user_code(callee, self._root_module) or extension_file_digest(callee) is not None
        if not own and not layers:
            self._note_library_binding(callee, path)
            return
        self._note_binding(callee, path)
        if own:
            if path is not None:
                self._emit("_put_caller_path", callee, path)
            self._emit("_put_push", callee, False, body.reported)
        for layer in layers:
            self._emit("_put_push", layer, False, body.reported)

    def _user_layers(self, callee: Any) -> list[Any] | None:
        """The user-code functions *callee* runs besides its own code: the
        other half of a decorated helper, the user function inside a library
        wrapper (np.vectorize, toolz.curry, lru_cache), a singledispatch
        implementation. Each is walked in its own right, under its own name
        and namespace. A cached one among them is noted as an edge. None when
        the layers cannot all be found (the walk then stops)."""
        try:
            callee_layers = callable_layers(callee)
        except UnwalkableLayers as e:
            self._emit("_put_layer_failure", str(e))
            return None
        layers = []
        for layer in callee_layers:
            if getattr(layer, "_cash_cached", False) is True and not is_mock(layer):
                self._note_cached(layer)
            elif own_code_is_user(layer, self._root_module):
                layers.append(layer)
        return layers

    def _note_library_binding(self, callee: Any, path: BindingPath | None) -> None:
        """A library function the call site names by a module attribute
        (``requests.get``, ``pd.read_csv``): not walked, but its binding is
        noted, so ``mock.patch("requests.get")`` -- or the whole module
        swapped for a MagicMock -- is seen on the next call and runs it
        uncached, instead of the fake answer being stored under the real key.

        Plain and builtin functions only, reached back through their path: a
        bound method is a new object on every attribute read and would look
        rebound on every call.
        """
        if (
            path is not None
            and issubclass(type(callee), (types.FunctionType, types.BuiltinFunctionType))
            and resolve_binding(*path) is callee
        ):
            self._note_binding(callee, path)

    def _queue_hash_only(self, target: Any, owner: Any, reported: bool, *, check_only: bool = False) -> None:
        """Queue *target* for its code alone. *check_only*: a class the body
        names only to test objects against (`check_targets`)."""
        if target is None or target is owner or not callable(target):
            return
        if getattr(target, "_cash_cached", False) is True:
            if not is_mock(target):
                self._note_cached(target)
            return
        if not is_user_code(target, self._root_module):
            return
        if check_only:
            self._emit("_put_check_push", target, reported)
        else:
            self._emit("_put_push", target, True, reported)

    def _queue_annotation_refs(self, obj: Any, reported: bool) -> None:
        """Queue, hash-only, the user classes and functions *obj*'s
        annotations name (see ``cash._annotation_refs``): pydantic runs a
        field type's validators, a ``get_type_hints`` builder constructs it."""
        for target in annotation_referents(obj, lambda o: is_user_code(o, self._root_module)):
            self._queue_hash_only(target, obj, reported)

    def _queue_class_refs(self, cls: Any, tree: ast.AST, reported: bool) -> None:
        """Queue user-code objects a CLASS body constructs, hash-only.

        Names only, from actual Call nodes: an annotation (``value: B``)
        never runs, and following it would invalidate on a type hint.
        """
        if not isinstance(cls, type):
            return
        for called in called_names_in_tree(tree):
            self._queue_hash_only(resolve_in_class_namespaces(cls, called), cls, reported)
        self._queue_annotation_refs(cls, reported)

    # --- what the report records ---

    def _note_cached(self, callee: Any) -> None:
        self._emit("_put_cached", callee)

    def _note_if_unbound(self, body: _Body, path: BindingPath, shadowed: frozenset[str]) -> None:
        """Note a bare called name its module has not bound (yet): a builtin,
        or a helper defined further down the file than the first call.
        Binding it later re-analyses the function (``bindings_changed``)."""
        name = path[1][0]
        if name in shadowed or name in body.local_imports:
            return
        if name not in (getattr(body.func, "__globals__", None) or ()):
            self._emit("_put_unbound", (path[0], name))

    def _note_binding(self, callee: Any, path: BindingPath | None) -> None:
        if path is not None:
            self._emit("_put_binding", callee, path)

    def _record_resolution_path(self, func: Callable[..., Any], qualname: str) -> None:
        """Note where to re-resolve *func* from ``sys.modules`` per call.

        Lets the decorator pick up in-process redefinitions (notebook cells,
        REPL) and rebinding (``monkeypatch``, ``mock.patch``). Preferably
        through the name the CALLER uses; otherwise the helper's own
        ``__qualname__`` split on '.', so methods like ``Klass.method``
        resolve. The root function is skipped -- it is not a "helper" and the
        decorator holds its own reference.
        """
        if func is self._root_func:
            return
        path = self._caller_paths.get(id(func))
        if path is not None and resolve_binding(*path) is func:
            self._helper_paths[qualname] = path
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
        # per-call re-resolution would hash the wrapper in its place and the
        # wrapped function's own edits would never reach the key.
        if home is not None and resolve_binding(*home) is not func:
            home = None
        if home is not None:
            self._helper_paths[qualname] = home
        else:
            try:
                self._helper_objects[qualname] = _weak.ref(func)
            except TypeError:  # not weak-referenceable
                pass

    def _report(self) -> PurityReport:
        # Stable order: by where (insertion) then line then kind.
        issues = tuple(sorted(self._issues, key=lambda i: (i.where, i.line, i.kind, i.description)))
        return PurityReport(
            issues=issues,
            helper_source_hashes=self._helper_hashes,
            helper_objects=self._helper_objects,
            helper_resolution_paths=self._helper_paths,
            opaque_callees=tuple(sorted(set(self._opaque))),
            helper_bindings=tuple(self._bindings),
            unbound_names=tuple(self._unbound),
            unloaded_modules=tuple(self._unloaded),
            waived_bindings=frozenset(self._waived_paths - self._unwaived_paths),
            unkeyable=tuple(self._unkeyable),
            walked_code=tuple(self._walked_code),
            environment_reads=frozenset(self._environment_reads),
            cached_callees=tuple(self._cached_callees),
            unwalkable=self._unwalkable,
            audited_whole=self._audited_whole(),
            check_only_classes=frozenset(
                self._walked_names[wid]
                for wid in self._check_classes - self._other_classes
                if wid in self._walked_names
            ),
        )

    def _audited_whole(self) -> tuple[Any, ...]:
        """`PurityReport.audited_whole`: the functions audited here whose own
        walk would be this one's below them.

        A walk from another function differs from this one where the start
        matters: the boundary of user code is the root's top-level package;
        a root's closure may hold an exec()-built helper it keys itself
        (`_captured_by_root`); a root over a library decorator starts from
        what it wraps (`_start`). Those are left out, and so is everything
        when the walk itself met a start-dependent case (`composable`)."""
        if not self.composable or self._unwalkable:
            return ()
        package = top_package(self._root_module)
        whole = []
        for fn in self._audited:
            if (
                type(fn) is not types.FunctionType
                or fn.__closure__
                or hasattr(fn, "__wrapped__")
                or top_package(getattr(fn, "__module__", None)) != package
            ):
                continue
            try:
                whole.append(_weak.ref(fn))
            except TypeError:
                continue
        return tuple(whole)


class _BoundMethod(NamedTuple):
    """A bound method in a recorded change (`HelperWalk._emit`): made anew
    each time the change is made again, as reading it again would, so the
    record keeps no method alive that the walk would have let go."""

    func: Any
    owner: Any


def _recipe(arg: Any) -> Any:
    """*arg* as `HelperWalk._visit` records it."""
    return _BoundMethod(arg.__func__, arg.__self__) if type(arg) is types.MethodType else arg


def _made_anew(arg: Any, made: dict[int, Any]) -> Any:
    """*arg* as a recorded change is made again: one new method per recorded
    one, the same within one replay as the one read was within its walk."""
    if type(arg) is not _BoundMethod:
        return arg
    method = made.get(id(arg))
    if method is None:
        method = made[id(arg)] = types.MethodType(arg.func, arg.owner)
    return method


def _is_function(value: Any) -> bool:
    """A plain function, or a JIT-compiled one (`cash.jit_functions`)."""
    return type(value) is types.FunctionType or jit_python_function(value) is not None


def _walk_id(func: Any) -> Any:
    """What makes two callables the same walk: a bound method is visited as
    its function and the object it is bound to, since ``Model.run`` read
    twice gives two method objects, one method."""
    if type(func) is types.MethodType:
        return (id(func.__func__), id(func.__self__))
    return id(func)


def _fresh_name_nodes(func_def: ast.AST, namespace: dict[str, Any]) -> frozenset[int]:
    """`purity_flow.fresh_name_nodes`, its flow half kept with the tree: it
    reads the tree and the fresh library calls, which are resolved through
    *namespace* on every walk."""
    calls: frozenset[int] = frozenset()
    if namespace:
        calls = resolve_fresh_call_candidates(kept_fresh_call_candidates(func_def), namespace)
    return kept_fresh_flow_nodes(func_def, calls) | calls


def _scope_locals(func_def: ast.AST) -> frozenset[str]:
    """`scope_locals`, kept with the tree."""
    return kept_names(func_def, "scope_locals", scope_locals)


def _library_code(func: Any) -> bool:
    """Was *func*'s code compiled from a file of an installed library (not
    the user's code: `is_user_code_file`)? False when it has no code."""
    code = getattr(getattr(func, "__func__", func), "__code__", None)
    filename = getattr(code, "co_filename", None)
    return isinstance(filename, str) and bool(filename) and not is_user_code_file(filename)


def _param_names(func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    args = func_def.args
    names = {arg.arg for arg in (args.args + args.posonlyargs + args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return frozenset(names)


def _captured_by_root(body: _Body, callee: Any) -> bool:
    """Is *callee* held in a closure cell of the cached function itself?
    Then `ClosureFold.fold_closure` keys it by its code, sourceless or not."""
    if body.depth != 0:
        return False
    for cell in getattr(body.func, "__closure__", None) or ():
        try:
            if cell.cell_contents is callee:
                return True
        except ValueError:
            continue
    return False


def _call_site_path(body: _Body, chain: tuple[str, ...] | None) -> BindingPath | None:
    """The binding a call site in *body* reaches its callee through."""
    if chain and chain[0].startswith(MODULE_NAME_PREFIX):  # sys.modules["mod"]
        return (chain[0][len(MODULE_NAME_PREFIX) : -1], chain[1:]) if len(chain) > 1 else None
    if chain and chain[0] in body.local_imports:
        module_name, prefix = body.local_imports[chain[0]]
        return (module_name, prefix + chain[1:]) if module_name in sys.modules else None
    return binding_path(body.func, chain)


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


def _drop_audited(issues: list[PurityIssue], audited: frozenset[int], function_scope: bool) -> list[PurityIssue]:
    """*issues* without those the waivers cover."""
    if not audited and not function_scope:
        return issues
    return [
        issue
        for issue in issues
        if not ((issue.line in audited or issue.waiver_line in audited) if issue.line else function_scope)
    ]


#: The checks a metaclass of the user's may answer itself (`HelperWalk._note_decided_at_run_time`).
_CHECKS = ("isinstance", "issubclass")
_DECIDERS = ("__instancecheck__", "__subclasscheck__")


def untraced() -> bool:
    """Does nothing record what a cached call ran? Before Python 3.12, or
    with ``CASH_TRACE=0`` (`cash.decorator.code_trace.on`)."""
    return sys.version_info < (3, 12) or os.environ.get("CASH_TRACE", "").strip().lower() in ("0", "off", "false", "no")


def _anchor_issue_lines(issues: list[PurityIssue], func: Any) -> list[PurityIssue]:
    """*issues* numbered as the defining file numbers its lines.

    Anchored on the same object `own_source` read: for a ``functools.wraps``
    wrapper, its own code. ``getsourcelines`` unwraps, so a print in the
    wrapper would be numbered from the WRAPPED function's first line, in
    another file.
    """
    try:
        target = func.__code__ if type(func) is types.FunctionType and hasattr(func, "__wrapped__") else func
        first = getsourcelines(target)[1]
        filename = inspect.getsourcefile(target) or ""
    except SOURCE_RETRIEVAL_ERRORS:
        return issues
    return [
        dataclasses.replace(
            issue,
            line=issue.line + first - 1 if issue.line else 0,
            waiver_line=issue.waiver_line + first - 1 if issue.waiver_line else 0,
            filename=filename,
        )
        for issue in issues
    ]


def _is_lambda(func: Any) -> bool:
    return type(func) is types.FunctionType and func.__code__.co_name == "<lambda>"


def _dedent_width(raw: str, dedented: str) -> int:
    """How many columns `textwrap.dedent` took off each line of *raw*."""
    for before, after in zip(raw.splitlines(), dedented.splitlines()):
        if after.strip():
            return len(before) - len(after)
    return 0


def _lambda_as_def(func: types.FunctionType, tree: ast.AST, dedented_by: int = 0) -> ast.FunctionDef | None:
    """The ``lambda`` in *tree* (the statement *func*'s source is, with
    *dedented_by* columns taken off each line) that is *func*, as the
    ``def`` it stands for: its arguments, and ``return`` its body. Found by
    where its code's instructions sit; None when no lambda in the statement
    holds them all, or when more than one could be it."""
    code = func.__code__
    try:
        first = getsourcelines(func)[1]
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    best = _lambda_node(code, tree, first - 1, dedented_by)
    if best is None:
        return None
    func_def = ast.FunctionDef(
        name="<lambda>",
        args=best.args,
        body=[ast.copy_location(ast.Return(value=best.body), best.body)],
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )
    return ast.fix_missing_locations(ast.copy_location(func_def, best))


def _lambda_node(code: types.CodeType, tree: ast.AST, offset: int, dedented_by: int) -> ast.Lambda | None:
    """The ``lambda`` node in *tree* whose code is *code*; *offset* is the
    line *tree*'s first line has in the file, *dedented_by* the columns
    taken off its lines (a lambda in a class or function body).

    Python 3.11+ records each instruction's column (``co_positions``): the
    innermost lambda that holds them all is the one. Python 3.10 records
    only lines, which cannot tell apart two lambdas on one line, nor a
    lambda from one nested in it: there the lambdas whose lines hold the
    code's are narrowed to those that compile to the same shape, and none
    is chosen while two that are written differently remain.
    """
    lambdas = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Lambda) and node.end_lineno is not None and node.end_col_offset is not None
    ]
    if hasattr(code, "co_positions"):
        spots = [
            (line - offset, col - dedented_by)
            for line, _, col, end_col in code.co_positions()
            # Not the RESUME the code starts with, which sits at column 0.
            if line is not None and col is not None and end_col is not None and end_col > col
        ]
        best = None
        for node in lambdas:
            start, end = (node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)
            if spots and all(start <= spot <= end for spot in spots):
                if best is None or (start >= (best.lineno, best.col_offset)):
                    best = node  # the innermost lambda that holds them
        return best
    lines = {line - offset for _, _, line in code.co_lines() if line is not None}
    held = [node for node in lambdas if lines and all(node.lineno <= line <= node.end_lineno for line in lines)]
    same = [node for node in held if _compiles_to_shape_of(node, code)]
    # Twins written alike audit alike, so either one will do.
    return same[0] if same and len({ast.dump(node) for node in same}) == 1 else None


def _code_shape(code: types.CodeType) -> tuple:
    """What a lambda's code is made of, apart from where its free names are
    bound: compiled on its own, a closure's free names read as globals."""
    nargs = code.co_argcount + code.co_kwonlyargcount
    nargs += bool(code.co_flags & inspect.CO_VARARGS) + bool(code.co_flags & inspect.CO_VARKEYWORDS)
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS),
        code.co_varnames[:nargs],
        frozenset(code.co_names) | frozenset(code.co_freevars),
        tuple(_const_shape(c) for c in code.co_consts),
    )


def _const_shape(const: object) -> object:
    if isinstance(const, types.CodeType):
        return ("code", const.co_name, _code_shape(const))
    return (type(const).__name__, repr(const))


def _compiles_to_shape_of(node: ast.Lambda, code: types.CodeType) -> bool:
    """Whether *node*, compiled on its own, has the shape of *code*."""
    try:
        compiled = compile(ast.fix_missing_locations(ast.Expression(body=node)), "<lambda>", "eval")
    except (SyntaxError, ValueError, TypeError):
        return False
    own = [c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == "<lambda>"]
    return len(own) == 1 and _code_shape(own[0]) == _code_shape(code)


def _find_first_function_def(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None
