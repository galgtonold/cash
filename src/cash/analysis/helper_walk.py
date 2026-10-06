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
import dataclasses
import inspect
import sys
import textwrap
import types
import weakref
from collections.abc import Callable
from typing import Any, NamedTuple

from .._annotation_refs import annotation_referents
from ..code_digest import callable_identity, compiled_identity, extension_file_digest
from ..effects import EffectKind
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..purity import is_pure, is_stateful
from ..source_reading import getsourcelines, own_source
from .ambient_reads import log_helper_names, log_only_ambient_reads, method_namespace
from .annotations import assume_safe_block_lines, audited_lines
from .ast_util import bytecode_global_refs, resolve_callee
from .callee_effects import scope_locals
from .helper_bindings import (
    binding_path,
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
from .purity_flow import fresh_name_nodes
from .purity_report import ISSUE_AMBIENT_READ, ISSUE_IMPURE_CALL, ISSUE_UNTRACKABLE_DEP, PurityIssue, PurityReport
from .purity_visitor import PurityVisitor
from .static_dispatch import MODULE_NAME_PREFIX, spell_static_dispatch

#: ``(module_name, attribute chain)``: where a binding is looked up again.
BindingPath = tuple[str, tuple[str, ...]]


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
        self._opaque: list[str] = []
        #: Names already given to a walked callable.
        self._names: set[str] = set()
        self._bindings: list[tuple[str, tuple[str, ...], Any]] = []
        self._seen_bindings: set[BindingPath] = set()
        self._unkeyable: list[str] = []
        #: id(callee) -> the first call-site binding that reached it
        self._caller_paths: dict[int, BindingPath] = {}
        self._waived_paths: set[BindingPath] = set()
        self._unwaived_paths: set[BindingPath] = set()
        self._environment_reads: set[tuple[str, str]] = set()
        self._cached_callees: list[Any] = []
        self._cached_seen: set[int] = set()
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
        #: (id(body), id(callee)) for each exec()/eval()-built callee judged.
        self._exec_built_sites: set[tuple[int, int]] = set()
        self._stack: list[_Entry] = self._start()

    # --- the walk ---

    def run(self) -> PurityReport:
        while self._stack:
            func, depth, hash_only, reported = self._stack.pop()
            reported = reported and not (is_pure(func) or is_stateful(func))
            walk_id = _walk_id(func)
            walked_reported = self._visited_ids.get(walk_id)
            if walked_reported is not None and (walked_reported or not reported):
                continue
            if self._must_stop():
                break
            qualname = self._claim(func, walk_id, walked_reported, reported)
            self._walk_one(_Entry(func, depth, hash_only, reported), qualname)
        if self._layer_failures and not self._unwalkable:
            self._unwalkable = self._layer_failures[0]
        return self._report()

    def _start(self) -> list[_Entry]:
        """The first entries: the root function, or, for ``@cash.cache`` over
        a LIBRARY decorator (``@retry(...)``, ``@torch.no_grad()``), the user
        functions its wrapper runs -- the wrapper's own body is someone else's
        code."""
        root = self._root_func
        root_reported = not (is_pure(root) or is_stateful(root))
        stack = [_Entry(root, 0, False, root_reported)]
        if (
            isinstance(root, types.FunctionType)
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
            n = 2
            while f"{qualname}#{n}" in self._names:
                n += 1
            qualname = f"{qualname}#{n}"
        self._names.add(qualname)
        self._walked_names[walk_id] = qualname
        return qualname

    def _walk_one(self, entry: _Entry, qualname: str) -> None:
        """Key one callable, audit it if it is a function body, and queue
        what it reaches."""
        func = entry.func
        try:
            src = own_source(func)
        except SOURCE_RETRIEVAL_ERRORS:
            self._walk_without_source(entry, qualname)
            return
        src = textwrap.dedent(src)

        # Hash the NORMALIZED source for cache-key invalidation of helpers,
        # so a comment or reformat in a helper does not invalidate its
        # callers. The root function's hash is captured separately by the
        # decorator via hash_callable_source; every walked callable is
        # recorded here so the decorator can fold them into the state hash
        # uniformly. `callable_identity` is the digest the live check
        # recomputes: for a wrapper it folds in what it wraps, which the text
        # alone does not.
        self._helper_hashes[qualname] = callable_identity(func)
        self._record_resolution_path(func, qualname)

        try:
            tree = ast.parse(src)
        except SyntaxError:
            if entry.reported:
                self._opaque.append(qualname)
            return

        func_def = None if entry.hash_only else _find_first_function_def(tree)
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
                self._opaque.append(qualname)
            self._queue_class_refs(func, tree, entry.depth, entry.reported)
            return

        body = self._audit(entry, qualname, func_def, src, tree)
        self._queue_called(body)
        self._queue_named_values(body)
        self._queue_attribute_values(body)
        self._queue_annotation_refs(func, entry.depth, entry.reported)

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
            self._opaque.append(qualname)
        self._helper_hashes[qualname] = compiled_identity(func)
        self._record_resolution_path(func, qualname)
        if not isinstance(func, types.FunctionType):
            return
        for chain in bytecode_global_refs(func):
            callee = resolve_callee_chain(func.__globals__, chain)
            if not isinstance(callee, types.FunctionType) or callee is func:
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
                self._caller_paths.setdefault(id(callee), path)
            self._stack.append(_Entry(callee, entry.depth + 1, False, entry.reported))

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
        func_def.decorator_list = []
        param_names = _param_names(func_def)
        namespace, local_imports = self._body_namespace(func, func_def)
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
        issues = visitor.issues + dispatch_issues
        if visitor.opens_tracked_database:
            issues = [i for i in issues if i.effect_kind is not EffectKind.DB_READ]
        if entry.depth > 0 and getattr(func, "__code__", None) in self._judged_helpers:
            # Judged where it is called (`clock_helper_read`). Reached any
            # other way (``fn = now; fn()``), it reports its own read.
            issues = [i for i in issues if i.kind != ISSUE_AMBIENT_READ]
        self._judged_helpers |= visitor.judged_helpers
        self._environment_reads |= visitor.environment_reads
        # Reads of module globals that are reassigned or mutated somewhere in
        # the module: the cached result won't change when the global does.
        issues += mutable_global_reads(func, func_def, qualname, visitor.read_names)

        # Drop what THIS function's source says it has already audited.
        # Filtered per function, against that function's own source, so a
        # waiver written in a helper covers the helper and nothing else.
        audited, function_scope = _waived_lines(src, tree, func)
        issues = _drop_audited(issues, audited, function_scope)
        if entry.reported:
            # Only now, after the waivers matched against the function's own
            # source: report lines as the FILE numbers them.
            self._issues += _anchor_issue_lines(issues, func)
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
        for local, (module_name, prefix) in local_imports.items():
            obj = resolve_local_import(module_name, prefix, self._root_module)
            if obj is not None:
                namespace[local] = obj
        return namespace, local_imports

    # --- queueing what a body reaches ---

    def _queue_called(self, body: _Body) -> None:
        """Every call the visitor recorded, judged or not: resolved, its
        binding noted, and the callee queued when it is user code."""
        for call_node in body.visitor.called_callable_nodes + body.visitor.impure_call_nodes:
            site_path = _call_site_path(body, callee_chain(call_node.func))
            if site_path is not None:
                start = getattr(call_node, "lineno", 0)
                end = getattr(call_node, "end_lineno", None) or start
                on_waived = any(n in body.audited for n in range(start, end + 1))
                (self._waived_paths if on_waived else self._unwaived_paths).add(site_path)
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
            if inspect.isfunction(value) or inspect.ismethod(value):
                self._queue_helper(body, value, 0, _call_site_path(body, (name,)))
            elif isinstance(value, type):
                # A user class named but not called by a user path:
                # `A.model_validate(d)` (a library method) or `build(A, d)`.
                # Its code shapes the result all the same; followed for the
                # key, not audited.
                self._queue_hash_only(value, body.func, body.depth, body.reported)

    def _queue_attribute_values(self, body: _Body) -> None:
        """The same through a module or a class: ``map(helper.g, xs)``,
        ``fn = helper.g``, ``df.apply(features.row)``.

        Resolved statically (no property runs), from a module or class the
        name is bound to, not a local or parameter that shadows it.
        """
        shadowed = (body.param_names | scope_locals(body.func_def)) - body.local_imports.keys()
        for node in body.visitor.read_attributes:
            chain = callee_chain(node)
            if chain is None or chain[0] in shadowed:
                continue
            if not isinstance(body.namespace.get(chain[0]), (types.ModuleType, type)):
                continue
            value = resolve_callee(node, body.namespace, modules_only=False)
            if value is None or is_mock(value):
                continue
            if getattr(value, "_cash_cached", False) is True or (
                (inspect.isfunction(value) or inspect.ismethod(value)) and is_user_code(value, self._root_module)
            ):
                self._queue_helper(body, value, getattr(node, "lineno", 0), _call_site_path(body, chain))
            elif isinstance(value, type):
                self._queue_hash_only(value, body.func, body.depth, body.reported)

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
            self._unkeyable.append(f"{where} is a {type(callee).__name__}")
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
        if isinstance(callee, types.MethodType) and isinstance(callee.__self__, type):
            self._queue_hash_only(callee.__self__, body.func, body.depth, body.reported)
        # A ``@pure`` or ``@stateful`` callee settles what the helper may DO,
        # not what it computes: it is walked below like any helper, so an
        # edit to it moves its callers' keys, and only its findings are left
        # out (``reported`` in the walk).
        if is_stateful(callee) and body.reported:
            self._issues.append(
                PurityIssue(
                    kind=ISSUE_IMPURE_CALL,
                    description=f"calls @stateful {qualname_of(callee)}()",
                    where=body.qualname,
                    line=line,
                )
            )
        if is_exec_built(callee) and not _captured_by_root(body, callee):
            # Its source is a string the program built or read: as
            # untrackable as `exec` / `eval` written in the body. Reported
            # once per body, at the first site: a call site (which a waiver
            # on its line covers) comes before the bare read of the name.
            self._note_library_binding(callee, path)
            site = (id(body.func), id(callee))
            if site in self._exec_built_sites:
                return
            self._exec_built_sites.add(site)
            if body.reported and line not in body.audited:
                issue = PurityIssue(
                    kind=ISSUE_UNTRACKABLE_DEP,
                    description=f"{'.'.join(path[1]) if path else callee.__qualname__}(...) - built by "
                    "exec()/eval() from a string, so an edit to that string is not seen",
                    where=body.qualname,
                    line=line,
                )
                self._issues += _anchor_issue_lines([issue], body.func)
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
                self._caller_paths.setdefault(id(callee), path)
            self._stack.append(_Entry(callee, body.depth + 1, False, body.reported))
        for layer in layers:
            self._stack.append(_Entry(layer, body.depth + 1, False, body.reported))

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
            self._layer_failures.append(str(e))
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
            and isinstance(callee, (types.FunctionType, types.BuiltinFunctionType))
            and resolve_binding(*path) is callee
        ):
            self._note_binding(callee, path)

    def _queue_hash_only(self, target: Any, owner: Any, depth: int, reported: bool) -> None:
        if target is None or target is owner or not callable(target):
            return
        if getattr(target, "_cash_cached", False) is True:
            if not is_mock(target):
                self._note_cached(target)
            return
        if not is_user_code(target, self._root_module):
            return
        self._stack.append(_Entry(target, depth + 1, True, reported))

    def _queue_annotation_refs(self, obj: Any, depth: int, reported: bool) -> None:
        """Queue, hash-only, the user classes and functions *obj*'s
        annotations name (see ``cash._annotation_refs``): pydantic runs a
        field type's validators, a ``get_type_hints`` builder constructs it."""
        for target in annotation_referents(obj, lambda o: is_user_code(o, self._root_module)):
            self._queue_hash_only(target, obj, depth, reported)

    def _queue_class_refs(self, cls: Any, tree: ast.AST, depth: int, reported: bool) -> None:
        """Queue user-code objects a CLASS body constructs, hash-only.

        Names only, from actual Call nodes: an annotation (``value: B``)
        never runs, and following it would invalidate on a type hint.
        """
        if not isinstance(cls, type):
            return
        for called in called_names_in_tree(tree):
            self._queue_hash_only(resolve_in_class_namespaces(cls, called), cls, depth, reported)
        self._queue_annotation_refs(cls, depth, reported)

    # --- what the report records ---

    def _note_cached(self, callee: Any) -> None:
        if id(callee) not in self._cached_seen:
            self._cached_seen.add(id(callee))
            self._cached_callees.append(held_ref(callee))

    def _note_binding(self, callee: Any, path: BindingPath | None) -> None:
        if path is None or path in self._seen_bindings:
            return
        self._seen_bindings.add(path)
        self._bindings.append((path[0], path[1], held_ref(callee)))

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
                self._helper_objects[qualname] = weakref.ref(func)
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
            waived_bindings=frozenset(self._waived_paths - self._unwaived_paths),
            unkeyable=tuple(self._unkeyable),
            environment_reads=frozenset(self._environment_reads),
            cached_callees=tuple(self._cached_callees),
            unwalkable=self._unwalkable,
        )


def _walk_id(func: Any) -> Any:
    """What makes two callables the same walk: a bound method is visited as
    its function and the object it is bound to, since ``Model.run`` read
    twice gives two method objects, one method."""
    if isinstance(func, types.MethodType):
        return (id(func.__func__), id(func.__self__))
    return id(func)


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
    return [issue for issue in issues if not (issue.line in audited if issue.line else function_scope)]


def _anchor_issue_lines(issues: list[PurityIssue], func: Any) -> list[PurityIssue]:
    """*issues* numbered as the defining file numbers its lines.

    Anchored on the same object `own_source` read: for a ``functools.wraps``
    wrapper, its own code. ``getsourcelines`` unwraps, so a print in the
    wrapper would be numbered from the WRAPPED function's first line, in
    another file.
    """
    try:
        target = func.__code__ if isinstance(func, types.FunctionType) and hasattr(func, "__wrapped__") else func
        first = getsourcelines(target)[1]
        filename = inspect.getsourcefile(target) or ""
    except SOURCE_RETRIEVAL_ERRORS:
        return issues
    return [
        dataclasses.replace(issue, line=issue.line + first - 1 if issue.line else 0, filename=filename)
        for issue in issues
    ]


def _find_first_function_def(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None
