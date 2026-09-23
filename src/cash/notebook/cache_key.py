"""Unified cache key computation for statement-level caching.

Single source of truth for all cache key generation. Every call site
(runtime, simulation, virtual restore, skip checks) MUST delegate here.
See ``copilot-instructions.md`` for the full architectural invariant.
"""

from __future__ import annotations

import ast
import builtins
import dis
import hashlib
import logging
import types
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple, Protocol, runtime_checkable

from cash.notebook.lineage_store import resolve_lineage
from cash.source_norm import unparse_without_docstrings

from .lineage_formula import (
    is_cash_instrumentation,
    is_module_like,
    module_read_lineage,
    statement_environment_component,
)

__all__ = [
    "CacheKeyContext",
    "CacheKeyResult",
    "VirtualCallable",
    "compute_cache_key",
    "control_outcome_key",
    "statement_source_hash",
    "write_provenance_key",
    "read_provenance_key",
]


def statement_source_hash(code: str) -> str:
    """The code half of a statement's cache key and of its outputs' lineage.

    ``sha256`` of *code* with its function and class docstrings removed (see
    ``unparse_without_docstrings``). Statements reach here as ``ast.unparse`` text, which
    has already dropped comments; this drops the other kind of prose, so
    rewording the docstring of a function defined in a cell does not re-run
    everything that calls it.

    Every place that recomputes this digest -- the simulator's projections,
    the upstream checker, the mismatch classifier's executed-hash lookup --
    must call this, not ``sha256(code)``, or the two sides stop agreeing for
    any statement that carries a docstring.
    """
    return hashlib.sha256(unparse_without_docstrings(code).encode("utf-8")).hexdigest()


def write_provenance_key(code: str) -> str:
    """Backend key for a file-writer's output-provenance record.

    Derived from the writer statement's source ALONE (not its input lineages),
    so an edited writer maps to a different key and never matches a stale
    record, while an unchanged writer resolves to the same key across a kernel
    restart. Namespaced under ``writeprov:`` so it can never collide with a
    ``stmt:`` cache entry.
    """
    return "writeprov:" + hashlib.sha256(code.encode("utf-8")).hexdigest()


def read_provenance_key(code: str) -> str:
    """Backend key for the files a statement READ when it last ran.

    The companion of :func:`write_provenance_key`. After a kernel restart the
    session's record of what each statement read is gone, and a reader static
    analysis cannot resolve (``pd.read_csv(f)`` over a glob result) made the
    whole read set unknown -- so the reconstruction scope gate re-fired every
    writer, dragging in the expensive producers of their payloads (round 21,
    replay corpus: a backtest and a forecast recomputed for a cell that needed
    neither).
    """
    return "readprov:" + hashlib.sha256(code.encode("utf-8")).hexdigest()


def import_bindings_key(code: str) -> str:
    """Backend key for what a ``from X import Y`` statement bound when it last ran.

    After a restart the simulation meets ``from statsmodels... import
    ExponentialSmoothing`` before the import has run again, with the module
    not loaded: it took the name for a module, and had no source digest for
    the class -- which the runtime folds into the lineage of every statement
    that reads it, a ``def`` included. The def's lineage disagreed, and so did
    every statement calling it (round 23, r23s2: a 45 s forecast re-ran).
    """
    return "impbind:" + hashlib.sha256(code.encode("utf-8")).hexdigest()


def mutation_verdict_key(source_hash: str) -> str:
    """Backend key for which receivers a bare method call was observed to mutate.

    ``TrackingState.mutation_verdicts`` holds it for the session, and the
    upstream simulation reads it to reproduce the runtime's decision. After a
    restart it was gone, the simulation fell back to "an unknown method
    mutates its receiver", and ``PACK.mkdir(exist_ok=True)`` bumped ``PACK``'s
    lineage where the runtime never had -- so nothing built from ``PACK``
    restored (round 23, r23s2).
    """
    return "mutverdict:" + source_hash


def control_outcome_key(code: str) -> str:
    """Backend key for what a top-level loop left behind when it last ran.

    ``TrackingState.control_outcomes`` holds it for the session: the lineages
    a loop left, which the runtime derives from the VALUES it built and the
    simulation cannot derive from code. After a restart that record was
    gone, the simulation's loop lineages disagreed with the entries the
    runtime wrote, and nothing downstream of a loop restored (round 23,
    r23s2: every per-file read of a 1,312-file folder, again). Written only
    for a loop that cannot have done anything else (see
    ``ControlStructureProcessor._persistable_callees``).
    """
    return "ctrlout:" + hashlib.sha256(code.encode("utf-8")).hexdigest()


class VirtualCallable(NamedTuple):
    """A notebook function the simulation has seen defined but the kernel has not.

    After a restart the upstream simulation meets ``summary = score(raw)``
    before ``def score`` has run again, so ``score`` is not in ``user_ns`` --
    and the two key components that come from the live function, its source
    digest and the globals its code reads (:func:`called_function_dependencies`),
    went missing from the simulated key. It never matched the entry the
    runtime wrote, so nothing that calls a notebook function restored after a
    restart; each one was re-run, and everything it needed with it.

    Both are derivable from the ``def`` statement: the runtime compiles
    ``ast.unparse(node)``, which is also what ``inspect.getsource`` returns
    for it. Keyed by lineage AND name (:func:`virtual_callable_key`): by
    lineage so a notebook that defines ``score`` twice pairs each call with
    the definition above it, and by name because one ``from m import f, g``
    gives both names the same lineage.
    """

    source_hash: str
    code: types.CodeType


def virtual_callable_key(lineage: str, name: str) -> str:
    """Where :class:`VirtualCallable` entries live: the binding's lineage, and its name."""
    return f"{lineage}:{name}"


@runtime_checkable
class FunctionTrackerProtocol(Protocol):
    """Protocol for objects that can provide function source hashes."""

    def get_function_source_hash(self, func: Any) -> str | None:
        """Return the source hash for a callable, or None if unavailable."""
        ...


@dataclass
class CacheKeyContext:
    """Groups the environment state needed for cache key computation.

    This reduces the parameter count of ``compute_cache_key`` by bundling
    lineage state, namespace, trackers, and debug options into one object.
    Parameters that change per-call (code, inputs, outputs, occurrence_index)
    remain as direct arguments to ``compute_cache_key``.
    """

    variable_lineage: Mapping[str, str]
    user_ns: Mapping[str, Any]
    function_tracker: FunctionTrackerProtocol | None = None
    virtual_lineage: dict[str, str] | None = None
    virtual_modules: set[str] | None = None
    compute_hash_fn: Callable[[object], str] | None = None
    debug: bool = False
    debug_print_fn: Callable[..., Any] | None = None
    #: Simulation only: lineage -> :class:`VirtualCallable`, for a function
    #: named as an input that is not live in ``user_ns``. The runtime never
    #: sets it, so its keys are unchanged.
    virtual_callables: Mapping[str, VirtualCallable] | None = None


class CacheKeyResult(NamedTuple):
    """Result of :func:`compute_cache_key`."""

    cache_key: str
    """Final ``stmt:xxxx`` cache key string."""

    source_hash: str
    """SHA-256 hex digest of the statement code."""

    input_hashes: list[str]
    """Ordered lineage hashes for non-module inputs."""

    func_source_hashes: list[str]
    """``['var:hash', ...]`` for function source dependencies."""

    module_source_hashes: list[str]
    """``['var:hash', ...]`` for tracked module dependencies."""


class VirtualNamespace(NamedTuple):
    """What the simulation knows about names the kernel does not hold yet.

    Passed to :func:`called_function_dependencies` so a callee that is only a
    simulated ``def`` (:class:`VirtualCallable`) is walked like a live one:
    its code, which of the names it reads are modules, and their lineages at
    the call's position.
    """

    code_for: Callable[[str], types.CodeType | None]
    is_module: Callable[[str], bool]
    lineage_of: Callable[[str], str | None]


def virtual_namespace(
    callables: Mapping[str, VirtualCallable],
    virtual_lineage: Mapping[str, str],
    variable_lineage: Mapping[str, str],
    virtual_modules: set[str],
) -> VirtualNamespace:
    """The simulation's answers, a name's simulated lineage first."""

    def lineage_of(name: str) -> str | None:
        return virtual_lineage.get(name) or variable_lineage.get(name)

    def code_for(name: str) -> types.CodeType | None:
        found = callables.get(virtual_callable_key(lineage_of(name) or "", name))
        return found.code if found is not None else None

    return VirtualNamespace(code_for, virtual_modules.__contains__, lineage_of)


def called_function_dependencies(
    inputs: Iterable[str],
    user_ns: Mapping[str, Any],
    variable_lineage: Mapping[str, str],
    virtual: VirtualNamespace | None = None,
) -> list[str]:
    """``"name:lineage"`` components for the globals a called function reaches for.

    A statement's inputs name the functions it CALLS; they do not name what those
    functions touch when they run. Python resolves a function's globals at CALL
    time, so ``r = a(3)`` genuinely depends on every global ``a`` reads — and the
    ordinary input-lineage path cannot supply them, because it is built when
    ``def a`` executes and only sees names bound ABOVE it. A callee defined BELOW
    the caller is therefore invisible to the call site.

    Walks ``__code__.co_names`` transitively, with a ``seen`` guard so mutual
    recursion terminates. ``co_names`` also carries attribute names (``.sqrt``);
    those simply resolve to nothing and contribute a constant, so they add noise
    to the key but never spurious invalidation.

    A missing name is recorded as ``ABSENT`` rather than skipped, and that is the
    load-bearing half: it is what makes DELETING a callee change the key. Without
    it the call site keeps its entry and cash serves a cached value for code that
    would now raise ``NameError`` — a masked error rather than a stale value.

    *virtual* (simulation only) stands in for a callee that is not in
    ``user_ns``. When one is used, every name is answered by the simulation --
    the runtime's key was built with all of them live, at that position.
    """
    seen: set[str] = set()
    stack = [name for name in inputs]
    referenced: set[str] = set()
    attribute_only: set[str] = set()
    used_virtual = False

    def code_of(name: str) -> types.CodeType | None:
        nonlocal used_virtual
        code_obj = getattr(user_ns.get(name), "__code__", None)
        if code_obj is None and virtual is not None and name not in user_ns:
            code_obj = virtual.code_for(name)
            used_virtual = used_virtual or code_obj is not None
        return code_obj

    def is_module(ref: str) -> bool:
        if isinstance(user_ns.get(ref), types.ModuleType):
            return True
        return virtual is not None and ref not in user_ns and virtual.is_module(ref)

    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        code_obj = code_of(name)
        if code_obj is None:
            continue
        attrs = _attribute_only_names(code_obj)
        for ref in code_obj.co_names:
            if ref in seen or ref in ("get_ipython", "__builtins__"):
                continue
            # Modules carry their own key component; builtins are constant.
            if not is_module(ref) and not hasattr(builtins, ref):
                referenced.add(ref)
            if ref in attrs:
                attribute_only.add(ref)
                continue
            stack.append(ref)

    referenced -= set(inputs)
    # A name the callees use ONLY as an attribute (``m.forecast(h)``) reads no
    # global, so it keeps the constant it always contributed -- even when a
    # notebook variable shares it. ``forecast = run_forecast(...)`` keyed
    # ``forecast:ABSENT`` before its first run and ``forecast:<lineage>``
    # after, so the simulation never found the entry and re-ran it (round 21).
    attribute_only -= {ref for name in seen for ref in _global_names(code_of(name))}
    if used_virtual:

        def lineage(ref: str) -> str:
            return virtual.lineage_of(ref) or "ABSENT"
    else:

        def lineage(ref: str) -> str:
            return variable_lineage.get(ref, "ABSENT")

    return sorted(f"{ref}:{'ABSENT' if ref in attribute_only else lineage(ref)}" for ref in referenced)


def called_function_globals(
    inputs: Iterable[str],
    user_ns: Mapping[str, Any],
    virtual: VirtualNamespace | None = None,
    keep_modules: bool = False,
) -> set[str]:
    """Notebook names the functions named in *inputs* read when called.

    The names behind :func:`called_function_dependencies`, for a caller that
    needs the names themselves -- the re-execution planner, which must know
    that ``save_png(kind, path)`` depends on the ``scores`` its body plots.
    Transitive through called user functions, and through nested code
    (comprehensions, inner functions). Builtins are left out, and modules
    unless *keep_modules* (they are not walked into either way).

    *virtual* stands in for a function not in ``user_ns`` (simulation only).
    """
    seen: set[str] = set()
    stack = list(inputs)
    found: set[str] = set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        code = getattr(user_ns.get(name), "__code__", None)
        if code is None and virtual is not None and name not in user_ns:
            code = virtual.code_for(name)
        code_objs = [code]
        while code_objs:
            code_obj = code_objs.pop()
            if code_obj is None:
                continue
            code_objs.extend(c for c in code_obj.co_consts if isinstance(c, types.CodeType))
            for ref in _global_names(code_obj):
                if ref in ("get_ipython", "__builtins__") or hasattr(builtins, ref):
                    continue
                if isinstance(user_ns.get(ref), types.ModuleType) or (
                    virtual is not None and ref not in user_ns and virtual.is_module(ref)
                ):
                    if keep_modules:
                        found.add(ref)
                    continue
                found.add(ref)
                stack.append(ref)
    return found - set(inputs)


_ATTRIBUTE_OPS = frozenset(
    {
        "LOAD_ATTR",
        "LOAD_METHOD",
        "STORE_ATTR",
        "DELETE_ATTR",
        "LOAD_SUPER_ATTR",
    }
)


#: code object -> its global names. A code object never changes, and a call
#: made per element of a comprehension disassembled its callee every time:
#: 1,470 walks for 355 calls, a quarter of what caching them cost (round 25,
#: r25s5).
_GLOBAL_NAMES_MEMO: dict[Any, frozenset[str]] = {}


def _global_names(code_obj: Any) -> frozenset[str]:
    """``co_names`` entries *code_obj* uses as something other than an attribute."""
    if code_obj is None:
        return frozenset()
    try:
        found = _GLOBAL_NAMES_MEMO.get(code_obj)
    except TypeError:
        found = None
    if found is not None:
        return found
    try:
        found = frozenset(
            ins.argval
            for ins in dis.get_instructions(code_obj)
            if ins.opname not in _ATTRIBUTE_OPS and isinstance(ins.argval, str) and ins.argval in code_obj.co_names
        )
    except (TypeError, ValueError):
        return frozenset(code_obj.co_names)  # unknown: treat every name as a global
    try:
        if len(_GLOBAL_NAMES_MEMO) >= 4096:
            _GLOBAL_NAMES_MEMO.clear()
        _GLOBAL_NAMES_MEMO[code_obj] = found
    except TypeError:
        pass
    return found


def _attribute_only_names(code_obj: Any) -> set[str]:
    return set(code_obj.co_names) - _global_names(code_obj)


def _process_input_var(
    var_name: str,
    virtual_modules: set[str],
    user_ns: Mapping[str, Any],
    variable_lineage: dict[str, str],
    virtual_lineage: dict[str, str],
    compute_hash_fn: Callable[[object], str] | None,
    function_tracker: FunctionTrackerProtocol | None,
    debug: bool,
    debug_print_fn: Callable[..., Any],
    input_hashes: list[str],
    func_source_hashes: list[str],
    module_source_hashes: list[str],
    virtual_callables: Mapping[str, VirtualCallable] | None = None,
    code: str | None = None,
) -> None:
    """Process one input variable, appending to input_hashes / func_source_hashes / module_source_hashes."""
    val = user_ns.get(var_name)

    if is_cash_instrumentation(val):
        # cash's own shim over a builtin. Contribute nothing -- exactly as this
        # name would if cash were not installed.
        return

    if is_module_like(var_name, val, virtual_modules):
        # Narrowed to the names the statement reads when that is safe; see
        # `lineage_formula.module_read_lineage`, which the output lineage on
        # both engines calls too, so all three agree.
        narrowed = module_read_lineage(function_tracker, var_name, val, code)
        if narrowed is not None:
            module_source_hashes.append(f"{var_name}:{narrowed}")
            return
        if var_name in variable_lineage:
            module_source_hashes.append(f"{var_name}:{variable_lineage[var_name]}")
            if debug:
                debug_print_fn(f"[CACHE_KEY] Module component for '{var_name}': {variable_lineage[var_name][:12]}...")
        return

    lineage = resolve_lineage(
        var_name,
        variable_lineage,
        value=val,
        virtual=virtual_lineage,
        compute_hash_fn=compute_hash_fn,
    )
    if lineage:
        input_hashes.append(lineage)
        if debug:
            debug_print_fn(f"[CACHE_KEY] Input '{var_name}' resolved to: {lineage[:16]}...")

    if val is None and lineage and virtual_callables and var_name not in user_ns:
        virtual = virtual_callables.get(virtual_callable_key(lineage, var_name))
        if virtual is not None:
            func_source_hashes.append(f"{var_name}:{virtual.source_hash}")
        return

    if val is not None and callable(val) and not isinstance(val, type) and function_tracker is not None:
        try:
            func_hash = function_tracker.get_function_source_hash(val)
            if func_hash is not None:
                func_source_hashes.append(f"{var_name}:{func_hash}")
                if debug:
                    debug_print_fn(f"[CACHE_KEY] Func component for '{var_name}': {func_hash[:12]}...")
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            logger.debug("[CACHE_KEY] Failed to get function source hash for '%s': %s", var_name, exc)


def _collect_output_module_hashes(
    outputs: set[str],
    code: str,
    variable_lineage: dict[str, str],
    user_ns: Mapping[str, Any],
    debug: bool,
    debug_print_fn: Callable[..., Any],
) -> list[str]:
    """Return ``["out:var:hash", ...]`` entries for output variables that are modules.

    Handles three output shapes:
    * ``import trackmod``               → output is a ModuleType
    * ``from X import Y`` (callable)    → output is a callable from a tracked module
    * ``from X import Y`` (non-callable)→ parse AST to find the source module
    """
    hashes: list[str] = []
    for out_var in sorted(outputs):
        if out_var not in variable_lineage:
            continue
        val = user_ns.get(out_var)
        if isinstance(val, types.ModuleType):
            hashes.append(f"out:{out_var}:{variable_lineage[out_var]}")
            if debug:
                debug_print_fn(
                    f"[CACHE_KEY] Output module lineage for '{out_var}': {variable_lineage[out_var][:12]}..."
                )
        elif callable(val):
            obj_module = getattr(val, "__module__", None)
            if obj_module and obj_module in variable_lineage:
                hashes.append(f"from_out:{out_var}:{obj_module}:{variable_lineage[obj_module]}")
                if debug:
                    debug_print_fn(
                        f"[CACHE_KEY] from-import output module lineage for "
                        f"'{out_var}' (module '{obj_module}'): "
                        f"{variable_lineage[obj_module][:12]}..."
                    )
        else:
            # Non-callable from-import: parse AST to find source module
            entry = _from_import_constant_hash(out_var, code, variable_lineage, debug, debug_print_fn)
            if entry:
                hashes.append(entry)
    return hashes


def _from_import_constant_hash(
    out_var: str,
    code: str,
    variable_lineage: dict[str, str],
    debug: bool,
    debug_print_fn: Callable[..., Any],
) -> str | None:
    """Return a lineage hash entry for a ``from X import Y`` where Y is a non-callable constant."""
    try:
        tree = ast.parse(code.strip())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    if imported_name == out_var:
                        from_mod = node.module
                        if from_mod in variable_lineage:
                            if debug:
                                debug_print_fn(
                                    f"[CACHE_KEY] from-import constant output lineage for "
                                    f"'{out_var}' (module '{from_mod}'): "
                                    f"{variable_lineage[from_mod][:12]}..."
                                )
                            return f"from_out:{out_var}:{from_mod}:{variable_lineage[from_mod]}"
                        break
    except SyntaxError:
        pass
    return None


def compute_cache_key(
    code: str,
    inputs: set[str],
    *,
    ctx: CacheKeyContext,
    outputs: set[str] | None = None,
    occurrence_index: int = 0,
    namespace: str = "stmt",
) -> CacheKeyResult:
    """Compute the cache key for a statement.

    This is THE canonical function for building cache keys. Every call site
    in the codebase must delegate here so that the key structure never
    diverges.

    Parameters
    ----------
    code : str
        The source code of the statement.
    inputs : set[str]
        Variable names detected as inputs by CodeAnalyzer.
    ctx : CacheKeyContext
        Bundles environment state (lineage, namespace, trackers, debug).
    outputs : set[str], optional
        Variable names detected as outputs by CodeAnalyzer. Used for import
        statements where the OUTPUT is a module — the module's lineage is
        included in the cache key so that re-importing after a module reload
        produces a different key.
    occurrence_index : int, optional
        Zero-based occurrence index for duplicate statements within a cell.
        When the same statement appears multiple times (e.g., ``c.increment()``
        repeated 3 times), each occurrence gets a unique index (0, 1, 2) to
        ensure distinct cache keys. Defaults to 0 (first occurrence).
    namespace : str, optional
        Key-space prefix. ``"stmt"`` (default) for statements; ``"call"`` for
        sub-statement call units. Namespacing rather than a second builder
        keeps ADR-007's single-source-of-truth invariant: one place computes
        the combined hash, and the prefix only decides which space it lands
        in. ``write_provenance_key``'s ``writeprov:`` is the same pattern.

    Returns
    -------
    tuple of (cache_key, source_hash, input_hashes, func_source_hashes, module_source_hashes)
        - cache_key: The final ``stmt:xxxx`` cache key string.
        - source_hash: SHA-256 of the code.
        - input_hashes: Ordered list of input lineage hashes (excluding modules).
        - func_source_hashes: List of ``"var:hash"`` strings for function sources.
        - module_source_hashes: List of ``"var:hash"`` strings for tracked modules.
    """
    # Extract from context
    variable_lineage = ctx.variable_lineage
    user_ns = ctx.user_ns
    function_tracker = ctx.function_tracker
    virtual_lineage = ctx.virtual_lineage or {}
    virtual_modules = ctx.virtual_modules or set()
    compute_hash_fn = ctx.compute_hash_fn
    debug = ctx.debug
    debug_print_fn = ctx.debug_print_fn or print

    source_hash = statement_source_hash(code)

    input_hashes: list[str] = []
    func_source_hashes: list[str] = []
    module_source_hashes: list[str] = []

    sorted_inputs = sorted(inputs)

    for var_name in sorted_inputs:
        if var_name in ("get_ipython", "__builtins__"):
            continue
        _process_input_var(
            var_name,
            virtual_modules,
            user_ns,
            variable_lineage,
            virtual_lineage,
            compute_hash_fn,
            function_tracker,
            debug,
            debug_print_fn,
            input_hashes,
            func_source_hashes,
            module_source_hashes,
            ctx.virtual_callables,
            code=code,
        )

    # Build the final combined hash string
    func_component = ""
    if func_source_hashes:
        func_component = ":" + ":".join(sorted(func_source_hashes))

    module_component = ""
    if module_source_hashes:
        module_component = ":" + ":".join(sorted(module_source_hashes))

    # For import statements, also include source hashes for OUTPUT modules.
    # When `import trackmod` is re-executed after module reload, the cache key
    # must differ from the pre-reload key. Otherwise the backend cache returns
    # the old module object even though importlib already reloaded it.
    if outputs:
        output_module_hashes = _collect_output_module_hashes(
            outputs, code, variable_lineage, user_ns, debug, debug_print_fn
        )
        module_source_hashes.extend(output_module_hashes)
        if module_source_hashes:
            module_component = ":" + ":".join(sorted(module_source_hashes))

    # Include occurrence index so duplicate statements within a cell get
    # distinct keys (first occurrence is ``occ0``).
    occurrence_component = f":occ{occurrence_index}"

    # Globals the called functions reach for at CALL time, including any bound
    # BELOW this statement — which the input-lineage path structurally cannot
    # see, because it is built when the ``def`` runs and only looks upward.
    # Omitted entirely when absent, so a statement that calls no user-defined
    # function keeps a byte-identical key.
    virtual = (
        virtual_namespace(ctx.virtual_callables, virtual_lineage, variable_lineage, virtual_modules)
        if ctx.virtual_callables
        else None
    )
    callee_deps = called_function_dependencies(sorted_inputs, user_ns, variable_lineage, virtual)
    callee_component = f":callees:{':'.join(callee_deps)}" if callee_deps else ""

    # What the environment reads in the statement return now: a new value is a
    # new entry. Empty when it reads none, so every other key is unchanged.
    environment = statement_environment_component(code, user_ns)

    combined_hash_str = (
        f"{source_hash}:{':'.join(input_hashes)}{func_component}{module_component}"
        f"{occurrence_component}{callee_component}{environment}"
    )
    combined_hash = hashlib.sha256(combined_hash_str.encode("utf-8")).hexdigest()
    cache_key = f"{namespace}:{combined_hash}"

    if debug:
        debug_print_fn(
            f"[CACHE_KEY] Code: {code[:80]}... | "
            f"source_hash: {source_hash[:12]}... | "
            f"input_hashes: {[h[:12] + '...' for h in input_hashes]} | "
            f"func: {func_component[:30] if func_component else '(none)'} | "
            f"module: {module_component[:30] if module_component else '(none)'} | "
            f"cache_key: {cache_key[:30]}..."
        )

    return CacheKeyResult(cache_key, source_hash, input_hashes, func_source_hashes, module_source_hashes)
