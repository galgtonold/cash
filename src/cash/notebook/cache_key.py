from __future__ import annotations

"""Unified cache key computation for statement-level caching.

Single source of truth for all cache key generation. Every call site
(runtime, simulation, virtual restore, skip checks) MUST delegate here.
See ``copilot-instructions.md`` for the full architectural invariant.
"""

import ast
import builtins
import hashlib
import logging
import types
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple, Protocol, runtime_checkable

from cash.notebook.lineage_store import LineageStore

__all__ = [
    "CacheKeyContext",
    "CacheKeyResult",
    "compute_cache_key",
    "write_provenance_key",
]


def write_provenance_key(code: str) -> str:
    """Backend key for a file-writer's output-provenance record.

    Derived from the writer statement's source ALONE (not its input lineages),
    so an edited writer maps to a different key and never matches a stale
    record, while an unchanged writer resolves to the same key across a kernel
    restart. Namespaced under ``writeprov:`` so it can never collide with a
    ``stmt:`` cache entry.
    """
    return "writeprov:" + hashlib.sha256(code.encode("utf-8")).hexdigest()

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
    variable_lineage: dict[str, str]
    user_ns: Mapping[str, Any]
    function_tracker: FunctionTrackerProtocol | None = None
    virtual_lineage: dict[str, str] | None = None
    virtual_modules: set[str] | None = None
    compute_hash_fn: Callable[[object], str] | None = None
    debug: bool = False
    debug_print_fn: Callable[..., Any] | None = None
    _lineage_store: LineageStore = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Wrap once so per-input resolution doesn't allocate a new LineageStore
        # per variable on the cache-key hot path. The store shares the backing
        # dict, so external writes to ``variable_lineage`` remain visible.
        self._lineage_store = LineageStore(backing=self.variable_lineage)

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

def is_cash_instrumentation(val: object) -> bool:
    """True when *val* is one of cash's own I/O-tracking wrappers.

    ``FileAccessTracker._patch_user_ns`` replaces ``user_ns['open']`` with a
    dispatcher wrapper so file reads can be tracked. That wrapper is a closure,
    so it cannot be pickled, so ``compute_hash`` falls back to
    ``sha256(str(id(obj)))`` -- a memory address that is different in every
    kernel.

    Any statement mentioning ``open`` therefore got a per-session cache key, its
    output lineage inherited that volatility, and every downstream key drifted
    with it: nothing restored after a restart and ``.cash`` grew a duplicate
    copy each time.

    Skipping is not merely a workaround, it restores the truth: with cash NOT
    installed, ``user_ns.get('open')`` is ``None`` (builtins do not live in
    ``user_ns``) and contributes nothing to the key. Cash's own instrumentation
    must be invisible to the key it computes.

    Tested with ``is True``, not truthiness: any object with a permissive
    ``__getattr__`` (``MagicMock``, RPC/ORM proxies) auto-creates a truthy
    attribute for ANY name, and treating those as instrumentation would drop a
    real input from the key -- trading this over-invalidation bug for an
    under-invalidation one, which is far worse. The wrapper sets the marker to
    literal ``True`` (``file_tracker.py`` ``_patch_user_ns``), so an identity
    test is both sufficient and safe.
    """
    return getattr(val, '_is_file_tracker_patch', False) is True


def is_module_like(var_name: str, val: object, virtual_modules: set[str]) -> bool:
    """Return True if the value should be treated as a module (skipped from input_hashes).

    PUBLIC because the lineage WRITE path and the upstream simulation must apply
    the identical test. They previously did not, and the asymmetry was:
    this read path skips a module, while the write path fell through to
    ``compute_hash(module)``, which cannot pickle a module and returns
    ``sha256(str(id(module)))`` -- a memory address, fresh on every kernel start.
    That poisoned the output lineage of the defining statement, which is an input
    hash for every downstream key, so nothing restored after a restart.
    """
    if var_name in virtual_modules:
        return True
    if val is None:
        return False
    try:
        if isinstance(val, types.ModuleType):
            return True
        # IPython internals and bound methods behave like modules — skip them
        if callable(val) and (var_name.startswith('_') or hasattr(val, '__self__')):
            return True
    except (AttributeError, TypeError) as exc:
        logger.debug("[CACHE_KEY] Failed to check module/callable type for '%s': %s", var_name, exc)
    return False


def called_function_dependencies(
    inputs: Iterable[str],
    user_ns: Mapping[str, Any],
    variable_lineage: Mapping[str, str],
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
    """
    seen: set[str] = set()
    stack = [name for name in inputs]
    referenced: set[str] = set()

    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        code_obj = getattr(user_ns.get(name), '__code__', None)
        if code_obj is None:
            continue
        for ref in code_obj.co_names:
            if ref in seen or ref in ('get_ipython', '__builtins__'):
                continue
            value = user_ns.get(ref)
            # Modules carry their own key component; builtins are constant.
            if not isinstance(value, types.ModuleType) and not hasattr(builtins, ref):
                referenced.add(ref)
            stack.append(ref)

    referenced -= set(inputs)
    return sorted(
        f"{ref}:{variable_lineage.get(ref, 'ABSENT')}" for ref in referenced
    )


def _process_input_var(
    var_name: str,
    virtual_modules: set[str],
    user_ns: Mapping[str, Any],
    variable_lineage: dict[str, str],
    virtual_lineage: dict[str, str],
    lineage_store: LineageStore,
    compute_hash_fn: Callable[[object], str] | None,
    function_tracker: FunctionTrackerProtocol | None,
    debug: bool,
    debug_print_fn: Callable[..., Any],
    input_hashes: list[str],
    func_source_hashes: list[str],
    module_source_hashes: list[str],
) -> None:
    """Process one input variable, appending to input_hashes / func_source_hashes / module_source_hashes."""
    val = user_ns.get(var_name)

    if is_cash_instrumentation(val):
        # cash's own shim over a builtin. Contribute nothing -- exactly as this
        # name would if cash were not installed.
        return

    if is_module_like(var_name, val, virtual_modules):
        if var_name in variable_lineage:
            module_source_hashes.append(f"{var_name}:{variable_lineage[var_name]}")
            if debug:
                debug_print_fn(
                    f"[CACHE_KEY] Module component for '{var_name}': "
                    f"{variable_lineage[var_name][:12]}..."
                )
        return

    lineage = lineage_store.resolve(
        var_name, value=val, virtual=virtual_lineage, compute_hash_fn=compute_hash_fn,
    )
    if lineage:
        input_hashes.append(lineage)
        if debug:
            debug_print_fn(f"[CACHE_KEY] Input '{var_name}' resolved to: {lineage[:16]}...")

    if val is not None and callable(val) and not isinstance(val, type) and function_tracker is not None:
        try:
            func_hash = function_tracker.get_function_source_hash(val)
            if func_hash is not None:
                func_source_hashes.append(f"{var_name}:{func_hash}")
                if debug:
                    debug_print_fn(
                        f"[CACHE_KEY] Func component for '{var_name}': "
                        f"{func_hash[:12]}..."
                    )
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
                    f"[CACHE_KEY] Output module lineage for '{out_var}': "
                    f"{variable_lineage[out_var][:12]}..."
                )
        elif callable(val):
            obj_module = getattr(val, '__module__', None)
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


    source_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()

    input_hashes: list[str] = []
    func_source_hashes: list[str] = []
    module_source_hashes: list[str] = []

    sorted_inputs = sorted(inputs)

    for var_name in sorted_inputs:
        if var_name in ('get_ipython', '__builtins__'):
            continue
        _process_input_var(
            var_name, virtual_modules, user_ns, variable_lineage, virtual_lineage,
            ctx._lineage_store, compute_hash_fn, function_tracker, debug, debug_print_fn,
            input_hashes, func_source_hashes, module_source_hashes,
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
    callee_deps = called_function_dependencies(sorted_inputs, user_ns, variable_lineage)
    callee_component = f":callees:{':'.join(callee_deps)}" if callee_deps else ""

    combined_hash_str = (
        f"{source_hash}:{':'.join(input_hashes)}{func_component}{module_component}"
        f"{occurrence_component}{callee_component}"
    )
    combined_hash = hashlib.sha256(combined_hash_str.encode('utf-8')).hexdigest()
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

