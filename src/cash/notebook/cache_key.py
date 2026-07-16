from __future__ import annotations

"""Unified cache key computation for statement-level caching.

Single source of truth for all cache key generation. Every call site
(runtime, simulation, virtual restore, skip checks) MUST delegate here.
See ``copilot-instructions.md`` for the full architectural invariant.
"""

import ast
import hashlib
import logging
import types
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol, runtime_checkable

from cash.notebook.lineage_store import LineageStore

__all__ = [
    "CacheKeyContext",
    "CacheKeyResult",
    "compute_cache_key",
    "write_provenance_key",
]


def write_provenance_key(code: str) -> str:
    """Backend key for a file-writer's output-provenance record (CAS-153).

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

def _is_module_like(var_name: str, val: object, virtual_modules: set[str]) -> bool:
    """Return True if the value should be treated as a module (skipped from input_hashes)."""
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

    if _is_module_like(var_name, val, virtual_modules):
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

    combined_hash_str = (
        f"{source_hash}:{':'.join(input_hashes)}{func_component}{module_component}{occurrence_component}"
    )
    combined_hash = hashlib.sha256(combined_hash_str.encode('utf-8')).hexdigest()
    cache_key = f"stmt:{combined_hash}"

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

