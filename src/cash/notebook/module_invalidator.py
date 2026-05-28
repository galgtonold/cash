"""Module invalidation logic for the notebook caching subsystem.

When a tracked local module's source file changes on disk, all cached
lineages that depend on that module must be invalidated so that
downstream statements are recomputed with the updated code.  This module
encapsulates the full invalidation pipeline:

1. **Lineage update** — recompute the module's own lineage hash from its
   (possibly changed) source file plus transitive local dependencies.
2. **Import tracking cleanup** — clear execution tracking for variables
   that were imported from the changed module and refresh their values
   from the reloaded module object.
3. **Downstream propagation** — walk the ``executed_input_lineages``
   graph to find variables whose computation depended on the old module
   lineage and invalidate them.  When per-symbol granularity is available
   (i.e. we know *which* symbols in the module changed), variables that
   only use unchanged symbols are preserved.

The single public entry point is :meth:`ModuleInvalidator.invalidate`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._protocols import ShellProtocol
    from .function_tracker import FunctionTracker
    from .statement_processor import StatementProcessor

logger = logging.getLogger(__name__)

class ModuleInvalidator:
    """Encapsulates module-change detection and lineage invalidation.

    Constructed once by :class:`CashMagics` and reused for the lifetime
    of the session.  All mutable state lives on the ``StatementProcessor``
    — this class is stateless aside from the ``debug`` flag and the
    ``shell`` reference (needed to refresh ``user_ns`` values).
    """

    def __init__(self, shell: ShellProtocol, *, debug: bool = False) -> None:
        self._shell = shell
        self._debug = debug

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def invalidate(
        self,
        changed_modules: dict[str, str],
        processor: StatementProcessor,
        per_module_changed_symbols: dict[str, set[str] | None] | None = None,
    ) -> None:
        """Invalidate lineage for *changed_modules* and all dependents.

        Args:
            changed_modules: ``{module_name: file_path}`` for every module
                whose source file changed since the last check.
            processor: The active :class:`StatementProcessor` that owns the
                lineage dictionaries.
            per_module_changed_symbols: Optional ``{module_name: {symbol, ...}}``
                for granular invalidation.  Pass ``None`` for full invalidation.
        """
        if per_module_changed_symbols is None:
            per_module_changed_symbols = {}

        old_module_lineages = self._update_module_lineages(changed_modules, processor)
        self._clear_from_imported_tracking(changed_modules, processor)

        if old_module_lineages:
            self._propagate_module_invalidation(
                old_module_lineages, per_module_changed_symbols, processor,
            )

    # ------------------------------------------------------------------
    # Step 1 — recompute module lineage hashes
    # ------------------------------------------------------------------

    def _update_module_lineages(
        self,
        changed_modules: dict[str, str],
        processor: StatementProcessor,
    ) -> dict[str, str]:
        """Recompute lineage hashes for changed modules.

        Returns ``{module_name: old_lineage_hash}`` for modules that had
        a lineage before the update — consumed by the propagation step.
        """
        ft = processor.function_tracker
        old_module_lineages: dict[str, str] = {}

        for mod_name, file_path in changed_modules.items():
            old_lineage = processor.variable_lineage.get(mod_name)
            if old_lineage:
                old_module_lineages[mod_name] = old_lineage

            new_lineage = self._compute_module_lineage_hash(mod_name, file_path, ft)
            processor.variable_lineage[mod_name] = new_lineage

            processor.recently_reloaded_modules.add(mod_name)

            processor.executed_cell_codes.pop(mod_name, None)
            processor.executed_input_lineages.pop(mod_name, None)
            processor.current_session_hashes.pop(mod_name, None)

            if self._debug:
                old_short = (old_lineage or 'NONE')[:12]
                print(
                    f"[MODULE_INVALIDATE] Updated lineage for '{mod_name}': "
                    f"{old_short}... -> {new_lineage[:12]}..."
                )

        return old_module_lineages

    # ------------------------------------------------------------------
    # Step 2 — clear execution tracking for from-imports
    # ------------------------------------------------------------------

    def _clear_callable_from_imports(self, mod_name: str, processor: StatementProcessor) -> None:
        """Category 1: clear tracking for callable from-imports from *mod_name*."""
        for var_name, var_value in list(self._shell.user_ns.items()):
            if var_name.startswith('_'):
                continue
            value_module = getattr(var_value, '__module__', None)
            if value_module and (
                value_module == mod_name
                or value_module.startswith(mod_name + '.')
            ):
                processor.executed_cell_codes.pop(var_name, None)
                processor.executed_input_lineages.pop(var_name, None)
                processor.current_session_hashes.pop(var_name, None)
                processor.variable_lineage.pop(var_name, None)
                if self._debug:
                    print(
                        f"[MODULE_INVALIDATE] Cleared tracking for "
                        f"from-imported '{var_name}' (module: {mod_name})"
                    )

    def _clear_constant_from_imports(
        self, mod_name: str, processor: StatementProcessor, reloaded_mod: Any
    ) -> None:
        """Category 2: clear/refresh non-callable from-imports (constants) from *mod_name*."""
        for var_name, src_mod in list(processor._tracking_state.from_import_sources.items()):
            if src_mod != mod_name and not src_mod.startswith(mod_name + '.'):
                continue
            processor.executed_cell_codes.pop(var_name, None)
            processor.executed_input_lineages.pop(var_name, None)
            processor.current_session_hashes.pop(var_name, None)
            processor.variable_lineage.pop(var_name, None)

            actual_mod = sys.modules.get(src_mod) if src_mod != mod_name else reloaded_mod
            if actual_mod and hasattr(actual_mod, var_name):
                new_val = getattr(actual_mod, var_name)
                self._shell.user_ns[var_name] = new_val
                if self._debug:
                    print(
                        f"[MODULE_INVALIDATE] Updated value for "
                        f"from-imported constant '{var_name}' = "
                        f"{repr(new_val)[:50]} (source module: {src_mod})"
                    )
            elif self._debug:
                print(
                    f"[MODULE_INVALIDATE] Cleared tracking for "
                    f"from-imported constant '{var_name}' "
                    f"(source module: {src_mod})"
                )

    def _clear_from_imported_tracking(
        self,
        changed_modules: dict[str, str],
        processor: StatementProcessor,
    ) -> None:
        """Clear execution tracking for variables imported from changed modules.

        Handles two categories:

        1. **Callable from-imports** — any ``user_ns`` value whose
           ``__module__`` matches the changed module.
        2. **Non-callable from-imports** — constants etc. tracked via
           ``from_import_sources``.  Their values are also refreshed
           from the reloaded module object.
        """
        for mod_name in changed_modules:
            self._clear_callable_from_imports(mod_name, processor)
            reloaded_mod = sys.modules.get(mod_name)
            self._clear_constant_from_imports(mod_name, processor, reloaded_mod)

    # ------------------------------------------------------------------
    # Step 3 — propagate to downstream dependents
    # ------------------------------------------------------------------

    def _decide_symbol_action(
        self,
        var_name: str,
        input_var: str,
        changed_syms: set | None,
        processor: StatementProcessor,
    ) -> str:
        """Return ``'invalidate'`` or ``'preserve'`` given symbol-change info."""
        if changed_syms is None:
            return 'invalidate'
        if len(changed_syms) == 0:
            if self._debug:
                print(f"[GRANULAR] Preserving '{var_name}': no symbols changed in '{input_var}'")
            return 'preserve'
        var_attrs = processor._tracking_state.module_attribute_deps.get(var_name, {}).get(input_var)
        if var_attrs:
            if var_attrs & changed_syms:
                if self._debug:
                    print(f"[GRANULAR] Invalidating '{var_name}': uses changed symbols {var_attrs & changed_syms}")
                return 'invalidate'
            if self._debug:
                print(f"[GRANULAR] Preserving '{var_name}': uses {var_attrs}, changed: {changed_syms}")
            return 'preserve'
        if self._debug:
            print(f"[GRANULAR] Invalidating '{var_name}': unknown attribute access on '{input_var}'")
        return 'invalidate'

    def _classify_var_invalidation(
        self,
        var_name: str,
        input_map: dict[str, str],
        old_module_lineages: dict[str, str],
        expanded_changed_symbols: dict[str, set[str] | None],
        processor: StatementProcessor,
    ) -> str | None:
        """Return ``'invalidate'``, ``'preserve'``, or ``None`` for one variable.

        Walks ``input_map`` to find the first stale module dependency, then
        applies granular symbol-level logic to decide whether the variable
        must be recomputed or can be kept.
        """
        for input_var, stored_lineage in input_map.items():
            if not (input_var in old_module_lineages and stored_lineage == old_module_lineages[input_var]):
                continue
            changed_syms = expanded_changed_symbols.get(input_var)
            return self._decide_symbol_action(var_name, input_var, changed_syms, processor)
        return None

    def _invalidate_var(self, var_name: str, processor: StatementProcessor) -> None:
        """Clear all cached lineage state for one downstream variable."""
        processor.variable_lineage.pop(var_name, None)
        processor.executed_cell_codes.pop(var_name, None)
        processor.executed_input_lineages.pop(var_name, None)
        processor.current_session_hashes.pop(var_name, None)
        processor._tracking_state.module_attribute_deps.pop(var_name, None)
        if self._debug:
            print(f"[MODULE_INVALIDATE] Cleared lineage for dependent var '{var_name}'")

    def _register_preserved_var(
        self,
        var_name: str,
        old_module_lineages: dict[str, str],
        processor: StatementProcessor,
    ) -> None:
        """Register a granularly-preserved variable for deferred lineage update."""
        input_map = processor.executed_input_lineages.get(var_name, {})
        for mod_name_key in old_module_lineages:
            if mod_name_key in input_map:
                processor._tracking_state.granular_preserved_vars.setdefault(mod_name_key, set()).add(var_name)
                if self._debug:
                    print(f"[GRANULAR] Registered '{var_name}' for deferred lineage update on '{mod_name_key}'")

    def _propagate_module_invalidation(
        self,
        old_module_lineages: dict[str, str],
        per_module_changed_symbols: dict[str, set[str] | None],
        processor: StatementProcessor,
    ) -> None:
        """Propagate invalidation to downstream variables.

        Uses **granular invalidation** when possible: if we know exactly
        which symbols changed and which symbols a variable accesses, we
        preserve variables that don't use any changed symbols.
        """
        expanded_changed_symbols = self._expand_changed_symbols(
            old_module_lineages, per_module_changed_symbols, processor.function_tracker,
        )

        vars_to_invalidate: set[str] = set()
        vars_preserved: set[str] = set()
        for var_name, input_map in list(processor.executed_input_lineages.items()):
            decision = self._classify_var_invalidation(
                var_name, input_map, old_module_lineages, expanded_changed_symbols, processor,
            )
            if decision == 'invalidate':
                vars_to_invalidate.add(var_name)
            elif decision == 'preserve':
                vars_preserved.add(var_name)

        for var_name in vars_to_invalidate:
            self._invalidate_var(var_name, processor)
        for var_name in vars_preserved:
            self._register_preserved_var(var_name, old_module_lineages, processor)
        if self._debug and vars_preserved:
            print(f"[GRANULAR] Preserved {len(vars_preserved)} variable(s): {vars_preserved}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_changed_symbols(
        old_module_lineages: dict[str, str],
        per_module_changed_symbols: dict[str, set[str] | None],
        ft: FunctionTracker,
    ) -> dict[str, set[str] | None]:
        """Transitively expand changed symbols via intra-module call deps.

        If ``dep`` changed and ``fun`` calls ``dep``, then ``fun`` is
        also effectively changed.
        """
        expanded: dict[str, set[str] | None] = {}
        for mod_name_key in old_module_lineages:
            raw_syms = per_module_changed_symbols.get(mod_name_key)
            if raw_syms is not None and len(raw_syms) > 0:
                mod = sys.modules.get(mod_name_key)
                mod_file = getattr(mod, '__file__', None) if mod else None
                call_deps = (
                    ft.get_intra_module_call_deps(mod_file) if mod_file else {}
                )
                result = ft.expand_changed_symbols_transitively(
                    raw_syms, call_deps,
                )
                expanded[mod_name_key] = result
                if result != raw_syms:
                    logger.debug(
                        "[GRANULAR] Expanded changed symbols for '%s': %s -> %s",
                        mod_name_key, raw_syms, result,
                    )
            else:
                expanded[mod_name_key] = raw_syms
        return expanded

    @staticmethod
    def _compute_module_lineage_hash(
        mod_name: str,
        file_path: str,
        ft: FunctionTracker,
    ) -> str:
        """Compute a lineage hash for a module including transitive deps.

        The hash incorporates the module's own source file concatenated
        with every local file it transitively depends on (sorted by path
        for determinism).
        """
        hasher = hashlib.sha256()

        if file_path and os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            except OSError as e:
                logger.debug("[MODULE] Could not read module file %r for hash: %s", file_path, e)

        dep_files: set = set()
        for dep_path, parent_mods in ft._dep_file_to_parents.items():
            if mod_name in parent_mods:
                dep_files.add(dep_path)

        for dep_path in sorted(dep_files):
            if os.path.isfile(dep_path):
                try:
                    with open(dep_path, 'rb') as f:
                        hasher.update(f.read())
                except OSError as e:
                    logger.debug("[MODULE] Could not read dep file %r for hash: %s", dep_path, e)

        digest = hasher.hexdigest()
        if not digest or digest == hashlib.sha256(b'').hexdigest():
            digest = hashlib.sha256(os.urandom(32)).hexdigest()
        return digest
