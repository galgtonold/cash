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

import logging
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .lineage_formula import imported_from, module_source_component, read_module_source_hash
from .upstream.mismatch_classifier import import_only

if TYPE_CHECKING:
    from ..tracking.function_tracker import FunctionTracker
    from ._protocols import ShellProtocol
    from .statement import StatementProcessor

logger = logging.getLogger(__name__)


class ModuleInvalidator:
    """Encapsulates module-change detection and lineage invalidation.

    Constructed once by :class:`CashMagics` and reused for the lifetime
    of the session. It holds no state of its own beyond the ``shell`` (needed
    to refresh ``user_ns`` values): what it changes is the session's
    :class:`TrackingState`, and a variable's recorded lineage is dropped
    only through :meth:`StatementProcessor.forget_variable`.
    """

    def __init__(self, shell: ShellProtocol) -> None:
        self._shell = shell

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
        if changed_modules:
            # Makes the upstream simulation re-simulate rather than replay its
            # pre-edit cache -- see TrackingState.module_generation. Taken
            # before the from-imports are cleared below, while they are still
            # recognisable by where they came from.
            state = processor.tracking_state
            state.module_generation += 1
            state.reloaded_names.update(self._names_reaching(changed_modules, state))

        old_module_lineages = self._update_module_lineages(changed_modules, processor)
        self._clear_from_imported_tracking(changed_modules, processor)

        if old_module_lineages:
            self._propagate_module_invalidation(
                old_module_lineages,
                per_module_changed_symbols,
                processor,
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
            new_lineage = self._compute_module_lineage_hash(mod_name, file_path, ft)

            # Every name the notebook bound this module to, not just its own.
            # A cache key is built from the names a statement mentions, so
            # `parsed = tl.parse_headers(corpus)` asks for the lineage of
            # `tl` -- and `import tickets_lib as tl` is what most notebooks
            # write. Updating only `tickets_lib` left `tl` holding the
            # pre-edit hash forever, so the statement's key never moved: the
            # module was reloaded, the badge said so, and the cell returned
            # the pre-edit answer anyway until the kernel was restarted.
            #
            # It also feeds the propagation step below, which matches
            # downstream variables on the lineage they RECORDED for their
            # inputs -- again the alias, never the module's real name.
            #
            # Why three minimal repros missed it, and why this suite did:
            # they all wrote `import mylib`, where the two names coincide.
            for name in self._names_bound_to(mod_name):
                old_lineage = processor.tracking_state.variable_lineage.get(name)
                if old_lineage:
                    old_module_lineages[name] = old_lineage
                lineage = self._lineage_as_imported(name, processor) or new_lineage
                processor.forget_variable(name)
                processor.tracking_state.variable_lineage[name] = lineage
                logger.debug(
                    "[MODULE_INVALIDATE] Updated lineage for %r: %s... -> %s...",
                    name,
                    (old_lineage or "NONE")[:12],
                    lineage[:12],
                )

            processor.recently_reloaded_modules.add(mod_name)

        return old_module_lineages

    def _lineage_as_imported(self, name: str, processor: StatementProcessor) -> str | None:
        """The lineage the import that bound *name* gives it, run again now,
        or None when no import is known for it.

        What a fresh kernel's import computes -- the reload's own file hash
        was a different formula. A statement that reads the module whole (one
        whose closure cannot be bounded: a helper that reads the clock) is
        keyed on this lineage, so everything the session computed after an
        edit was keyed apart from what the next morning looked up: nothing
        restored until a second restart.
        """

        code = processor.tracking_state.executed_cell_codes.get(name)
        value = self._shell.user_ns.get(name)
        if not code or value is None or not import_only(code):
            return None
        try:
            return processor.lineage_builder.lineage_if_rerun(processor.tracking_state, name, value, code)
        except Exception:  # noqa: BLE001 - the reload's own hash is always a valid fallback
            logger.debug("[MODULE] could not re-derive %s's import lineage", name, exc_info=True)
            return None

    def _module_named(self, name: str) -> Any | None:
        """The module *name* refers to, whether it is a real name or an alias."""
        module = sys.modules.get(name)
        if module is not None:
            return module
        user_ns = getattr(self._shell, "user_ns", None)
        candidate = user_ns.get(name) if isinstance(user_ns, dict) else None
        return candidate if isinstance(candidate, ModuleType) else None

    def _names_reaching(self, changed_modules: dict[str, str], state: Any) -> set[str]:
        """Every name a cell can use a changed module through."""
        names: set[str] = set()
        for mod_name in changed_modules:
            names.update(self._names_bound_to(mod_name))
            prefix = mod_name + "."
            for var_name, value in list(self._shell.user_ns.items()):
                owner = getattr(value, "__module__", None)
                if isinstance(owner, str) and (owner == mod_name or owner.startswith(prefix)):
                    names.add(var_name)
            for var_name, src in state.from_import_sources.items():
                if src == mod_name or src.startswith(prefix):
                    names.add(var_name)
        return names

    def _names_bound_to(self, mod_name: str) -> list[str]:
        """The module's own name, plus every namespace alias for it.

        Identity, not `__name__`: ``importlib.reload`` mutates the module in
        place, so a name bound by ``import x as y`` still IS the object in
        ``sys.modules`` after the reload. A name that merely happens to hold
        a different module of the same name is not this module and is left
        alone.

        The module's own name comes first and is always included, even when
        nothing in the namespace is bound to it -- an ``import`` inside a
        function, or a module reached only through another module, still
        needs its lineage updated.
        """
        names = [mod_name]
        module = sys.modules.get(mod_name)
        if module is None:
            return names
        user_ns = getattr(self._shell, "user_ns", None)
        if not isinstance(user_ns, dict):
            return names
        names.extend(
            name
            for name, value in list(user_ns.items())
            if value is module and name != mod_name and not name.startswith("_")
        )
        return names

    # ------------------------------------------------------------------
    # Step 2 — clear execution tracking for from-imports
    # ------------------------------------------------------------------

    def _clear_callable_from_imports(self, mod_name: str, processor: StatementProcessor) -> None:
        """Category 1: clear tracking for callable from-imports from *mod_name*.

        Except a name imported by ``from mod import name`` whose narrowed
        source component -- what ``name`` reaches inside the module -- is the
        same against the reloaded file as the one its lineage was built with.
        Its code did not change, so neither does anything built on it: it is
        refreshed to the reloaded object and keeps its lineage. Dropping it,
        as every name used to be dropped, left `DATA = load(6)` refused as
        "Input variable missing lineage" after an edit to an unrelated
        function in the same file.
        """
        for var_name, var_value in list(self._shell.user_ns.items()):
            if var_name.startswith("_"):
                continue
            value_module = getattr(var_value, "__module__", None)
            if value_module and (value_module == mod_name or value_module.startswith(mod_name + ".")):
                if self._keep_unchanged_from_import(var_name, var_value, processor):
                    continue
                processor.forget_variable(var_name)
                logger.debug(
                    "[MODULE_INVALIDATE] Cleared tracking for from-imported %r (module: %s)", var_name, mod_name
                )

    def _keep_unchanged_from_import(
        self,
        var_name: str,
        var_value: Any,
        processor: StatementProcessor,
    ) -> bool:
        """True -- and the name refreshed -- when its narrowed component still holds.

        Needs all three: a component recorded when the lineage was built (only
        ever a NARROWED one; see ``from_import_components``), the ``from ...
        import`` statement that bound the name, and the same component
        recomputed from that statement against the reloaded file. Anything
        missing or different falls through to clearing, as before.
        """
        state = processor.tracking_state
        recorded = state.from_import_components.get(var_name)
        code = processor.tracking_state.executed_cell_codes.get(var_name)
        if not recorded or not code:
            return False
        try:
            source = imported_from(var_name, code)
            if source is None:
                return False
            current = module_source_component(processor.function_tracker, var_value, var_name, code)
            if current != recorded:
                return False
            module = sys.modules.get(source[0])
            if module is None or not hasattr(module, source[1]):
                return False
            fresh = getattr(module, source[1])
        except Exception:  # noqa: BLE001 - any doubt means clear, the old behaviour
            logger.debug("keeping from-import %s failed", var_name, exc_info=True)
            return False
        # What a re-import would bind, from the module the statement names. For
        # a function it matters beyond the value: the old object's line numbers
        # describe the old file, and its source hash is read by them.
        self._shell.user_ns[var_name] = fresh
        logger.debug("[MODULE_INVALIDATE] Kept %r: what it reaches in the module is unchanged", var_name)
        return True

    def _clear_constant_from_imports(self, mod_name: str, processor: StatementProcessor, reloaded_mod: Any) -> None:
        """Category 2: clear/refresh non-callable from-imports (constants) from *mod_name*."""
        for var_name, src_mod in list(processor.tracking_state.from_import_sources.items()):
            if src_mod != mod_name and not src_mod.startswith(mod_name + "."):
                continue
            # This map holds callables as well as constants, so this pass saw
            # -- and dropped -- every name Category 1 had just kept.
            if self._keep_unchanged_from_import(var_name, self._shell.user_ns.get(var_name), processor):
                continue
            processor.forget_variable(var_name)

            actual_mod = sys.modules.get(src_mod) if src_mod != mod_name else reloaded_mod
            if actual_mod and hasattr(actual_mod, var_name):
                self._shell.user_ns[var_name] = getattr(actual_mod, var_name)
                logger.debug(
                    "[MODULE_INVALIDATE] Refreshed from-imported constant %r (source module: %s)", var_name, src_mod
                )
            else:
                logger.debug(
                    "[MODULE_INVALIDATE] Cleared tracking for from-imported constant %r (source module: %s)",
                    var_name,
                    src_mod,
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
            return "invalidate"
        if len(changed_syms) == 0:
            logger.debug("[GRANULAR] Preserving %r: no symbols changed in %r", var_name, input_var)
            return "preserve"
        var_attrs = processor.tracking_state.module_attribute_deps.get(var_name, {}).get(input_var)
        if var_attrs:
            if var_attrs & changed_syms:
                logger.debug("[GRANULAR] Invalidating %r: uses changed symbols %s", var_name, var_attrs & changed_syms)
                return "invalidate"
            logger.debug("[GRANULAR] Preserving %r: uses %s, changed: %s", var_name, var_attrs, changed_syms)
            return "preserve"
        logger.debug("[GRANULAR] Invalidating %r: unknown attribute access on %r", var_name, input_var)
        return "invalidate"

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
        processor.forget_variable(var_name)
        # The value is still in memory, built by the pre-edit module. Dropping
        # its lineage makes a READER of it recompute, but a cell further down
        # reading only something built from it compared lineages and saw
        # nothing to compare, and exported the pre-edit numbers. Ask
        # for its binding to be re-run instead -- TrackingState.rerun_bindings.
        if var_name in self._shell.user_ns:
            processor.tracking_state.rerun_bindings.add(var_name)
        logger.debug("[MODULE_INVALIDATE] Cleared lineage for dependent var %r", var_name)

    def _register_preserved_var(
        self,
        var_name: str,
        old_module_lineages: dict[str, str],
        processor: StatementProcessor,
    ) -> None:
        """Register a granularly-preserved variable for deferred lineage update."""
        input_map = processor.tracking_state.executed_input_lineages.get(var_name, {})
        for mod_name_key in old_module_lineages:
            if mod_name_key in input_map:
                processor.tracking_state.granular_preserved_vars.setdefault(mod_name_key, set()).add(var_name)
                logger.debug("[GRANULAR] Registered %r for deferred lineage update on %r", var_name, mod_name_key)

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
            old_module_lineages,
            per_module_changed_symbols,
            processor.function_tracker,
        )

        vars_to_invalidate: set[str] = set()
        vars_preserved: set[str] = set()
        for var_name, input_map in list(processor.tracking_state.executed_input_lineages.items()):
            decision = self._classify_var_invalidation(
                var_name,
                input_map,
                old_module_lineages,
                expanded_changed_symbols,
                processor,
            )
            if decision == "invalidate":
                vars_to_invalidate.add(var_name)
            elif decision == "preserve":
                vars_preserved.add(var_name)

        for var_name in vars_to_invalidate:
            self._invalidate_var(var_name, processor)
        for var_name in vars_preserved:
            self._register_preserved_var(var_name, old_module_lineages, processor)
        if vars_preserved:
            logger.debug("[GRANULAR] Preserved %d variable(s): %s", len(vars_preserved), vars_preserved)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_changed_symbols(
        self,
        old_module_lineages: dict[str, str],
        per_module_changed_symbols: dict[str, set[str] | None],
        ft: FunctionTracker,
    ) -> dict[str, set[str] | None]:
        """Transitively expand changed symbols via intra-module call deps.

        If ``dep`` changed and ``fun`` calls ``dep``, then ``fun`` is
        also effectively changed.

        A key here may be an alias (``tl`` for ``tickets_lib``), because
        that is how downstream variables recorded the dependency. Both the
        symbol set and the module file are looked up under the module's REAL
        name: keyed by the alias they come back empty, which is not wrong --
        an empty set means "invalidate everything" -- but it throws away the
        granularity that keeps unrelated variables alive.
        """
        expanded: dict[str, set[str] | None] = {}
        for mod_name_key in old_module_lineages:
            mod = self._module_named(mod_name_key)
            real_name = getattr(mod, "__name__", mod_name_key) if mod else mod_name_key
            raw_syms = per_module_changed_symbols.get(mod_name_key)
            if raw_syms is None:
                raw_syms = per_module_changed_symbols.get(real_name)
            if raw_syms is not None and len(raw_syms) > 0:
                mod_file = getattr(mod, "__file__", None) if mod else None
                call_deps = ft.get_intra_module_call_deps(mod_file) if mod_file else {}
                result = ft.expand_changed_symbols_transitively(
                    raw_syms,
                    call_deps,
                )
                expanded[mod_name_key] = result
                if result != raw_syms:
                    logger.debug(
                        "[GRANULAR] Expanded changed symbols for '%s': %s -> %s",
                        mod_name_key,
                        raw_syms,
                        result,
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
        """The lineage a changed module takes when no import is known to give
        it one: its source identity (``read_module_source_hash``) together with
        every local file it transitively depends on -- the same identity the
        module's source component of an import's lineage carries, so a comment
        or a reformat moves neither.

        A module whose file cannot be read gets a fixed marker for that, which
        no readable version of it can share.
        """
        dep_files = {dep_path for dep_path, parent_mods in ft.dep_file_to_parents.items() if mod_name in parent_mods}
        digest = read_module_source_hash(file_path, dep_files) if file_path else None
        if digest is None:
            logger.debug("[MODULE] Could not read module file %r for its lineage", file_path)
            return f"unreadable-module:{mod_name}"
        return digest
