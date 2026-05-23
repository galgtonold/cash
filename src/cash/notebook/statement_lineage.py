"""Lineage computation for statement outputs.

Owns the operation "produce the lineage hash for each output variable
of a statement, plus all the bookkeeping that goes with it" — input
lineage assembly, function-source hashing, module-source hashing
(both ``import X`` and ``from X import Y`` cases), per-variable
content hashing, and granular module-attribute dependency tracking.

Single public entry: :meth:`StatementLineageBuilder.capture_and_track_variables`,
plus :meth:`build_output_lineages` for the cache-write side.

**Anti-god-class rule (load-bearing):** this module computes lineage
hashes and writes them through :class:`LineageStore` and the
tracking-state dicts.  It does **not** decide caching policy (skip
checks, freshness), execute statements, or replay output.  Those are
::class:`CacheFreshnessChecker`'s, ``StatementProcessor``'s, and
:class:`StatementRestorer`'s jobs, respectively.

Carries the heaviest state surface of the four siblings — it needs
many tracking-state dict refs plus processor-owned dicts
(``_granular_preserved_vars``, ``module_attribute_deps``,
``from_import_sources``).  Treat that as the cost of consolidating the
lineage-building logic into one cohesive class.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import pickle
import sys
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .statement_file_deps import compute_file_hash_component, read_module_source_hash

if TYPE_CHECKING:
    from ._protocols import ShellProtocol, TrackingState
    from .function_tracker import FunctionTracker
    from .statement_file_deps import StatementFileDeps

logger = logging.getLogger(__name__)


class StatementLineageBuilder:
    """Compute + record lineage hashes for statement outputs.

    Holds aliased references to many tracking-state dicts plus three
    processor-owned dicts (``_granular_preserved_vars``,
    ``module_attribute_deps``, ``from_import_sources``).  The
    processor-owned dicts are passed by reference at construction; the
    tracking-state dicts are aliased via :meth:`set_tracking_state`.
    """

    def __init__(
        self,
        shell: 'ShellProtocol',
        tracking_state: 'TrackingState',
        function_tracker: 'FunctionTracker',
        file_deps: 'StatementFileDeps',
        granular_preserved_vars: dict[str, set[str]],
        module_attribute_deps: dict[str, dict[str, set[str]]],
        from_import_sources: dict[str, str],
        compute_hash: Callable[[Any], str] | None = None,
        debug: bool = False,
    ) -> None:
        self.shell = shell
        self.function_tracker = function_tracker
        self._file_deps = file_deps
        self._granular_preserved_vars = granular_preserved_vars
        self.module_attribute_deps = module_attribute_deps
        self.from_import_sources = from_import_sources
        self.compute_hash = compute_hash
        self.debug = debug
        self.set_tracking_state(tracking_state)

    def set_tracking_state(self, state: 'TrackingState') -> None:
        """Re-wire individual tracking-state dict refs (alias pattern)."""
        self.variable_lineage = state.variable_lineage
        self.executed_input_lineages = state.executed_input_lineages
        self.lineage = state.lineage
        self.executed_cell_hashes = state.executed_cell_hashes
        self.executed_cell_codes = state.executed_cell_codes
        self.variable_sources = state.variable_sources
        self.variable_hashes = state.variable_hashes
        self.current_session_hashes = state.current_session_hashes

    # ------------------------------------------------------------------
    # Public entries
    # ------------------------------------------------------------------

    def capture_and_track_variables(
        self,
        outputs: set[str],
        inputs: set[str],
        code: str,
        source_hash: str,
        cache_key: str,
        accessed_files: set[str] | None = None,
        tree: ast.Module | None = None,
    ) -> dict[str, Any]:
        """Capture output variables, compute their lineage, and update tracking state.

        Returns ``{var_name: value}`` for the captured outputs.
        """
        captured_vars: dict[str, Any] = {}
        user_ns = self.shell.user_ns

        file_hash_component = ""
        if accessed_files:
            file_hash_component = compute_file_hash_component(accessed_files)

        for var_name in outputs:
            if var_name not in user_ns:
                continue
            value = user_ns[var_name]
            captured_vars[var_name] = value

            input_lineage_hashes, input_lineage_map = self._build_input_lineages(inputs, user_ns)
            self.executed_input_lineages[var_name] = input_lineage_map

            func_lineage_component = ""
            func_source_hashes = self.function_tracker.get_callable_source_hashes(inputs, user_ns)
            if func_source_hashes:
                func_parts = [f"{k}:{v}" for k, v in sorted(func_source_hashes.items())]
                func_lineage_component = ":" + ":".join(func_parts)

            # Include tracked module source hash in lineage.
            module_lineage_component = self._compute_module_lineage_component(
                value, var_name, code, tree
            )

            # Compute lineage hash for the output variable
            lineage_str = f"{source_hash}:{':'.join(sorted(input_lineage_hashes))}{file_hash_component}{func_lineage_component}{module_lineage_component}"
            output_lineage_hash = hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()

            # Record via LineageStore so the dict entry and ``_cash_lineage_hash``
            # are written together and cannot drift.
            self.lineage.record(var_name, output_lineage_hash, value=value)

            self._apply_granular_module_update(var_name, value, output_lineage_hash)

            if var_name not in self.executed_cell_hashes:
                self.executed_cell_hashes[var_name] = set()
            self.executed_cell_hashes[var_name].add(source_hash)

            self.executed_cell_codes[var_name] = code

            self._update_module_attribute_deps(var_name, code, user_ns)
            self._update_variable_content_hashes(var_name, value, output_lineage_hash)

            self.variable_sources[var_name] = cache_key

            self._file_deps.update_for_var(var_name, accessed_files, inputs, value)

        return captured_vars

    def build_output_lineages(self, outputs: set[str]) -> dict[str, str]:
        """Collect ``{var: lineage_hash}`` for all outputs that have a lineage."""
        return {v: self.variable_lineage[v] for v in outputs if v in self.variable_lineage}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_input_lineages(
        self, inputs: set[str], user_ns: dict,
    ) -> tuple[list[str], dict[str, str]]:
        """Build input lineage hashes list and map for a set of input variables."""
        input_lineage_hashes: list[str] = []
        input_lineage_map: dict[str, str] = {}
        for input_var in inputs:
            if input_var in self.variable_lineage:
                lineage = self.variable_lineage[input_var]
                input_lineage_hashes.append(lineage)
                input_lineage_map[input_var] = lineage
            elif input_var in user_ns:
                try:
                    lineage = self.compute_hash(user_ns[input_var])
                    input_lineage_hashes.append(lineage)
                    input_lineage_map[input_var] = lineage
                except (TypeError, ValueError, AttributeError, pickle.PicklingError) as e:
                    if self.debug:
                        logger.warning("Warning: Could not hash input '%s' for lineage: %s", input_var, e)
        return input_lineage_hashes, input_lineage_map

    def _apply_granular_module_update(
        self, var_name: str, value: Any, output_lineage_hash: str,
    ) -> None:
        """Apply deferred granular lineage update when a tracked module is re-imported."""
        if isinstance(value, types.ModuleType) and var_name in self._granular_preserved_vars:
            preserved = self._granular_preserved_vars.pop(var_name)
            for pv in preserved:
                pv_inputs = self.executed_input_lineages.get(pv)
                if pv_inputs is not None and var_name in pv_inputs:
                    pv_inputs[var_name] = output_lineage_hash
                    if self.debug:
                        logger.debug("[GRANULAR] Deferred update: '%s'.'%s' -> %s...", pv, var_name, output_lineage_hash[:12])

    def _update_module_attribute_deps(
        self, var_name: str, code: str, user_ns: dict,
    ) -> None:
        """Update granular module attribute dependency tracking for *var_name*."""
        try:
            attr_accesses = self.function_tracker.extract_module_attribute_accesses(code)
            mod_deps: dict[str, set[str]] = {}
            for input_name, attrs in attr_accesses.items():
                input_val = user_ns.get(input_name)
                if isinstance(input_val, types.ModuleType) and input_name in self.function_tracker._tracked_modules:
                    mod_deps[input_name] = attrs
            if mod_deps:
                self.module_attribute_deps[var_name] = mod_deps
            elif var_name in self.module_attribute_deps:
                del self.module_attribute_deps[var_name]
        except (AttributeError, TypeError, ValueError, SyntaxError):
            logger.debug("[PROCESSOR] Module attribute tracking failed for '%s', falling back to full invalidation", var_name)
            self.module_attribute_deps.pop(var_name, None)

    def _update_variable_content_hashes(
        self, var_name: str, value: Any, output_lineage_hash: str,
    ) -> None:
        """Update variable_hashes and current_session_hashes for *var_name*."""
        type_name = type(value).__name__
        if type_name in ('DataFrame', 'Series', 'ndarray'):
            # For large objects, use lineage hash as proxy for content hash
            if var_name not in self.variable_hashes:
                self.variable_hashes[var_name] = set()
            self.variable_hashes[var_name].add(output_lineage_hash)
            self.current_session_hashes[var_name] = output_lineage_hash
        elif self.compute_hash:
            try:
                content_hash = self.compute_hash(value)
                if var_name not in self.variable_hashes:
                    self.variable_hashes[var_name] = set()
                self.variable_hashes[var_name].add(content_hash)
                self.current_session_hashes[var_name] = content_hash
            except (TypeError, ValueError, AttributeError, pickle.PicklingError) as e:
                if self.debug:
                    logger.debug("[CACHE DEBUG] Could not hash captured variable '%s': %s", var_name, e)

    def _compute_module_lineage_component(
        self,
        value: Any,
        var_name: str,
        code: str,
        tree: ast.Module | None = None,
    ) -> str:
        """Compute the module source hash component for lineage tracking.

        Handles three cases:
        1. Direct module import (``import X``): hash the module source + deps.
        2. Callable from a tracked module (``from X import func``): hash its
           source module.
        3. Non-callable from a tracked module (``from X import CONST``): parse
           the AST to discover the source module and hash it.

        Returns a lineage string fragment like ``:mod_src:<hash>`` or ``""``.
        """
        if isinstance(value, types.ModuleType):
            mod_file = getattr(value, '__file__', None)
            if not (mod_file and os.path.isfile(mod_file) and var_name in self.function_tracker._tracked_modules):
                return ""
            dep_files = {
                dep_path
                for dep_path, _ in self.function_tracker._dep_file_to_parents.items()
                if var_name in self.function_tracker._dep_file_to_parents[dep_path]
            }
            mod_source_hash = read_module_source_hash(mod_file, dep_files)
            return f":mod_src:{mod_source_hash}" if mod_source_hash else ""

        if callable(value):
            obj_module = getattr(value, '__module__', None)
            if not (obj_module and obj_module in self.function_tracker._tracked_modules):
                return ""
            self.from_import_sources[var_name] = obj_module
            mod_obj = sys.modules.get(obj_module)
            mod_file = getattr(mod_obj, '__file__', None) if mod_obj else None
            if not (mod_file and os.path.isfile(mod_file)):
                return ""
            mod_source_hash = read_module_source_hash(mod_file)
            if not mod_source_hash:
                return ""
            if self.debug:
                logger.debug("[CACHE DEBUG] Including module source hash for '%s' from '%s': %s...", var_name, obj_module, mod_source_hash[:12])
            return f":from_mod_src:{mod_source_hash}"

        # Non-callable: parse the AST to find the source module.
        return self._resolve_import_module_lineage(var_name, code, tree)

    def _lookup_from_import_mod_hash(self, var_name: str, from_mod: str) -> str:
        """Return a ``:from_mod_src:<hash>`` lineage fragment for a tracked from-import module."""
        self.from_import_sources[var_name] = from_mod
        if from_mod not in self.function_tracker._tracked_modules:
            return ""
        mod_obj = sys.modules.get(from_mod)
        mod_file = getattr(mod_obj, '__file__', None) if mod_obj else None
        if not (mod_file and os.path.isfile(mod_file)):
            return ""
        mod_source_hash = read_module_source_hash(mod_file)
        if not mod_source_hash:
            return ""
        if self.debug:
            logger.debug("[CACHE DEBUG] Including module source hash for constant '%s' from '%s': %s...", var_name, from_mod, mod_source_hash[:12])
        return f":from_mod_src:{mod_source_hash}"

    def _resolve_import_module_lineage(
        self,
        var_name: str,
        code: str,
        tree: ast.Module | None = None,
    ) -> str:
        """Resolve module lineage for a non-callable ``from X import Y``."""
        try:
            tree_check = tree if tree is not None else ast.parse(code.strip())
        except SyntaxError:
            if self.debug:
                logger.debug("[PROCESSOR] Failed to parse import for '%s'", var_name)
            return ""

        for node in tree_check.body:
            if not (isinstance(node, ast.ImportFrom) and node.module):
                continue
            for alias in node.names:
                imported_name = alias.asname or alias.name
                if imported_name != var_name:
                    continue
                return self._lookup_from_import_mod_hash(var_name, node.module)
        return ""
