"""Restoring a simulated statement's outputs from the cache.

The upstream check restores what it can instead of re-running it:
:class:`CacheRestorer` loads a statement's cached outputs into the namespace
and records their producer, probes the checked cell's own statements for cache
hits that make an upstream re-run unnecessary, and reports the statements the
plan leaves alone. It keys each statement exactly as the forward simulation
(:class:`VirtualLineage`) does, through it.
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Any

from ..._clock import perf_counter as _perf_counter
from ..._paths import resolve_file_dep_path
from ...analysis.ast_util import parse_cached
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source, statement_code
from ...tracking.file_dep_snapshot import snapshot_is_fresh
from ..cache_key import (
    CacheKeyContext,
    compute_cache_key,
)
from ..cache_status import CacheStatus
from ..call_refs import resolve_call_refs
from ..control_structures import is_control_structure
from ..lineage_formula import (
    key_hidden_reads,
)
from ._types import key_inputs, key_lineages

if TYPE_CHECKING:
    from .virtual_lineage import VirtualLineage


__all__ = ["CacheRestorer", "lineage_confirmed_vars", "lineage_conflict"]

logger = logging.getLogger(__name__)


# Stands in the namespace for a variable the forward probe found a current-cell
# cache hit for: a statement whose input is not in the namespace is never
# looked up (``cacheability_decision._has_missing_lineage``), so without it the
# hit that restores the variable could not happen. The restore replaces it.
_FORWARD_PROBE_PLACEHOLDER = object()


def lineage_conflict(
    metadata: dict[str, Any], file_deps: dict[str, Any], expected_lineages: dict[str, str] | None
) -> str | None:
    """An output whose cached lineage is not the one the simulation expects,
    or None.

    Not compared for an entry with file dependencies: its lineages fold the
    files' state, which the freshness check has already judged.
    """
    if file_deps or not expected_lineages or "output_lineages" not in metadata:
        return None
    cached = metadata["output_lineages"]
    for var, expected in expected_lineages.items():
        if cached.get(var) and cached[var] != expected:
            return var
    return None


def lineage_confirmed_vars(
    metadata: dict[str, Any], file_deps: dict[str, Any], expected_lineages: dict[str, str] | None
) -> frozenset[str]:
    """Outputs whose cached lineage was positively matched against the expected one.

    Only these may have an EMPTY cached value restored over a non-empty live
    one: a confirmed lineage makes the empty value the correct result (a
    filter that legitimately matched nothing), not a corrupt entry. Where
    :func:`lineage_conflict` compares nothing, nothing is confirmed.
    """
    if file_deps or not expected_lineages or "output_lineages" not in metadata:
        return frozenset()
    cached = metadata["output_lineages"]
    return frozenset(var for var, expected in expected_lineages.items() if cached.get(var) and cached[var] == expected)


class CacheRestorer:
    """Restores upstream statements from the cache, keyed as simulated."""

    def __init__(self, virtual_lineage: VirtualLineage) -> None:
        self.virtual_lineage = virtual_lineage
        #: Names the forward probe bound to ``_FORWARD_PROBE_PLACEHOLDER``.
        self._probe_placeholders: set[str] = set()

    def try_virtual_restore(
        self,
        stmt_code: str,
        outputs: set[str],
        inputs: set[str],
        input_hashes: dict[str, str],
        virtual_modules: set[str] | None = None,
        expected_lineages: dict[str, str] | None = None,
    ) -> tuple[set[str], float, float]:
        """Attempt to restore a statement using virtual input hashes.

        Directly queries backend and updates memory if successful.

        Returns:
            Tuple of (set of variables successfully restored, restore_time_seconds, saved_time_seconds).
        """
        start_time = _perf_counter()

        if not self.virtual_lineage.cash_instance:
            return set(), 0.0, 0.0

        if virtual_modules is None:
            virtual_modules = set()

        try:
            # 1. Reconstruct Cache Key using the unified function.
            # Pass input_hashes as virtual_lineage so the unified function
            # can look up lineages for inputs that aren't in variable_lineage yet.
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                key_inputs(inputs, input_hashes),
                ctx=CacheKeyContext(
                    variable_lineage=self.virtual_lineage.tracking_state.variable_lineage,
                    user_ns=self.virtual_lineage.shell.user_ns,
                    function_tracker=self.virtual_lineage.function_tracker,
                    virtual_lineage=key_lineages(input_hashes),
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.virtual_lineage.compute_hash_fn,
                    virtual_callables=self.virtual_lineage._virtual_callables,
                ),
                outputs=outputs,
            )

            logger.debug("[UPSTREAM] Attempting virtual restore Key: %s", cache_key)

            # 2. Query Memory Backend first (fastest) - Or just generic backend
            metadata, cached_data = self.virtual_lineage.cash_instance.backend.get(cache_key)
            if cached_data is not None:
                # Call results the entry refers to rather than copies (call_refs).
                cached_data = resolve_call_refs(cached_data, self.virtual_lineage.cash_instance.backend)

            # Extract saved execution time
            saved_time = metadata.get("execution_time", 0.0) if metadata else 0.0

            if metadata and cached_data is not None:
                # 3. Check file dependencies (Critical!)
                file_deps = metadata.get("file_dependencies", {})
                fresh, stale = snapshot_is_fresh(file_deps)
                if not fresh:
                    logger.debug("[UPSTREAM] Restore failed: stale file dependency (%s)", stale)
                    return set(), _perf_counter() - start_time, 0.0

                conflict = lineage_conflict(metadata, file_deps, expected_lineages)
                if conflict is not None:
                    logger.debug("[UPSTREAM] Restore failed: lineage mismatch for %s", conflict)
                    return set(), _perf_counter() - start_time, 0.0

                # 4. Success! Restore into shell.
                # Cache stores variables under 'variables' key (see StatementStore._payload)
                variables_to_restore = cached_data.get("variables", {})
                restored_vars = self._restore_vars_from_cache(
                    variables_to_restore,
                    metadata,
                    lineage_confirmed_vars(metadata, file_deps, expected_lineages),
                )
                self._update_tracking_after_restore(restored_vars, metadata, input_hashes)
                return restored_vars, _perf_counter() - start_time, saved_time

        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.debug("[UPSTREAM] Virtual restore error: %s", e)

        return set(), _perf_counter() - start_time, 0.0

    def _restore_vars_from_cache(
        self,
        variables_to_restore: dict,
        metadata: dict,
        lineage_confirmed: frozenset[str] = frozenset(),
    ) -> set[str]:
        """Restore variables into shell namespace.  Returns the set of restored var names.

        ``lineage_confirmed`` names the variables whose cached lineage matched
        the expected one; it defaults to empty so any caller that cannot
        establish that keeps the conservative behaviour.
        """
        restored_vars: set[str] = set()
        for var, val in variables_to_restore.items():
            if var in self.virtual_lineage.shell.user_ns and var not in lineage_confirmed:
                # Refuse to let an empty cached value clobber live data UNLESS
                # its lineage was confirmed above. Without that confirmation an
                # empty value is indistinguishable from a corrupt entry, and
                # overwriting 1000 rows with 0 is the more expensive mistake.
                # With it, blocking the restore is what costs correctness: the
                # statement re-executes forever and a legitimately-empty result
                # can never be served from cache.
                existing = self.virtual_lineage.shell.user_ns[var]
                try:
                    if len(existing) > 0 and len(val) == 0:
                        logger.debug(
                            "[UPSTREAM] Restore BLOCKED for '%s': cached value is empty "
                            "but in-memory has %d items, and its lineage is unconfirmed. "
                            "Keeping in-memory value.",
                            var,
                            len(existing),
                        )
                        continue
                except (TypeError, AttributeError):
                    pass
            self.virtual_lineage.shell.user_ns[var] = val
            restored_vars.add(var)
            if "output_lineages" in metadata:
                new_lineage = metadata["output_lineages"].get(var)
                if var in self.virtual_lineage.tracking_state.lineage and new_lineage is not None:
                    # Recorded with the value, so the live object carries
                    # _cash_lineage_hash too.
                    self.virtual_lineage.tracking_state.lineage.record(var, new_lineage, value=val)
                else:
                    # Variable wasn't tracked in the lineage store before, but
                    # we still want the attribute attached so future cache-key
                    # computation finds it via the ladder fallback.
                    try:
                        val._cash_lineage_hash = new_lineage
                        val._cash_lineage_src = "statement"
                    except (AttributeError, TypeError):
                        logger.debug(
                            "Cannot attach _cash_lineage_hash to restored variable %s",
                            var,
                        )
        return restored_vars

    def _update_tracking_after_restore(
        self,
        restored_vars: set[str],
        metadata: dict,
        input_hashes: dict[str, str],
    ) -> None:
        """Record what produced each of *restored_vars*, as running the
        statement would have: its lineage, code, code hash, input lineages
        and file dependencies."""
        state = self.virtual_lineage.tracking_state
        output_lineages = metadata.get("output_lineages", {}) if "output_lineages" in metadata else {}
        stored_code = metadata.get("code")
        stored_hash = metadata.get("source_hash")

        # Resolve file deps once.
        resolved_paths: set[str] = set()
        file_deps_meta = metadata.get("file_dependencies", {})
        if file_deps_meta:
            for stored_path in file_deps_meta:
                resolved = resolve_file_dep_path(stored_path)
                if resolved is not None:
                    resolved_paths.add(resolved)

        for var in restored_vars:
            lin = output_lineages.get(var) if output_lineages else None
            if lin is not None:
                state.lineage.record(var, lin)
            if stored_code:
                state.executed_cell_codes[var] = stored_code
            if stored_hash:
                state.executed_cell_hashes.setdefault(var, set()).add(stored_hash)
            if input_hashes:
                state.executed_input_lineages[var] = dict(input_hashes)
            if resolved_paths:
                state.executed_file_deps.setdefault(var, set()).update(resolved_paths)

    def eliminate_broken_vars_via_current_cell_probe(
        self,
        broken_vars: set[str],
        notebook_cells: list[str],
        current_cell_idx: int,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> None:
        """Remove variables from *broken_vars* that would be restored by current cell cache hits.

        When a broken variable (e.g. ``df``) is absent from memory but the
        current cell contains a statement that both uses it as input AND
        produces it as output (e.g. ``df['col'] = heavy_computation(df)``),
        and that statement would be a cache hit on DISK, then the cache
        restore will inject both the output variable and its data into
        memory.  In that case we do NOT need upstream re-execution to
        produce the broken variable — the cache restore will provide it.

        This avoids expensive upstream re-execution for scenarios like
        kernel restarts where heavy current-cell statements are on disk.
        """
        if not self.virtual_lineage.cash_instance or not broken_vars:
            return

        try:
            raw_cell = notebook_cells[current_cell_idx]
        except IndexError:
            return
        tree = parse_cell_source(raw_cell)
        if tree is None:
            return
        clean_cell = clean_cell_source(raw_cell)
        # Track which broken vars are resolved by forward cache hits.
        # We simulate forward through the current cell's statements:
        # if a statement (a) would cache-hit and (b) its outputs overlap
        # with broken_vars, those outputs become available in memory.
        resolved_by_cache = set()
        # A broken var a statement that misses reads before any hit restores it
        # is needed from upstream: a later hit would come too late.
        needed_first: set[str] = set()
        occurrences: dict[str, int] = {}

        for node in tree.body:
            if is_control_structure(node):
                # Too complex to probe: what it reads is needed as it runs.
                try:
                    reads, _ = CodeAnalyzer.analyze_code_block(ast.unparse(node))
                except (SyntaxError, ValueError, TypeError):
                    reads = set(broken_vars)
                needed_first |= reads & broken_vars
                continue

            # The statement's key, built as ``_update_virtual_lineage`` builds
            # it (and so as the runtime does): its text with an expression's
            # trailing ``;``, its occurrence in the cell, its reads with the
            # hidden ones (an RNG a draw reads), and its writes with the
            # globals its callees write.
            try:
                stmt_code = statement_code(node, clean_cell)
            except (ValueError, TypeError):
                continue
            occurrence_index = occurrences.get(stmt_code, 0)
            occurrences[stmt_code] = occurrence_index + 1

            effects, inputs, outputs = self.virtual_lineage._statement_reads_writes(
                stmt_code, parse_cached(stmt_code), virtual_modules
            )
            outputs |= effects.callee_globals
            unresolved = inputs & (broken_vars - resolved_by_cache - needed_first)
            if not unresolved:
                continue

            # Probe the cache's metadata only: nothing is restored here.
            try:
                cache_key, _, _, _, _ = compute_cache_key(
                    stmt_code,
                    inputs | key_hidden_reads(stmt_code, self.virtual_lineage.tracking_state),
                    ctx=CacheKeyContext(
                        variable_lineage=self.virtual_lineage.tracking_state.variable_lineage,
                        user_ns=self.virtual_lineage.shell.user_ns,
                        function_tracker=self.virtual_lineage.function_tracker,
                        virtual_lineage=virtual_lineage,
                        virtual_modules=virtual_modules,
                        compute_hash_fn=self.virtual_lineage.compute_hash_fn,
                        virtual_callables=self.virtual_lineage._virtual_callables,
                    ),
                    outputs=outputs,
                    occurrence_index=occurrence_index,
                )
                metadata = self.virtual_lineage._get_metadata_only(cache_key)
            except (KeyError, TypeError, ValueError, OSError):
                metadata = None
            # A metadata-only record (the value stayed in RAM, or was too large
            # to write) restores nothing; file deps must still be valid (mtime +
            # size, both forms -- see _validate_file_freshness for rationale).
            if (
                not metadata
                or metadata.get("metadata_only")
                or not self.virtual_lineage._validate_file_freshness(
                    metadata.get("file_dependencies", {}), memo_key=cache_key
                )
            ):
                needed_first |= unresolved
                continue

            # Cache hit! Its restore puts back what its entry recorded,
            # including any broken vars among them.
            produced = outputs & unresolved & set(metadata.get("output_lineages") or ())
            if not produced:
                continue
            resolved_by_cache.update(produced)
            # The runtime keys the statement with these lineages, as the
            # simulation did: record them, so the restore finds the entry, and
            # hold each name's place until the restore fills it.
            for var in produced:
                if var in virtual_lineage:
                    # Recorded now, so later statements probing the cache key
                    # with it.
                    self.virtual_lineage.tracking_state.lineage.record(var, virtual_lineage[var])
                if var not in self.virtual_lineage.shell.user_ns:
                    self.virtual_lineage.shell.user_ns[var] = _FORWARD_PROBE_PLACEHOLDER
                    self._probe_placeholders.add(var)
            logger.debug(
                "[UPSTREAM] Forward probe: cache hit for '%s' resolves broken vars: %s",
                stmt_code[:50],
                produced,
            )

        if resolved_by_cache:
            broken_vars -= resolved_by_cache
            logger.debug(
                "[UPSTREAM] Forward probe eliminated %d broken vars: %s. Remaining: %s",
                len(resolved_by_cache),
                resolved_by_cache,
                broken_vars,
            )

    def drop_probe_placeholders(self) -> None:
        """Unbind the names the forward probe held that no restore filled.

        The probe binds a placeholder for the cell it checked; its restore
        replaces it as the cell runs. One still bound afterwards (the restore
        failed) is not a value: left in place it would read as present to the
        next check and to the user. It goes, with the lineage recorded for it.
        """
        for var in self._probe_placeholders:
            if self.virtual_lineage.shell.user_ns.get(var) is _FORWARD_PROBE_PLACEHOLDER:
                del self.virtual_lineage.shell.user_ns[var]
                self.virtual_lineage.tracking_state.lineage.discard(var)
        self._probe_placeholders.clear()

    def collect_skipped_statement_metrics(
        self,
        simulation_trace: list,
        stmts_to_run_indices: list[int],
        restored_statements_info: list,
        virtual_modules: set[str],
        stmt_lookup_times: dict[str, float],
    ) -> list[dict]:
        """Identify implicitly skipped statements and collect their cache metrics.

        Skipped statements are dependencies of restored variables that were
        neither scheduled for execution nor explicitly restored. Returns a list
        of metric dicts to be appended to ``restored_statements_info``.
        """
        executed_indices = set(stmts_to_run_indices)
        restored_indices = set()
        restored_outputs = set()
        for info in restored_statements_info:
            if "position" in info:
                restored_indices.add(info["position"])
                if "restored_vars" in info:
                    restored_outputs.update(info["restored_vars"])

        dependency_chain: set[int] = set()
        if restored_outputs:
            needed = set(restored_outputs)
            for i in range(len(simulation_trace) - 1, -1, -1):
                entry = simulation_trace[i]
                if entry.outputs & needed:
                    dependency_chain.add(i)
                    needed.update(entry.inputs)

        skipped_metrics: list[dict] = []
        for i, entry in enumerate(simulation_trace):
            if i in executed_indices or i in restored_indices or i not in dependency_chain:
                continue
            metric = self._skipped_stmt_metric(
                i, entry.stmt_code, entry.outputs, entry.inputs, entry.input_hashes, virtual_modules
            )
            if metric is not None:
                skipped_metrics.append(metric)
        return skipped_metrics

    def _skipped_stmt_metric(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        inputs: set[str],
        input_hashes: dict[str, str],
        virtual_modules: set[str],
    ) -> dict | None:
        """Return a metric dict for a single skipped statement, or ``None`` on error."""
        logger.debug("[UPSTREAM] Checking skipped stmt [%d]: %.30s...", i, stmt_code)
        try:
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                key_inputs(inputs, input_hashes),
                ctx=CacheKeyContext(
                    variable_lineage=self.virtual_lineage.tracking_state.variable_lineage,
                    user_ns=self.virtual_lineage.shell.user_ns,
                    function_tracker=self.virtual_lineage.function_tracker,
                    virtual_lineage=key_lineages(input_hashes),
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.virtual_lineage.compute_hash_fn,
                    virtual_callables=self.virtual_lineage._virtual_callables,
                ),
                outputs=outputs,
            )
            metadata = self.virtual_lineage._get_metadata_only(cache_key)
            if metadata:
                saved_time = metadata.get("execution_time", 0.0)
                is_metadata_only = metadata.get("metadata_only", False)
                logger.debug(
                    "[UPSTREAM] Skipped stmt [%d] hit cache. Saved: %ss (metadata_only=%s)",
                    i,
                    saved_time,
                    is_metadata_only,
                )
                entry: dict = {
                    "code": stmt_code,
                    "status": CacheStatus.SKIPPED,
                    "saved_time": saved_time,
                    "is_upstream": True,
                    "source": "Skipped",
                    "position": i,
                    "has_cache": True,
                }
                if "storage" in metadata:
                    entry["storage"] = metadata["storage"]
                return entry
            logger.debug("[UPSTREAM] Skipped stmt [%d] miss cache: %s. Key: %s", i, stmt_code[:60], cache_key)
            return {
                "code": stmt_code,
                "status": CacheStatus.SKIPPED,
                "saved_time": 0.0,
                "is_upstream": True,
                "source": "Skipped",
                "position": i,
                "has_cache": False,
            }
        except (KeyError, TypeError, OSError, ValueError) as e:
            logger.debug("[UPSTREAM] Error checking skipped stmt: %s", e)
            return None
