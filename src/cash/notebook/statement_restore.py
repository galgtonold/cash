"""Statement-level cache restoration.

Owns the operation "hydrate a statement's outputs from a cached
payload" — write the variables into ``user_ns``, reconstitute lineage
+ source tracking, replay captured stdout / stderr / rich outputs.

Single public entry: :meth:`StatementRestorer.restore_from_cache`,
plus :meth:`StatementRestorer.persist_metadata_only` for the
small-but-cacheable case where only metadata gets persisted.

**Distinct from the variable-granular Restorer.**  See CONTEXT.md:

* :class:`StatementRestorer` (this module) — restores a statement's
  *outputs as a unit*: multiple vars + replayed display data + RNG
  state.  Triggered on a cache *hit* for a freshly-running statement.
* :class:`Restorer` (``restore.py``) — restores **one variable** from
  cache so it's present in ``user_ns`` before something depends on it.
  Triggered during upstream resolution.

Different unit of work, same backend.  Both can be safely active in
the same session because their callers ensure they don't collide.

**Anti-god-class rule (semi-load-bearing):** unlike its sibling
modules, this one DOES import from ``IPython.display`` — replaying
captured rich outputs is intrinsically an IPython operation.  If the
codebase ever needs a non-IPython restore path, the right move is to
factor *that* out as a non-replay sibling, not to spread the IPython
imports further.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from IPython.display import HTML, display, publish_display_data

from .cache_status import CacheStatus
from .randomness import restore_rng_state

if TYPE_CHECKING:
    from ._protocols import ShellProtocol, TrackingState
    from .statement_file_deps import StatementFileDeps
    from .statement_processor import ProcessResult, StatementCacheMetadata

logger = logging.getLogger(__name__)


def _get_statement_code_and_hash(
    metadata: 'StatementCacheMetadata | None',
) -> tuple[str | None, str | None]:
    """Return stored statement code and hash from cache metadata."""
    if not metadata:
        return None, None
    return metadata.get('code'), metadata.get('source_hash')


class StatementRestorer:
    """Hydrate a statement's outputs from a cached payload.

    Holds aliased references to the tracking-state dicts and the
    sibling :class:`StatementFileDeps` (for file-dep restoration).
    Mutates ``user_ns`` and tracking state directly; replays captured
    display output via IPython.
    """

    def __init__(
        self,
        shell: 'ShellProtocol',
        tracking_state: 'TrackingState',
        file_deps: 'StatementFileDeps',
        compute_hash: Callable[[Any], str] | None = None,
        debug: bool = False,
    ) -> None:
        self.shell = shell
        self._file_deps = file_deps
        self.compute_hash = compute_hash
        self.debug = debug
        self.set_tracking_state(tracking_state)

    def set_tracking_state(self, state: 'TrackingState') -> None:
        """Re-wire individual tracking-state dict references (alias pattern)."""
        self.lineage = state.lineage
        self.variable_hashes = state.variable_hashes
        self.variable_sources = state.variable_sources
        self.current_session_hashes = state.current_session_hashes
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes

    @staticmethod
    def persist_metadata_only(
        backend: Any, cache_key: str, metadata: dict[str, Any],
    ) -> None:
        """Persist only metadata (no data payload) to disk for badge display after restart.

        Walks the backend chain to find backends that support metadata-only
        writes (e.g. FileBackend).  Ensures timing info survives kernel
        restarts even when the actual data was too large / too cheap to cache.
        """
        if hasattr(backend, 'set_metadata_only'):
            backend.set_metadata_only(cache_key, metadata)

    def restore_from_cache(
        self,
        cached_data: Any,
        metadata: 'StatementCacheMetadata | None',
        silent: bool,
        process_start: float,
        render_badge: bool = True,
    ) -> None:
        """Restore a cached statement's outputs into ``user_ns`` and replay display."""
        t_restore = time.time()

        try:
            payload = cached_data
            if isinstance(payload, dict) and 'variables' in payload:
                restored_vars = payload['variables']
                stdout = payload.get('stdout', '')
                stderr = payload.get('stderr', '')
                # Prefer the new key but fall back to legacy 'outputs' so
                # existing on-disk caches still hydrate their rich output.
                rich_outputs = payload.get('rich_outputs', payload.get('outputs', []))
                rng_state = payload.get('rng_state')
                if rng_state:
                    if self.debug:
                        logger.debug("[CACHE DEBUG] Restoring RNG state")
                    restore_rng_state(rng_state)
            else:
                restored_vars = payload
                stdout = stderr = ""
                rich_outputs = []

            t_var = time.time()
            for var_name, value in restored_vars.items():
                self._restore_one_var(var_name, value, metadata)

            self._file_deps.restore_from_metadata(restored_vars, metadata)
            var_restore_time = time.time() - t_var

            output_replay_time = 0.0
            if not silent:
                output_replay_time = self._replay_cached_outputs(stdout, stderr, rich_outputs, metadata, render_badge)

            restore_time = time.time() - t_restore
            total_time = time.time() - process_start

            if self.debug:
                logger.debug("[TIMING] Var restore: %.1fms | Output: %.1fms", var_restore_time*1000, output_replay_time*1000)
                logger.debug("[TIMING] Total restore: %.1fms | OVERALL: %.1fms", restore_time*1000, total_time*1000)
                logger.debug("[CACHE DEBUG] ✓ Restored from cache")

        except (KeyError, TypeError, ValueError, AttributeError, OSError) as e:
            if self.debug:
                logger.debug("[CACHE DEBUG] Error restoring cache: %s", e)
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _restore_one_var(
        self,
        var_name: str,
        value: Any,
        metadata: 'StatementCacheMetadata | None',
    ) -> None:
        """Write one restored variable into the shell namespace and update tracking state."""
        self.shell.user_ns[var_name] = value

        if metadata:
            output_lineages = metadata.get('output_lineages', {})
            if var_name in output_lineages:
                self.lineage.record(var_name, output_lineages[var_name], value=value)

            stored_code, stored_hash = _get_statement_code_and_hash(metadata)
            if stored_hash:
                if var_name not in self.executed_cell_hashes:
                    self.executed_cell_hashes[var_name] = set()
                elif isinstance(self.executed_cell_hashes[var_name], str):
                    self.executed_cell_hashes[var_name] = {self.executed_cell_hashes[var_name]}
                self.executed_cell_hashes[var_name].add(stored_hash)
            if stored_code:
                self.executed_cell_codes[var_name] = stored_code

        self._record_restored_var_hash(var_name, value, metadata)

        if metadata and 'key' in metadata:
            self.variable_sources[var_name] = metadata['key']

    def _record_restored_var_hash(
        self,
        var_name: str,
        value: Any,
        metadata: 'StatementCacheMetadata | None',
    ) -> None:
        """Update variable_hashes / current_session_hashes for a single restored variable."""
        type_name = type(value).__name__
        if type_name in ('DataFrame', 'Series', 'ndarray'):
            lineage_hash = (metadata.get('output_lineages', {}) if metadata else {}).get(var_name)
            if lineage_hash:
                self.variable_hashes.setdefault(var_name, set()).add(lineage_hash)
                self.current_session_hashes[var_name] = lineage_hash
        elif self.compute_hash:
            try:
                content_hash = self.compute_hash(value)
                self.variable_hashes.setdefault(var_name, set()).add(content_hash)
                self.current_session_hashes[var_name] = content_hash
            except (TypeError, ValueError, AttributeError, RecursionError) as e:
                if self.debug:
                    logger.debug("[CACHE DEBUG] Could not hash restored variable '%s': %s", var_name, e)

    def _replay_cached_outputs(
        self,
        stdout: str,
        stderr: str,
        rich_outputs: list,
        metadata: 'StatementCacheMetadata | None',
        render_badge: bool,
    ) -> float:
        """Replay stdout/stderr/rich outputs and render the RESTORED badge.

        Returns elapsed seconds (for timing-debug accounting).
        """
        from .badge_renderer import render_status_badge

        t_output = time.time()
        if stdout:
            print(stdout, end='')
        if stderr:
            print(stderr, end='', file=sys.stderr)

        saved_time = metadata.get('execution_time', 0.0) if metadata else 0.0
        source = metadata.get('source') if metadata else None
        storage = metadata.get('storage') if metadata else None
        if render_badge:
            html = render_status_badge(
                CacheStatus.RESTORED,
                time_saved=saved_time, source=source, storage=storage,
            )
            try:
                display(HTML(html))
            except (ImportError, TypeError, ValueError):
                logger.debug("[STATEMENT_RESTORE] Failed to display status badge HTML")

        for output in rich_outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)
        return time.time() - t_output
