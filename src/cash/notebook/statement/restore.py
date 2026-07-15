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

**Anti-god-class rule (semi-load-bearing):** this module uses
``IPython.display`` — replaying captured rich outputs is intrinsically
an IPython operation.  Only ``processor.py`` does the same (its own
replay + last-expression repr); keep it to those two.  If the codebase
ever needs a non-IPython restore path, the right move is to factor
*that* out as a non-replay sibling, not to spread the IPython imports
further.

That import is **function-local, not module-level** (CAS-129).  Base
``cash`` declares ``dependencies = []`` — IPython lives in the
``[notebook]`` extra — but this module sits on the ``import cash``
chain (``core`` → ``notebook`` → ``upstream`` → ``statement``), so a
module-level ``from IPython.display import ...`` made a bare
``pip install cash-lib`` unimportable.  Keep the import inside
:meth:`StatementRestorer._replay_cached_outputs`.

**Do not "fix" this into a module-level try/except with no-op stubs.**
That shape keeps the module importable but makes a real display call
silently render nothing.  ``processor.py`` had exactly that and was
brought in line with this module (CAS-132): import locally, let a
genuine display attempt without IPython raise.  Both are pinned by
``tests/test_notebook/test_display_without_ipython.py``.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..randomness import restore_object_rng_states, restore_rng_state

if TYPE_CHECKING:
    from .._protocols import ShellProtocol, TrackingState
    from ._metadata import StatementCacheMetadata
    from .file_deps import StatementFileDeps

logger = logging.getLogger(__name__)


def _get_statement_code_and_hash(
    metadata: 'StatementCacheMetadata | None',
) -> tuple[str | None, str | None]:
    """Return stored statement code and hash from cache metadata."""
    if not metadata:
        return None, None
    return metadata.code, metadata.source_hash


class StatementRestorer:
    """Hydrate a statement's outputs from a cached payload.

    Stateless apart from the shell reference and the optional content
    hasher; all :class:`TrackingState` access happens through the
    ``tracking_state`` method parameter.  Holds the sibling
    :class:`StatementFileDeps` for file-dep restoration.  Mutates
    ``user_ns`` and ``tracking_state`` directly; replays captured display
    output via IPython.
    """

    def __init__(
        self,
        shell: 'ShellProtocol',
        file_deps: 'StatementFileDeps',
        compute_hash: Callable[[Any], str] | None = None,
        debug: bool = False,
    ) -> None:
        self.shell = shell
        self._file_deps = file_deps
        self.compute_hash = compute_hash
        self.debug = debug

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
        tracking_state: 'TrackingState',
        cached_data: Any,
        metadata: 'StatementCacheMetadata | None',
        silent: bool,
        process_start: float,
    ) -> None:
        """Restore a cached statement's outputs into ``user_ns`` and replay display."""
        t_restore = time.time()

        try:
            payload = cached_data
            if isinstance(payload, dict) and 'variables' in payload:
                restored_vars = payload['variables']
                stdout = payload.get('stdout', '')
                stderr = payload.get('stderr', '')
                rich_outputs = payload.get('rich_outputs', [])
                rng_state = payload.get('rng_state')
                if rng_state:
                    if self.debug:
                        logger.debug("[CACHE DEBUG] Restoring RNG state")
                    restore_rng_state(rng_state)
                # Absent on entries written before CAS-90 — restore_object_rng_states
                # treats None/{} as a no-op, so old cache entries load unchanged.
                object_rng_states = payload.get('rng_object_states')
            else:
                restored_vars = payload
                stdout = stderr = ""
                rich_outputs = []
                object_rng_states = None

            t_var = time.time()
            for var_name, value in restored_vars.items():
                self._restore_one_var(tracking_state, var_name, value, metadata)

            # CAS-90: advance object-held generators to the post-state the
            # cached statement left them in — the module-global equivalent of
            # restore_rng_state above.  Runs AFTER the variable loop so that a
            # carrier which is also an OUTPUT of this statement ends on the
            # canonical post-state rather than whatever ordering the dict
            # happened to have.
            if object_rng_states:
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] Restoring object RNG state for %s",
                        ', '.join(sorted(object_rng_states)),
                    )
                restore_object_rng_states(object_rng_states, self.shell.user_ns)

            self._file_deps.restore_from_metadata(tracking_state, restored_vars, metadata)
            var_restore_time = time.time() - t_var

            output_replay_time = 0.0
            if not silent:
                output_replay_time = self._replay_cached_outputs(stdout, stderr, rich_outputs)

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
        tracking_state: 'TrackingState',
        var_name: str,
        value: Any,
        metadata: 'StatementCacheMetadata | None',
    ) -> None:
        """Write one restored variable into the shell namespace and update tracking state."""
        self.shell.user_ns[var_name] = value

        if metadata:
            output_lineages = metadata.output_lineages or {}
            if var_name in output_lineages:
                tracking_state.lineage.record(var_name, output_lineages[var_name], value=value)

            stored_code, stored_hash = _get_statement_code_and_hash(metadata)
            if stored_hash:
                if var_name not in tracking_state.executed_cell_hashes:
                    tracking_state.executed_cell_hashes[var_name] = set()
                tracking_state.executed_cell_hashes[var_name].add(stored_hash)
            if stored_code:
                tracking_state.executed_cell_codes[var_name] = stored_code

        self._record_restored_var_hash(tracking_state, var_name, value, metadata)

        if metadata and metadata.key is not None:
            tracking_state.variable_sources[var_name] = metadata.key

    def _record_restored_var_hash(
        self,
        tracking_state: 'TrackingState',
        var_name: str,
        value: Any,
        metadata: 'StatementCacheMetadata | None',
    ) -> None:
        """Update variable_hashes / current_session_hashes for a single restored variable."""
        type_name = type(value).__name__
        if type_name in ('DataFrame', 'Series', 'ndarray'):
            lineage_hash = ((metadata.output_lineages or {}) if metadata else {}).get(var_name)
            if lineage_hash:
                tracking_state.variable_hashes.setdefault(var_name, set()).add(lineage_hash)
                tracking_state.current_session_hashes[var_name] = lineage_hash
        elif self.compute_hash:
            try:
                content_hash = self.compute_hash(value)
                tracking_state.variable_hashes.setdefault(var_name, set()).add(content_hash)
                tracking_state.current_session_hashes[var_name] = content_hash
            except (TypeError, ValueError, AttributeError, RecursionError) as e:
                if self.debug:
                    logger.debug("[CACHE DEBUG] Could not hash restored variable '%s': %s", var_name, e)

    def _replay_cached_outputs(
        self,
        stdout: str,
        stderr: str,
        rich_outputs: list,
    ) -> float:
        """Replay stdout/stderr/rich outputs.

        Returns elapsed seconds (for timing-debug accounting).
        """
        t_output = time.time()
        if stdout:
            print(stdout, end='')
        if stderr:
            print(stderr, end='', file=sys.stderr)

        if rich_outputs:
            # Imported lazily so `import cash` works without IPython, which is
            # an optional ([notebook] extra) dependency — see the module
            # docstring (CAS-129).  Deliberately NOT wrapped in a try/except
            # no-op: a notebook user replaying rich output expects it to
            # actually render, so failing loudly beats silently dropping it.
            # Guarded by `if rich_outputs` so the common no-rich-output restore
            # skips the import entirely on this hot path.
            from IPython.display import display, publish_display_data

            for output in rich_outputs:
                if isinstance(output, dict) and 'data' in output:
                    publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
                else:
                    display(output)
        return time.time() - t_output
