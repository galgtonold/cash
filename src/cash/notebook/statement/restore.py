"""Statement-level cache restoration.

Owns the operation "hydrate a statement's outputs from a cached
payload" — write the variables into ``user_ns``, reconstitute lineage
+ source tracking, replay captured stdout / stderr / rich outputs.

Single public entry: :meth:`StatementRestorer.restore_from_cache`.
It restores a statement's *outputs as a unit*: multiple vars + replayed
display data + RNG state, on a cache *hit* for a freshly-running
statement.

Replayed output goes through :func:`~cash.notebook.statement.capture.replay_outputs`,
which imports ``IPython.display`` only when there is rich output to show.
The decorator does not need IPython and this module is on the ``import cash``
chain, so it must import without IPython; a genuine display attempt without it raises rather than rendering
nothing. Both are pinned by ``tests/test_notebook/session/test_display_without_ipython.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from cash._clock import perf_counter as _perf_counter

from ...tracking.randomness import restore_object_rng_states, restore_rng_state
from ..holder_patches import HolderPatch, apply_patch
from ..lineage_formula import held_lineage
from ..restored_var import apply_held_var, apply_restored_var
from .capture import replay_outputs

if TYPE_CHECKING:
    from .._protocols import ShellProtocol
    from ..tracking_state import TrackingState
    from ._metadata import StatementCacheMetadata

logger = logging.getLogger(__name__)


def rng_replay_is_current(payload: Mapping[str, Any], seed_epochs: Mapping[str, str]) -> bool:
    """Whether a cached statement's RNG state may still be replayed, under the
    seeding regime *seed_epochs* (``StatementRandomness.seed_epochs``).

    Replaying a cached statement's post-execution RNG state keeps the random
    stream coherent when a restore stands in for an execution: the next draw
    then continues from where a real run would have left it.

    That is only true WITHIN one seeding regime. Re-seed the RNG and the
    replay becomes actively destructive -- it rewinds the generator to the
    state the COLD run left behind, silently discarding the seed the user
    just set. The following draw then recomputes (its key changed) and still
    produces the old seed's numbers, because it draws from the old seed's
    state. Keying the draw is necessary but not sufficient; this is the
    other half.

    So an entry may replay its RNG state only while the epochs it was
    written under still hold.
    """
    for module, epoch in payload["rng_epochs"].items():
        if seed_epochs.get(module, epoch) != epoch:
            logger.debug(
                "[CACHE DEBUG] Skipping RNG replay for %s: re-seeded since caching",
                module,
            )
            return False
    return True


class StatementRestorer:
    """Hydrate a statement's outputs from a cached payload.

    Stateless apart from the shell reference and the optional content
    hasher; all :class:`TrackingState` access happens through the
    ``tracking_state`` method parameter, the seeding regime comes with each
    call, and what a restored value records
    there is :func:`~cash.notebook.restored_var.apply_restored_var`'s.
    Mutates ``user_ns`` and ``tracking_state`` directly; replays captured
    display output via IPython.
    """

    def __init__(
        self,
        shell: "ShellProtocol",
        compute_hash: Callable[[Any], str] | None = None,
    ) -> None:
        self.shell = shell
        self.compute_hash = compute_hash

    def restore_from_cache(
        self,
        tracking_state: "TrackingState",
        cached_data: Any,
        metadata: "StatementCacheMetadata | None",
        silent: bool,
        process_start: float,
        *,
        seed_epochs: Mapping[str, str],
    ) -> None:
        """Restore a cached statement's outputs into ``user_ns`` and replay display.

        *seed_epochs* is the seeding regime in force now; the entry's RNG state
        is replayed only while it still holds (:func:`rng_replay_is_current`).
        """
        t_restore = _perf_counter()

        try:
            payload = cached_data
            if isinstance(payload, dict) and "variables" in payload:
                restored_vars = payload["variables"]
                stdout = payload.get("stdout", "")
                stderr = payload.get("stderr", "")
                rich_outputs = payload.get("rich_outputs", [])
                rng_state = payload.get("rng_state")
                if rng_state and rng_replay_is_current(payload, seed_epochs):
                    logger.debug("[CACHE DEBUG] Restoring RNG state")
                    restore_rng_state(rng_state, tracking_state.rng_live_states)
                # Absent on older entries — restore_object_rng_states
                # treats None/{} as a no-op, so old cache entries load unchanged.
                object_rng_states = payload.get("rng_object_states")
            else:
                restored_vars = payload
                stdout = stderr = ""
                rich_outputs = []
                object_rng_states = None

            t_var = _perf_counter()
            holders = (metadata.holders if metadata is not None else None) or {}
            moved = holders if metadata is not None and metadata.holders_moved else {}
            if metadata is not None and metadata.key:
                tracking_state.held_with[metadata.key] = dict(moved)
                tracking_state.restored_file_states[metadata.key] = restored_file_state_digest(
                    metadata.file_dependencies
                )
            # A holder stored by where it holds the outputs' objects: the
            # live one, given the restored objects there. Found before anything
            # is written, so a holder without those places any more fails the
            # restore before it starts.
            patched = {
                name: apply_patch(self.shell.user_ns[name], value, restored_vars)
                for name, value in restored_vars.items()
                if isinstance(value, HolderPatch)
            }
            for var_name, value in restored_vars.items():
                if var_name in holders:
                    value = patched.get(var_name, value)
                    self.shell.user_ns[var_name] = value
                    before = holders[var_name]
                    apply_held_var(
                        tracking_state,
                        var_name,
                        value,
                        held_lineage(before, metadata.key or "") if var_name in moved else before,
                        compute_hash=self.compute_hash,
                    )
                    continue
                self._restore_one_var(tracking_state, var_name, value, metadata)

            # advance object-held generators to the post-state the
            # cached statement left them in — the module-global equivalent of
            # restore_rng_state above.  Runs AFTER the variable loop so that a
            # carrier which is also an OUTPUT of this statement ends on the
            # canonical post-state rather than whatever ordering the dict
            # happened to have.
            if object_rng_states:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[CACHE DEBUG] Restoring object RNG state for %s",
                        ", ".join(sorted(object_rng_states)),
                    )
                restore_object_rng_states(object_rng_states, self.shell.user_ns)

            var_restore_time = _perf_counter() - t_var

            output_replay_time = 0.0
            if not silent:
                output_replay_time = self._replay_cached_outputs(stdout, stderr, rich_outputs)

            restore_time = _perf_counter() - t_restore
            total_time = _perf_counter() - process_start

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[TIMING] Var restore: %.1fms | Output: %.1fms", var_restore_time * 1000, output_replay_time * 1000
                )
                logger.debug("[TIMING] Total restore: %.1fms | OVERALL: %.1fms", restore_time * 1000, total_time * 1000)
                logger.debug("[CACHE DEBUG] ✓ Restored from cache")

        except (KeyError, TypeError, ValueError, AttributeError, OSError) as e:
            logger.debug("[CACHE DEBUG] Error restoring cache: %s", e)
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _restore_one_var(
        self,
        tracking_state: "TrackingState",
        var_name: str,
        value: Any,
        metadata: "StatementCacheMetadata | None",
    ) -> None:
        """Write one restored variable into the shell namespace and record it.

        A fitted estimator is rebound like any other output: the variables
        holding it, or an object inside it, were stored with it as holders
        and are rebound to the same restored objects above.
        """
        self.shell.user_ns[var_name] = value
        apply_restored_var(tracking_state, var_name, value, metadata, compute_hash=self.compute_hash)

    def _replay_cached_outputs(
        self,
        stdout: str,
        stderr: str,
        rich_outputs: list,
    ) -> float:
        """Replay stdout/stderr/rich outputs.

        Returns elapsed seconds (for timing-debug accounting).
        """
        t_output = _perf_counter()
        replay_outputs(stdout, stderr, rich_outputs)
        return _perf_counter() - t_output


def restored_file_state_digest(file_dependencies: Mapping[str, Any] | None) -> str:
    """A digest of the file state an entry was stored with (its
    ``file_dependencies``: each file, folder, probe and remote object with
    the fingerprint it had), or "" when it has none. Equal for one state,
    different for another: the entry was only restored because each file
    still matches its fingerprint."""
    if not file_dependencies:
        return ""
    text = json.dumps(file_dependencies, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()
