"""Statement-level cache restoration.

Owns the operation "hydrate a statement's outputs from a cached
payload" — write the variables into ``user_ns``, reconstitute lineage
+ source tracking, replay captured stdout / stderr / rich outputs.

Single public entry: :meth:`StatementRestorer.restore_from_cache`.

**Distinct from the variable-granular Restorer:**

* :class:`StatementRestorer` (this module) — restores a statement's
  *outputs as a unit*: multiple vars + replayed display data + RNG
  state.  Triggered on a cache *hit* for a freshly-running statement.
* :class:`Restorer` (``restore.py``) — restores **one variable** from
  cache so it's present in ``user_ns`` before something depends on it.
  Triggered during upstream resolution.

Different unit of work, same backend.  Both can be safely active in
the same session because their callers ensure they don't collide.

Replayed output goes through :func:`~cash.notebook.statement.capture.replay_outputs`,
which imports ``IPython.display`` only when there is rich output to show.
Base ``cash`` has no dependencies (IPython lives in the ``[notebook]`` extra)
and this module is on the ``import cash`` chain, so it must import without
IPython; a genuine display attempt without it raises rather than rendering
nothing. Both are pinned by ``tests/test_notebook/test_display_without_ipython.py``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...tracking.randomness import restore_object_rng_states, restore_rng_state
from ..restored_var import apply_restored_var
from .capture import replay_outputs

if TYPE_CHECKING:
    from .._protocols import ShellProtocol, TrackingState
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
        inplace_restore: "set[str] | frozenset[str] | None" = None,
        *,
        seed_epochs: Mapping[str, str],
    ) -> None:
        """Restore a cached statement's outputs into ``user_ns`` and replay display.

        *inplace_restore* names estimator-fit receivers whose fitted
        state must be transferred onto the EXISTING object rather than rebinding
        the name, so every alias sees the fit. Empty/None for every other
        statement, which keeps the plain-rebind behaviour.

        *seed_epochs* is the seeding regime in force now; the entry's RNG state
        is replayed only while it still holds (:func:`rng_replay_is_current`).
        """
        t_restore = time.time()

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
                    restore_rng_state(rng_state)
                # Absent on older entries — restore_object_rng_states
                # treats None/{} as a no-op, so old cache entries load unchanged.
                object_rng_states = payload.get("rng_object_states")
            else:
                restored_vars = payload
                stdout = stderr = ""
                rich_outputs = []
                object_rng_states = None

            t_var = time.time()
            inplace = inplace_restore or frozenset()
            for var_name, value in restored_vars.items():
                self._restore_one_var(tracking_state, var_name, value, metadata, inplace)

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

            var_restore_time = time.time() - t_var

            output_replay_time = 0.0
            if not silent:
                output_replay_time = self._replay_cached_outputs(stdout, stderr, rich_outputs)

            restore_time = time.time() - t_restore
            total_time = time.time() - process_start

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
        inplace_restore: "set[str] | frozenset[str]" = frozenset(),
    ) -> None:
        """Write one restored variable into the shell namespace and record it.

        For a var in *inplace_restore* (a bare ``estimator.fit(...)``
        receiver) the fitted state is transferred ONTO the existing object
        rather than rebinding the name, so every alias of the receiver
        (``backup = clf``) observes the fit -- mirroring what an in-place
        ``.fit()`` does at runtime.
        """
        self._write_restored_value(var_name, value, inplace_restore)
        apply_restored_var(tracking_state, var_name, value, metadata, compute_hash=self.compute_hash)

    def _write_restored_value(
        self,
        var_name: str,
        value: Any,
        inplace_restore: "set[str] | frozenset[str]",
    ) -> None:
        """Land a restored *value* into ``user_ns`` -- in place for an
        estimator-fit receiver, else a plain rebind.

        An in-place transfer mutates the EXISTING receiver object so every alias
        (``backup = clf``) observes the restored (fitted) state, matching what
        the original in-place ``.fit()`` did. Falls back to a rebind when the
        name is absent (no alias can exist, so a rebind is safe), the existing
        object and the cached value are not the SAME class, or the transfer
        raises for any reason -- a restore must never crash.
        """
        if var_name in inplace_restore and var_name in self.shell.user_ns:
            existing = self.shell.user_ns[var_name]
            if type(existing) is type(value):
                try:
                    self._transfer_state_in_place(existing, value)
                    return
                except Exception as e:  # noqa: BLE001 -- never crash a restore
                    logger.debug(
                        "[CACHE DEBUG] In-place restore of '%s' failed (%s); rebinding",
                        var_name,
                        e,
                    )
        self.shell.user_ns[var_name] = value

    @staticmethod
    def _transfer_state_in_place(existing: Any, value: Any) -> None:
        """Copy *value*'s state onto *existing* in place.

        Prefers the pickle protocol (``__setstate__`` fed from ``__getstate__``)
        so an object with a custom state contract -- sklearn estimators define
        both -- is transferred exactly as it would be unpickled. Falls back to a
        ``__dict__`` swap for a plain object (whose ``object.__setstate__`` is
        absent). Raises on failure; the caller catches and rebinds.
        """
        getstate = getattr(value, "__getstate__", None)
        setstate = getattr(existing, "__setstate__", None)
        if callable(getstate) and callable(setstate):
            state = getstate()
            if state is not None:
                setstate(state)
                return
        existing.__dict__.clear()
        existing.__dict__.update(value.__dict__)

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
        replay_outputs(stdout, stderr, rich_outputs)
        return time.time() - t_output
