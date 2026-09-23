"""Statement-level cache restoration.

Owns the operation "hydrate a statement's outputs from a cached
payload" — write the variables into ``user_ns``, reconstitute lineage
+ source tracking, replay captured stdout / stderr / rich outputs.

Single public entry: :meth:`StatementRestorer.restore_from_cache`,
plus :meth:`StatementRestorer.persist_metadata_only` for the
small-but-cacheable case where only metadata gets persisted.

**Distinct from the variable-granular Restorer:**

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

That import is **function-local, not module-level**.  Base
``cash`` declares ``dependencies = []`` — IPython lives in the
``[notebook]`` extra — but this module sits on the ``import cash``
chain (``core`` → ``notebook`` → ``upstream`` → ``statement``), so a
module-level ``from IPython.display import ...`` made a bare
``pip install cash-lib`` unimportable.  Keep the import inside
:meth:`StatementRestorer._replay_cached_outputs`.

**Do not "fix" this into a module-level try/except with no-op stubs.**
That shape keeps the module importable but makes a real display call
silently render nothing.  ``processor.py`` had exactly that and was
brought in line with this module: import locally, let a
genuine display attempt without IPython raise.  Both are pinned by
``tests/test_notebook/test_display_without_ipython.py``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...tracking.randomness import restore_object_rng_states, restore_rng_state
from ..restored_var import apply_restored_var
from .capture import replay_outputs

if TYPE_CHECKING:
    from .._protocols import ShellProtocol, TrackingState
    from ._metadata import StatementCacheMetadata

logger = logging.getLogger(__name__)


class StatementRestorer:
    """Hydrate a statement's outputs from a cached payload.

    Stateless apart from the shell reference and the optional content
    hasher; all :class:`TrackingState` access happens through the
    ``tracking_state`` method parameter, and what a restored value records
    there is :func:`~cash.notebook.restored_var.apply_restored_var`'s.
    Mutates ``user_ns`` and ``tracking_state`` directly; replays captured
    display output via IPython.
    """

    def __init__(
        self,
        shell: "ShellProtocol",
        compute_hash: Callable[[Any], str] | None = None,
        debug: bool = False,
        rng_seed_epochs: dict[str, str] | None = None,
    ) -> None:
        self.shell = shell
        self.compute_hash = compute_hash
        self.debug = debug
        # SHARED with the processor's ledger (same dict object), so a seed
        # statement executed after construction is visible here.
        self._rng_seed_epochs = rng_seed_epochs if rng_seed_epochs is not None else {}

    def _rng_replay_is_current(self, payload: dict[str, Any]) -> bool:
        """Whether this entry's RNG state may still be replayed.

        Replaying a cached statement's post-execution RNG state keeps the random
        stream coherent when a restore stands in for an execution: the next draw
        then continues from where a real run would have left it.

        That is only true WITHIN one seeding regime. Re-seed the RNG and the
        replay becomes actively destructive -- it rewinds the generator to the
        state the COLD run left behind, silently discarding the seed the user
        just set. The following draw then recomputes (its key changed) and still
        produces the old seed's numbers, because it draws from the old seed's
        state. Keying the draw was necessary but not sufficient; this is the
        other half.

        So an entry may replay its RNG state only while the epochs it was
        written under still hold.
        """
        for module, epoch in payload["rng_epochs"].items():
            if self._rng_seed_epochs.get(module, epoch) != epoch:
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] Skipping RNG replay for %s: re-seeded since caching",
                        module,
                    )
                return False
        return True

    @staticmethod
    def persist_metadata_only(
        backend: Any,
        cache_key: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist only metadata (no data payload) to disk for badge display after restart.

        Only a tier that keeps metadata alone (e.g. FileBackend) writes it.
        Ensures timing info survives kernel restarts even when the actual data
        was too large / too cheap to cache.
        """
        backend.set_metadata_only(cache_key, metadata)

    def restore_from_cache(
        self,
        tracking_state: "TrackingState",
        cached_data: Any,
        metadata: "StatementCacheMetadata | None",
        silent: bool,
        process_start: float,
        inplace_restore: "set[str] | frozenset[str] | None" = None,
    ) -> None:
        """Restore a cached statement's outputs into ``user_ns`` and replay display.

        *inplace_restore* names estimator-fit receivers whose fitted
        state must be transferred onto the EXISTING object rather than rebinding
        the name, so every alias sees the fit. Empty/None for every other
        statement, which keeps the plain-rebind behaviour.
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
                if rng_state and self._rng_replay_is_current(payload):
                    if self.debug:
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
                if self.debug:
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

            if self.debug:
                logger.debug(
                    "[TIMING] Var restore: %.1fms | Output: %.1fms", var_restore_time * 1000, output_replay_time * 1000
                )
                logger.debug("[TIMING] Total restore: %.1fms | OVERALL: %.1fms", restore_time * 1000, total_time * 1000)
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
                    if self.debug:
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
