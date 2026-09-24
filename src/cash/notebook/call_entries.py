"""The backend entries of intercepted calls: looking one up, deciding a
result may be written, and writing it.

The same ``backend.get``/``backend.set`` the statement path uses
(:mod:`cash.backends._base`), not a separate store. A key found here is a hit
only while the files it read and its statement's TTL say it is fresh.
"""

from __future__ import annotations

import logging
import time as _time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from cash.analysis.cacheability_decision import identity_coupled_reason
from cash.backends._base import ttl_expired
from cash.backends.value_policy import worth_its_bytes
from cash.notebook._trace import trace_event
from cash.notebook.call_refs import (
    DIGEST_FIELD,
    ESTIMATED_FIELD,
    SIZE_FIELD,
    UNHASHED_PREFIX,
    digest_and_size,
)
from cash.notebook.consumables import is_consumable_unrestorable
from cash.object_hashing import estimate_object_size, pickled_size_estimate
from cash.tracking.file_dep_snapshot import attach_code_relative, snapshot_dependencies, snapshot_is_fresh

from ..cost_model import estimated_restore_time

logger = logging.getLogger(__name__)

__all__ = ["CallEntries"]

#: A call cheaper than this gets no content digest, so no statement refers to it.
_REF_MIN_COMPUTE_S = 0.1

#: Result types whose identity no program can rely on -- see
#: :meth:`CallEntries.storable`. Exact types only: a subclass may carry state.
_IDENTITY_FREE = frozenset({int, float, complex, bool, str, bytes, type(None)})


class CallEntries:
    """Reads and writes one notebook session's call entries.

    **Never why user code breaks.** A failed read is a miss and a failed write
    is an uncached call.
    """

    def __init__(
        self,
        cash_instance,
        ttl_provider: Callable[[], int | None] | None = None,
        persist_provider: Callable[[], bool] | None = None,
    ):
        self._cash = cash_instance
        # The TTL in force for the statement this call sits in, read at lookup
        # time because one `CallCache` serves every statement and each brings
        # its own annotation. `None` means "no TTL".
        self._ttl_provider = ttl_provider or (lambda: None)
        # `# @cash:persist` / `%cash_persist` for the statement this call sits
        # in, read at store time for the same reason. `False` leaves the
        # decision to the cost model.
        self._persist_provider = persist_provider or (lambda: False)
        #: Cache keys of sites known to mutate an argument or consume RNG,
        #: discovered by observing a MISS, or whose hit cost more than it
        #: saved. Permanent for the life of this object (one notebook
        #: session): once a site is known to have an effect call caching
        #: cannot replay, it must never be served or written again, on any
        #: later call -- including one wrapped by a fresh ``CallUnit.wrap()``
        #: for the same site (``CallCache.resolve`` re-wraps per statement
        #: execution).
        self._refused: set[str] = set()

    def refuse(self, key: str) -> None:
        """Never look *key* up or write it again this session."""
        self._refused.add(key)

    def is_refused(self, key: str) -> bool:
        return key in self._refused

    #: A hit is judged a loss only past this, so timer noise on a cheap call
    #: never refuses it.
    _MIN_HIT_LOSS_S = 0.05

    def restore_pays(self, result, elapsed: float) -> bool:
        """Whether restoring *result* is predicted to beat computing it again.

        The statement path has refused a value whose predicted restore exceeds
        80% of its compute since the cost model was fitted; the call cache,
        which holds the biggest values in a notebook,
        never asked. Predicted for disk -- where it
        comes back from after a restart, the case a cache is for.
        """
        try:
            size = estimate_object_size(result)
            predicted = estimated_restore_time(type(result).__name__, size, "disk")
        except Exception:  # noqa: BLE001 - no prediction: store, as before
            return True
        return predicted <= max(self._MIN_HIT_LOSS_S, 0.8 * elapsed)

    def drop_if_hit_costs_more(self, key: str, hit_cost: float, saved: float) -> None:
        """A hit that took longer than the compute it saved is a loss: stop.

        Measured, not predicted -- a prediction can be wrong for a type it was
        not fitted on, and one sweep's hits were ~10 s against ~4 s of compute,
        reported as "4/4 hit". The entry is dropped and the site runs plain
        for the rest of the session (:meth:`refuse`, the same bench the argument-
        mutation and RNG refusals use), so the next run computes rather than
        paying the loss again.
        """
        if hit_cost <= max(self._MIN_HIT_LOSS_S, saved or 0.0):
            return
        self._refused.add(key)
        try:
            self._cash.backend.delete(key)
        except Exception:  # noqa: BLE001 - reclaiming is best effort; the refusal holds
            pass
        logger.debug(
            "[CALL_UNIT] hit on %s took %.2fs to save %.2fs: dropped, runs plain", key[:16], hit_cost, saved or 0.0
        )

    def storable(self, result, args, kwargs) -> bool:
        """Refuse values whose *identity* is load-bearing.

        Three families, all of which the statement path already refuses in its
        own vocabulary:

        1. **The result IS one of the arguments.** ``def f(d): d['k']=1;
           return d`` -- a hit would hand back a deserialised copy, so
           ``a = f(d)`` gives ``a is not d`` where Python guarantees identity.
           The statement path's alias rule only reaches a bare bind
           (``b = a``); the computed-RHS version is
           structurally unfixable per-statement. At the call node the live
           arguments are in hand, so it is one ``is`` check.

           Not for a plain scalar. CPython shares one object for small ints
           and interned strings, so ``score(1, 10)`` returns the very ``10``
           it was passed -- the check refused that call on every run, and it
           was always the first iteration of a sweep that re-ran. No program
           can rely on the identity of an int or a str.

        2. **Identity-coupled library objects** -- a matplotlib Figure/Axes is
           only correct while it IS the object pyplot's registry points at.
           The RAM tier deep-copies on store and ``Figure.__setstate__``
           re-registers the COPY as the current figure, so a later bare
           ``plt.savefig()`` writes the cache's snapshot. Refusing here lands
           BEFORE the write, which is what stops the copy being made at all.

        3. **A consumable the store cannot copy** -- an open file, a
           generator. The RAM tier keeps it by reference, so a hit hands back
           the very object a reader already drained. The statement path
           refuses such an output for the same reason.

        A caching optimisation must never be why user code fails. The two
        ``is`` loops above cannot themselves raise -- identity comparison
        never does -- so the only place this can fail is the
        ``identity_coupled_reason`` call, guarded below. Refusing to store is
        free (the call just runs uncached next time); wrongly storing is not
        (it is exactly the silent-wrong-answer / hijacked-identity bug this
        method exists to prevent), which argues for failing toward ``False``.
        But ``identity_coupled_reason`` is pure MRO-qualname introspection --
        by design it never imports matplotlib and has no I/O -- so this
        except is a belt no realistic value should ever reach; returning
        ``True`` here mirrors the already-shipped fallback in
        ``call_unit._is_storable`` (same delegation, same except
        clause) so a call's storability does not silently depend on which of
        the two dispatch paths happened to route it.
        """
        if type(result) not in _IDENTITY_FREE:
            for arg in args:
                if result is arg:
                    return False
            for arg in kwargs.values():
                if result is arg:
                    return False
        try:
            return identity_coupled_reason("<intercepted call>", result) is None and not is_consumable_unrestorable(
                result
            )
        except Exception:  # noqa: BLE001 - never let the predicate break the call
            return True

    def lookup(self, key: str) -> tuple[bool, Any, float, dict]:
        """``(hit, value, recorded_execution_time, metadata)`` -- one backend read.

        ``backend.get`` returns ``(metadata, value)`` (``cash.backends._base``);
        ``metadata is None`` is the key-presence test the statement path itself
        uses (``CacheFreshnessChecker.check_cache``), since a stored ``None``
        value is still a legitimate hit. *metadata* is returned too (rather
        than just the cost pulled out of it) so the caller can replay the
        file/remote/stdout/stderr channels this entry recorded -- see
        ``call_effects.replay_deps`` / ``replay_output``. Backends round-trip
        metadata as an opaque plain ``dict`` (see ``CacheMetadata``'s
        docstring in ``backends/_base.py``); a non-mapping value is treated
        defensively as empty rather than trusted.

        **A key match alone is not enough to call this a hit.** A call's
        cache KEY carries source + argument/loop-var lineage -- never file
        content -- so a stored entry whose recorded file read has since
        changed on disk would otherwise be served forever, regardless of
        this task's dependency-propagation fix: propagating a dependency the
        call itself never re-checks would just make the STATEMENT re-declare
        a staleness nobody underneath it ever notices. ``_auto_file_deps_fresh``
        re-validates it through ``snapshot_is_fresh``, the check
        ``FileDeps.auto_file_deps_fresh`` makes -- a stale entry is treated as a miss like any other,
        so it falls through to a genuine recompute (and gets overwritten
        under the same key) rather than being replayed.
        """
        try:
            metadata, value = self._cash.backend.get(key)
        except Exception:  # noqa: BLE001
            return False, None, 0.0, {}
        if metadata is None:
            return False, None, 0.0, {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        if not self._auto_file_deps_fresh(metadata):
            return False, None, 0.0, {}
        if not self._ttl_fresh(metadata):
            return False, None, 0.0, {}
        try:
            cost = float(metadata.get("execution_time", 0.0))
        except (TypeError, ValueError, AttributeError):
            cost = 0.0
        return True, value, cost, metadata

    def _ttl_fresh(self, metadata: Mapping[str, Any]) -> bool:
        """The statement's TTL applied to a call entry, by the rule
        every cache path shares (:func:`cash.backends._base.ttl_expired`).

        Before this, ``call_unit.py`` contained no reference to ``ttl`` at all,
        so call entries never expired. Once call interception became the
        default that quietly hollowed out the annotation: the
        STATEMENT would expire and re-execute while the expensive call inside
        it was still served from an entry with no expiry. Measured on
        ``# @cash:ttl=0`` -- the spelling the docs give for data that must
        never be served stale -- the work did not re-run at all until
        ``# @cash:no-cache-calls`` was added as well.
        """
        try:
            timestamp = float(metadata.get("timestamp") or 0)
        except (TypeError, ValueError):
            timestamp = 0.0
        return not ttl_expired(timestamp, self._ttl_provider())

    @staticmethod
    def _auto_file_deps_fresh(metadata: Mapping[str, Any]) -> bool:
        """Is every dependency the call recorded still as it was?

        The same snapshot shape and the same :func:`snapshot_is_fresh` the
        decorator uses, so the two cannot drift on what "fresh" means -- or on
        where a file beside the callee's own code is looked for. Absent/empty
        ``auto_file_deps`` (a call that read no files) is vacuously fresh.
        """
        try:
            fresh, _stale = snapshot_is_fresh(metadata.get("auto_file_deps"))
        except Exception:  # noqa: BLE001 - fail closed: cannot prove fresh
            return False
        return fresh

    def store(
        self,
        key: str,
        value,
        elapsed: float,
        *,
        file_deps: frozenset[str] = frozenset(),
        remote_deps: frozenset[str] = frozenset(),
        stdout: str = "",
        stderr: str = "",
        callee_globals: Mapping[str, Any] | None = None,
        function: str | None = None,
        plain_value: bool = False,
        code_module: str | None = None,
    ) -> tuple[str, Any] | None:
        """Write through ``backend.set(key, value, metadata)`` -- the same
        two-positional-argument shape the statement path uses
        (``StatementStore``), not a merged single-dict entry.

        ``file_deps``/``remote_deps`` are snapshotted (mtime/size/hash, or a
        remote validator token) into ONE ``auto_file_deps`` dict -- the exact
        field name and shape ``Cash``'s own decorator writes
        (``_snapshot_tracked_deps`` in ``core.py``) -- rather than two bare
        path lists. A bare list has nothing for :meth:`_auto_file_deps_fresh`
        to compare against; the snapshot is what makes this call's OWN hit
        path able to notice the file it read has since changed, not just
        propagate a dependency nobody re-checks.
        ``stdout``/``stderr`` are omitted when empty, same as
        ``auto_file_deps`` -- an ordinary cached call (no file reads, no
        output) keeps writing the same sparse two-key entry as before.
        Existing consumers that iterate backend metadata (``%cash_stats``,
        the explorer, eviction) already tolerate that sparse shape, and gain
        no new required field when these stay absent.

        Returns the ``(digest, size)`` a statement may refer to this entry by,
        for the caller to hold the result under, or ``None``.
        """
        # `referenced`: the statement holding this result decides whether it
        # is worth its disk, and says so (`PersistencePolicy.decide`).
        metadata: dict[str, Any] = {"execution_time": elapsed, "timestamp": _time.time(), "referenced": True}
        held: tuple[str, Any] | None = None
        if function:
            # What `cash inspect` names the entry by: a call key is `call:<sha>`,
            # so every intercepted call used to be listed as "call".
            metadata["function"] = function
        # `TieredBackend` reads exactly this key to bypass the ~0.1s
        # persistence floor, so threading the statement's resolved annotation
        # here is the whole fix -- the statement path writes the same field
        # from the same `force_persist` (`StatementStore.save`).
        #
        # Written only when True, keeping the sparse-entry shape every other
        # optional channel here follows, and it is a plain `bool`: metadata is
        # eagerly unpickled for EVERY entry at startup, so nothing but builtins
        # belongs in it (see the `callee_globals` note below).
        if self._persist_provider():
            metadata["force_persist"] = True
        if file_deps or remote_deps:
            try:
                # A file beside the callee's own code is checked in each
                # install's own copy, as the decorator records it.
                snap = attach_code_relative(snapshot_dependencies(file_deps, remote_deps), code_module)
            except Exception:  # noqa: BLE001 - never let dep snapshotting break the store
                snap = None
            if snap:
                metadata["auto_file_deps"] = snap
        # A statement holding this result stores a reference to this entry
        # (``call_refs``). Only for a call worth persisting -- hashing every
        # byte of a cheap call's result would cost more than the copy saves --
        # and not for one carrying captured globals, whose value is wrapped.
        #
        # Nor pickled whole when its size already refuses it: pickling
        # a 1.7 GiB result to learn that took 2.7 s after 2.8 s of
        # compute. It gets a one-off token for a digest and its estimated size
        # (`ESTIMATED_FIELD`): judged for disk on its own, and referred to only
        # by the statement it is the plain result of.
        #
        # Nor for the call a statement is nothing but (``a, b = build()``,
        # `plain_value`): that statement's reference is trusted without a
        # digest, so a token serves -- a 402 MiB result was worth
        # keeping, and pickling it for a digest took 2.6 s of 3.7.
        if elapsed >= _REF_MIN_COMPUTE_S and not callee_globals:
            estimate = self._too_big_to_digest(value, elapsed, plain_value)
            found = digest_and_size(value) if estimate is None else (UNHASHED_PREFIX + uuid.uuid4().hex, estimate)
            if found:
                metadata[DIGEST_FIELD], metadata[SIZE_FIELD] = found
                if estimate is not None:
                    metadata[ESTIMATED_FIELD] = True
                held = found
        if stdout:
            metadata["stdout"] = stdout
        if stderr:
            metadata["stderr"] = stderr
        # Omitted when empty, like every other optional channel here,
        # so an ordinary cached call keeps writing the same sparse entry.
        #
        # The payload rides on the VALUE; metadata gets only a plain bool.
        # Metadata is unpickled for EVERY entry in the directory by anything
        # that surveys the cache -- `list_entries`, which `%cash_on` and the
        # CLI both call -- so a user object there is deserialised whether or
        # not it is ever used. Eviction used to do this too, on the write
        # worker, and no longer does: it ranks from a directory walk. A
        # survey of 5859 real metadata files found 31 of 33 fields are plain
        # builtins; the two that were not are a cash-owned serializer class and
        # a `numpy.int64` that leaked in as `size` and made those files
        # unreadable in any environment without numpy. This field must not be
        # the third.
        #
        # An earlier version put it in metadata to avoid changing the value
        # shape under an unchanged key. That objection is void: the `g:` key
        # component appears exactly when there are globals to capture, so an
        # entry carrying this payload has a key no earlier version could mint.
        # There is no old entry to collide with.
        if callee_globals:
            metadata["has_callee_globals"] = True
            value = (value, dict(callee_globals))
        try:
            self._cash.backend.set(key, value, metadata)
        except Exception:  # noqa: BLE001
            logger.debug("call unit: store failed for %s", key)
        return held

    @staticmethod
    def _too_big_to_digest(value, elapsed: float, plain_value: bool = False) -> int | None:
        """The estimated pickled size of *value* when it is not to be
        digested, else ``None``: when even half of it is more than its compute
        is worth on disk (half: a digest skipped for a value worth keeping
        costs other statements their reference), or when it is its
        statement's plain value, whose reference needs no digest."""

        estimate = pickled_size_estimate(value)
        if not estimate:
            return None  # nothing to estimate from: digest as before
        if not plain_value and worth_its_bytes(estimate // 2, elapsed):
            return None
        trace_event("call_digest_skipped", bytes_estimated=estimate, seconds=round(elapsed, 3))
        return estimate
