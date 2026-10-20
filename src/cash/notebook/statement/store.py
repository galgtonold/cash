"""Writing a statement's result to the cache, and deciding whether to.

:class:`StatementStore` owns the store half of the statement pipeline: it
applies the persistence policy's statement gate (the "too cheap to cache"
floor and the restore-cost budget, `PersistencePolicy.too_cheap_to_store` and
`PersistencePolicy.refuses_value`), the refusals that route a result to a
metadata-only entry, and the write itself -- the payload, the metadata and the
files it was built from.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import pickle
import time
import types
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from cash import cost_model, handed_digests, kept_state, stored_pickle
from cash._clock import perf_counter as _perf_counter
from cash._memo import PRODUCER_SNAPSHOTS, LruMemo
from cash._user_attrs import attribute_or
from cash.backends import memory_backend
from cash.backends.persistence_policy import PersistencePolicy, restore_kind
from cash.backends.store_notices import DECORATED_CALLS_FIELD
from cash.closure_sharing import closure_check_budget, closures_sharing
from cash.diagnostics import log_diagnostic, warn_diagnostic
from cash.exceptions import CashCacheStoreFailedWarning
from cash.notebook.anchors import AnchorUnrebuilt, with_anchor_refs
from cash.notebook.statement._metadata import StatementCacheMetadata
from cash.notebook.statement.capture import replay_record
from cash.notebook.statement.carrier_advances import PAYLOAD_FIELD as CARRIERS_FIELD
from cash.notebook.statement.miss_guard import GUARD_SKIP_REASON
from cash.notebook.statement.run import ECHO_FIELD
from cash.sizing import estimate_object_size, pickled_size_estimate
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_dependencies
from cash.tracking.randomness import capture_object_rng_states
from cash.value_hash import known_size
from cash.value_types import IMMUTABLE_LEAF_TYPES

from ..call_key import holds_a_closure_with_state
from ..call_refs import DECORATOR_ENTRY, REF_BYTES_FIELD, REFS_FIELD, CallRef
from ..closed_stream import ClosedStream, stored_form
from ..lineage_store import frozen_tag

if TYPE_CHECKING:
    from cash.notebook._protocols import CashInstanceProtocol, ShellProtocol
    from cash.notebook.statement.amplification import AmplificationGuard
    from cash.notebook.statement.call_routing import CallRouting
    from cash.notebook.statement.lineage import StatementLineageBuilder
    from cash.notebook.statement.rebuild_cost import RebuildCostLedger
    from cash.notebook.statement.run import StatementExecution, StatementRun
    from cash.notebook.tracking_state import TrackingState

logger = logging.getLogger(__name__)

__all__ = ["StatementStore"]


def _snapshot_with_inherited(
    file_dependencies: set[str],
    accessed_remote: set[str],
    inherited_snapshots: dict[str, dict] | None,
    known: dict[str, tuple[Any, ...]] | None = None,
    probes: dict[str, str] | None = None,
    checked: Mapping[str, Any] | None = None,
) -> dict[str, dict]:
    """Snapshot the files a statement read itself; take the ones it only
    inherited from its inputs' producers as they recorded them. *known* are
    the digests taken as the statement read its files
    (`FileAccessTracker.known_reads`), reused while a file's stat is unchanged.
    *probes* are the paths it only checked for (``probe_answers``): the entry
    is fresh while each is still missing, or still there. *checked* are the
    stats the check that no input moved took just before
    (``FileAccessTracker.checked_stats``)."""
    inherited_snapshots = inherited_snapshots or {}
    own = set(file_dependencies) - inherited_snapshots.keys()
    absent = present = None
    if probes:
        absent = [path for path, asked in probes.items() if asked == "absent"]
        present = {path: asked for path, asked in probes.items() if asked != "absent"}
    snapshot = snapshot_dependencies(own, accessed_remote, absent, known=known, present=present, checked=checked)
    for path, recorded in inherited_snapshots.items():
        snapshot.setdefault(path, recorded)
    return snapshot


def _frozen_tags(captured_vars: Mapping[str, Any], outputs: set[str]) -> dict[str, list[str]] | None:
    """``{output: its frozen tag}`` for the outputs holding a ``frozen=True``
    result (``StatementCacheMetadata.frozen_tags``), or None."""
    found = {}
    for name in outputs:
        if name in captured_vars:
            tag = frozen_tag(captured_vars[name])
            if tag is not None:
                found[name] = tag
    return found or None


def _output_sizes(captured_vars: Mapping[str, Any], outputs: set[str]) -> dict[str, int] | None:
    """``{output: the bytes its pickle came to}`` for the outputs whose size
    is known (`value_hash.known_size`, ``StatementCacheMetadata.output_sizes``),
    or None."""
    found = {}
    for name in outputs:
        if name in captured_vars:
            size = known_size(captured_vars[name])
            if size is not None:
                found[name] = size
    return found or None


def _carriers_advanced(run: StatementRun) -> list[str] | None:
    """The generators *run* drew from, for its metadata: the simulation reads
    them after a restart (`carrier_advances`). None when it read none."""
    return None if run.carriers_advanced is None else sorted(run.carriers_advanced)


def _version_slot(source_hash: str, outputs: set[str]) -> str:
    """What makes two stored results versions of one statement: its source and
    the names it binds. Two cells with the same statement share a slot; a
    version one of them reads in this process is never pruned for the other."""
    return hashlib.sha256(f"{source_hash}|{','.join(sorted(outputs))}".encode()).hexdigest()[:32]


def _cost_fields(prediction: dict[str, Any] | None) -> dict[str, Any]:
    """The cost-model prediction as the metadata fields that carry it."""
    if prediction is None:
        return {}
    return {
        "cost_model_size_bytes": prediction["size_bytes"],
        "cost_model_restore_seconds": prediction["restore_seconds"],
        "cost_model_type_name": prediction["type_name"],
        "cost_model_family": prediction["family"],
    }


def _is_only_definitions(code: str) -> bool:
    """Whether *code* is nothing but ``def``/``class`` statements."""
    try:
        body = ast.parse(code).body
    except (SyntaxError, ValueError):
        return False
    return bool(body) and all(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in body)


class _AnchorsUnstorable(Exception):
    """The value of a statement with anchors could not be pickled with them
    (`with_anchor_refs`): it is not stored. *names*: the anchor the copy
    could not be rebuilt around (`AnchorUnrebuilt`); none when the value's
    own pickling raised, as it would without anchors."""

    def __init__(self, message: str, names: Iterable[str]) -> None:
        super().__init__(message)
        self.names = sorted(names)


def _warn_closures_unchecked(outputs: Iterable[str], seconds: float) -> None:
    """Say that a statement's outputs were too big to look through for a
    function whose closure holds another of their parts in *seconds*, so
    they were not stored (STORE-PARTS-UNCHECKED)."""
    listed = ", ".join(f"'{name}'" for name in sorted(outputs))
    warn_diagnostic(
        CashCacheStoreFailedWarning,
        "STORE-PARTS-UNCHECKED",
        f"cash did not store this statement's result: it could not look through {listed} within "
        f"{seconds:.1f}s for a function whose closure holds another part of them, which a restore would "
        "hand back apart. It ran, and runs again next time.",
        "nothing, if the statement is cheap. To have it cached, bind the large data and the objects "
        "holding functions in separate statements.",
    )


def _warn_not_stored(exc: BaseException) -> None:
    """Say that a statement's result was not stored, and why: STORE-FAILED,
    unless no pickle takes the value, which goes to the log only. A
    statement binding a logging handler or a tqdm bar holds a lock or a
    stream no pickle takes, and a warning on each would have no fix. Any
    other failure -- the backend refusing the write (a full disk, a store
    that raised) -- is warned: the statement runs again every time."""
    if isinstance(exc, _AnchorsUnstorable) and exc.names and not _unpicklable(exc):
        # Cash's own step, not the backend: the copy that puts a reference to
        # each held variable in its place cannot rebuild the value around
        # one (``results.append`` is rebuilt as ``getattr(results,
        # 'append')``, and a reference has no ``append``).
        name = exc.names[0]
        warn_diagnostic(
            CashCacheStoreFailedWarning,
            "STORE-REF-UNSUPPORTED",
            f"cash did not store this statement's result: it holds the object of '{name}' in a form cash "
            f"cannot keep as a reference to that variable, such as a method bound to it ({exc}); "
            "it ran, and runs again next time.",
            f"nothing, for a correct result. To have it cached, hold '{name}' itself in the value "
            "rather than a method bound to it.",
        )
        return
    if isinstance(exc, _AnchorsUnstorable) and exc.__cause__ is not None:
        exc = exc.__cause__  # the value's own pickling raised, as it would without anchors
    if isinstance(exc, kept_state.DefinedAgain):
        fix = (
            "run the cells that build the value again after the cell that defines the function, "
            "so it holds the function defined now."
        )
    elif _unpicklable(exc):
        logger.warning("[CACHE] Failed to write to cache backend: %s", exc)
        return
    else:
        fix = "read the exception: a full disk, a cache_dir you cannot write to, or a value whose own copy or pickle code raised."
    warn_diagnostic(
        CashCacheStoreFailedWarning,
        "STORE-FAILED",
        f"cash could not store this statement's result ({type(exc).__name__}: {exc}); "
        "it ran, and runs again next time.",
        fix,
    )


def _unpicklable(exc: BaseException) -> bool:
    """Whether *exc*, or an exception it was raised from, says a value takes
    no pickle (``cannot pickle '_thread.RLock' object``)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, pickle.PicklingError):
            return True
        text = str(current).lower()
        if isinstance(current, (TypeError, AttributeError)) and ("pickle" in text or "local object" in text):
            # ``Can't get local object 'make.<locals>.<lambda>'``: a function
            # defined inside another, which pickle finds by name only.
            return True
        current = current.__cause__ or current.__context__
    return False


class StatementStore:
    """Decides whether a statement's result is worth storing, and stores it.

    Collaborators are injected: the lineage builder writes the entry's
    lineages, the amplification guard and the rebuild-cost ledger can refuse
    or force a write, and the call router swaps served call results for
    references.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        cash_instance: CashInstanceProtocol,
        *,
        lineage_builder: StatementLineageBuilder,
        amplification: AmplificationGuard,
        rebuild_cost: RebuildCostLedger,
        calls: CallRouting,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.cash_instance = cash_instance
        self._lineage_builder = lineage_builder
        self._amplification = amplification
        self._rebuild_cost = rebuild_cost
        self._calls = calls
        #: Names a later top-level statement of the running cell writes
        #: (:meth:`set_written_later_in_cell`).
        self.written_later_in_cell: frozenset[str] = frozenset()
        #: The file snapshots each producing entry recorded, by cache key,
        #: under the ``"__epoch__"`` of the hash scheme they were taken with.
        self._producer_snapshots: LruMemo[str, dict[str, dict]] = LruMemo(PRODUCER_SNAPSHOTS)
        self._producer_snapshots_epoch: Any = None
        #: Statements already warned about a file changed while they ran.
        self._warned_input_changed: set[str] = set()

    def set_written_later_in_cell(self, names: frozenset[str]) -> None:
        """Record the names a later top-level statement of the cell writes.

        An intermediate value of the cell stays in RAM; only the cell's final
        version of a name is persisted, at the cell's end.
        """
        self.written_later_in_cell = names

    #: The perpetual-miss guard spares a statement whose value is written in at
    #: most this share of what computing it cost: each write it wastes is then
    #: nearly free, and one later hit repays all of them.
    CHEAP_WRITE_SHARE = 0.1

    def write_is_cheap(self, outputs: set[str], captured_vars: dict[str, Any], execution_time: float) -> bool:
        """Whether writing *outputs* costs little next to *execution_time*.

        The guard exists for values whose every write is wasted money -- a
        bare fit re-serialising a large model, -25 s a session. Five upstream
        edits in a row also churn a key, and that is an ordinary morning of
        model tuning: a cross-validation, seconds to compute and a few
        numbers to store, stopped being saved and the next restart ran every
        CV again. Estimated with the cost model the size-aware skip uses.
        """

        if execution_time <= 0:
            return False
        try:
            write = 0.0
            for name in outputs:
                if name not in captured_vars:
                    continue
                value = captured_vars[name]
                write += cost_model.estimated_serialize_time(type(value).__name__, estimate_object_size(value), "disk")
        except Exception:  # noqa: BLE001 - an estimate it cannot make guards as before
            return False
        return write <= self.CHEAP_WRITE_SHARE * execution_time

    def save(
        self,
        run: StatementRun,
        execution: StatementExecution,
        captured_vars: dict[str, Any],
        miss_guarded: bool = False,
        *,
        seed_epochs: Mapping[str, str],
    ) -> StatementCacheMetadata | None:
        """Store *run*'s result, or the metadata alone when a gate refuses the
        value; None when the statement gets no entry at all.

        *seed_epochs* is the seeding regime each RNG was last seeded under
        (``StatementRandomness.seed_epochs``), stored with the RNG state.
        """
        if attribute_or(execution.result, "skipped", False):
            return None
        if self.inputs_moved(run, execution):
            return None

        accessed_files = execution.accessed_files
        all_file_deps = set(accessed_files) if accessed_files else set()
        inherited_snapshots: dict[str, dict] = {}

        # CRITICAL: Include inherited file dependencies from input variables
        # This ensures that when Cell 3 (`df`) is cached, it stores the CSV file's mtime
        # even though Cell 3 didn't directly read the file. This allows proper invalidation.
        #
        # An inherited file is recorded as the input's PRODUCER recorded it --
        # the state the value was built from -- rather than read and hashed
        # again here. Re-snapshotted per statement, every statement derived from
        # a frame read out of 5,222 files re-read all 5,222.
        # A later lookup still checks the real file against it.
        direct = all_file_deps.copy()
        for input_var in run.inputs:
            inherited = self.tracking_state.executed_file_deps.get(input_var)
            if not inherited:
                continue
            all_file_deps.update(inherited)
            recorded = self._producer_file_snapshots(input_var, run.entry_sources.get(input_var))
            for path in inherited:
                if path in recorded and path not in direct:
                    inherited_snapshots.setdefault(path, recorded[path])

        return self._store(
            run,
            execution,
            captured_vars,
            file_dependencies=all_file_deps,
            miss_guarded=miss_guarded,
            inherited_snapshots=inherited_snapshots,
            seed_epochs=seed_epochs,
        )

    def inputs_moved(self, run: StatementRun, execution: StatementExecution, reuse_end_stats: bool = False) -> bool:
        """Did a file the statement read change after it read it, or a
        folder it listed gain or lose an entry from outside?

        The entry's file fingerprints are taken when it is stored, after the
        statement ran: a file another process replaced while it ran would be
        recorded as the NEW file next to a value computed from the old one,
        and every later run would find the entry fresh. The value is kept for
        this run, not stored, and the statement warns once
        (``STORE-INPUT-CHANGED``). Asked when the statement has run and again
        before its entry is written; the first ask may take the stats its
        deferred digests were taken under as it ended (*reuse_end_stats*),
        the second stats every file again.
        """
        tracker = execution.file_tracker
        if tracker is None or not (tracker.read_stats or tracker.listed_stats):
            return False
        try:
            moved = tracker.inputs_changed_since_read(reuse_end_stats=reuse_end_stats)
        except Exception:
            # Not cached: a check that could not finish never says the files
            # held still.
            logger.debug("[PROCESSOR] Could not check the files read for a change", exc_info=True)
            return True
        if not moved:
            return False
        if run.source_hash not in self._warned_input_changed:
            self._warned_input_changed.add(run.source_hash)
            first_line = (run.code.splitlines() or [""])[0].strip()
            if len(first_line) > 60:
                first_line = first_line[:57] + "..."
            shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
            what = (
                f"`{first_line}`: {shown} changed while the statement was running, after it had "
                "been read. Its value is kept for this run but not cached, because it cannot be "
                "told which version of the file it was computed from."
            )
            fix = (
                "nothing, if something else writes these files while this runs -- the next run "
                "reads the settled file and caches normally. If the statement reads a file and "
                "then writes to it, that is why: split the read and the write."
            )
            log_diagnostic(logger, "STORE-INPUT-CHANGED", what, fix)
            warn_diagnostic(CashCacheStoreFailedWarning, "STORE-INPUT-CHANGED", what, fix)
        return True

    def _producer_file_snapshots(self, var_name: str, key: str | None = None) -> dict[str, dict]:
        """The file snapshots *var_name*'s producing statement stored, or {}:
        the entry *key* when given (`StatementRun.entry_sources`)."""
        key = key or self.tracking_state.variable_sources.get(var_name)
        backend = attribute_or(self.cash_instance, "backend", None) if self.cash_instance else None
        if not key or backend is None:
            return {}

        memo = self._producer_snapshots
        if self._producer_snapshots_epoch != file_dep_snapshot.HASH_EPOCH:
            memo.clear()
            self._producer_snapshots_epoch = file_dep_snapshot.HASH_EPOCH
        cached = memo.get(key)
        if cached is not None:
            return cached
        try:
            meta = backend.peek_metadata(key)
        except Exception:  # noqa: BLE001 - a snapshot it cannot read is taken afresh
            meta = None
        snaps = (meta or {}).get("file_dependencies") or {} if isinstance(meta, dict) else {}
        memo[key] = snaps
        return snaps

    def policy(self) -> PersistencePolicy:
        """The persistence policy under the current config, read per statement
        so ``cash.configure`` takes effect on the next one."""
        config = attribute_or(self.cash_instance, "config", None)
        return PersistencePolicy.from_config(config) if config is not None else PersistencePolicy()

    def should_skip_large_object_caching(
        self,
        captured_vars: dict[str, Any],
        execution_time: float,
        force_persist: bool = False,
        has_file_dependencies: bool = False,
        exempt: set[str] = frozenset(),
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Decide whether keeping a statement's output values is worthwhile.

        A value is refused when restoring it is predicted to cost more than
        the policy's restore budget (`PersistencePolicy.refuses_value`), with
        the restore time predicted by `cost_model` for where the value is read
        back from: the first tier (`restore_kind`).

        Never refused: under ``force_persist`` (``@cash:persist``), or when the
        statement reads files itself (reading is the expensive part then).
        Nor for a variable in *exempt*, though it is predicted.

        Returns:
            ``(should_skip, reason, prediction)`` where *reason* is a human-readable
            explanation when skipping (else ``None``), and *prediction* is the cost-model
            dict for the largest variable seen (keys: ``size_bytes``, ``restore_seconds``,
            ``type_name``, ``family``), or ``None`` when estimation failed for all vars.
            The prediction is returned on every path, so cost-model validation
            sees the same family attribution whichever gate decided.
        """
        policy = self.policy()
        backend_kind = restore_kind(attribute_or(self.cash_instance, "backend", None))

        largest_prediction: dict[str, Any] | None = None
        # Only the FIRST refused var decides; every var is predicted.
        skip_decision: tuple[str | None, dict[str, Any] | None] | None = None
        for var_name, var_value in captured_vars.items():
            skip, reason, prediction = self._check_var_restore_budget(
                var_name, var_value, execution_time, backend_kind, policy
            )
            if prediction is not None:
                if largest_prediction is None or prediction["size_bytes"] > largest_prediction["size_bytes"]:
                    largest_prediction = prediction
            if skip and skip_decision is None and var_name not in exempt:
                skip_decision = (reason, prediction)

        if force_persist or has_file_dependencies:
            return False, None, largest_prediction

        if skip_decision is not None:
            reason, skip_prediction = skip_decision
            return True, reason, skip_prediction

        return False, None, largest_prediction

    def _check_var_restore_budget(
        self,
        var_name: str,
        var_value: Any,
        execution_time: float,
        backend_kind: str,
        policy: PersistencePolicy,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Return (skip, reason, prediction) for a single variable based on the
        predicted restore cost from the fitted cost model.

        ``prediction`` is a dict with keys ``size_bytes``, ``restore_seconds``,
        ``type_name``, ``family``; or ``None`` if size estimation raises.
        """

        is_ram_backend = backend_kind == "ram"
        try:
            # An object of a class of its own is sized by the frames and
            # arrays it holds, not by its own few bytes: a statsmodels results
            # object holding its design matrix read as 48 bytes, so a 145 MiB
            # copy of it was predicted free, made, and then refused the disk.
            obj_size = max(estimate_object_size(var_value), pickled_size_estimate(var_value))
            type_name = type(var_value).__name__
            family = cost_model.resolve_family(type_name)
            est_restore_time = cost_model.estimated_restore_time(type_name, obj_size, backend_kind)
            # What unpickling one of its class took here (`round_trip_seconds`):
            # a RAM hit copies it that way, a disk hit unpickles it.
            round_trip = memory_backend.round_trip_seconds(var_value, obj_size)
            if is_ram_backend:
                # What a hit costs here, as the RAM tier measures it: a copy
                # at this machine's speed now, or nothing for a table it shares.
                measured = memory_backend.hit_seconds(var_value, obj_size)
                if measured is not None:
                    est_restore_time = measured
            elif round_trip is not None:
                est_restore_time = max(est_restore_time, round_trip)
            prediction: dict[str, Any] = {
                "size_bytes": obj_size,
                "restore_seconds": est_restore_time,
                "type_name": type_name,
                "family": family,
            }
            if family == cost_model.GENERIC_FAMILY:
                prediction["class_key"] = memory_backend.class_key(type(var_value))

            if policy.refuses_value(execution_time, est_restore_time):
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copying" if is_ram_backend else "serializing"
                pct_label = f"{policy.min_savings_pct * 100:.0f}%"
                reason = (
                    f"Restoring '{var_name}' ({size_mb:.0f} MB {type_name}) would take "
                    f"~{est_restore_time:.2f}s vs {execution_time:.2f}s compute "
                    f"({backend_label}, <{pct_label} savings) — "
                    f"use @cash:persist to force"
                )
                logger.debug("[SIZE_AWARE] %s", reason)
                return True, reason, prediction
            if obj_size > 10 * 1024 * 1024:
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copy" if is_ram_backend else "serialize"
                logger.debug(
                    "[SIZE_AWARE] Caching '%s' (%.1fMB %s) — est. %s %.2fs vs %.2fs compute",
                    var_name,
                    size_mb,
                    type_name,
                    backend_label,
                    est_restore_time,
                    execution_time,
                )
            return False, None, prediction
        except (TypeError, ValueError, AttributeError, OSError, RecursionError):
            logger.debug("[SIZE_AWARE] Failed to estimate object size, allowing caching")
        return False, None, None

    @staticmethod
    def _filter_safe_vars(captured_vars: dict[str, Any]) -> dict[str, Any]:
        """Every captured variable but a module, a closed file as its name and
        mode (`closed_stream`). Whether a value pickles is the backend's to
        find out when it stores it."""
        return {k: stored_form(v) for k, v in captured_vars.items() if not isinstance(v, types.ModuleType)}

    def _store(
        self,
        run: StatementRun,
        execution: StatementExecution,
        captured_vars: dict[str, Any],
        *,
        file_dependencies: set[str],
        miss_guarded: bool = False,
        inherited_snapshots: dict[str, dict] | None = None,
        seed_epochs: Mapping[str, str],
    ) -> StatementCacheMetadata | None:
        """Store execution results and metadata in the cache. Returns
        metadata, or ``None`` when the statement was so cheap to compute
        that we don't even write a metadata-only entry (the next lookup
        will miss cleanly rather than hit a metadata-only entry and
        pay a per-file read just to decide 'recompute')."""
        t_store = _perf_counter()
        # What this key recorded is about to change (``_producer_file_snapshots``).
        self._producer_snapshots.pop(run.cache_key, None)

        if self._too_cheap_to_store(run, execution, file_dependencies, execution.cost):
            if run.outputs and not self._calls.in_loop and replay_record(execution.captured) is not None:
                # What it showed is kept for a cell run that skips it.
                return self._store_metadata_only(run, execution, None, {})
            return None
        try:
            payload, referenced = self._payload(run, execution, captured_vars, seed_epochs)
        except _AnchorsUnstorable as exc:
            reason = f"its value holds variables' objects by reference, and it cannot be stored so ({exc})"
            _warn_not_stored(exc)
            return self._store_metadata_only(run, execution, reason, {})
        if not referenced and self._too_cheap_to_store(run, execution, file_dependencies, execution.store_cost):
            # Its own work is not worth an entry: the calls inside it are what
            # was, and the cache holds them. The metadata still records its
            # output lineages, which a restart needs.
            return self._store_metadata_only(run, execution, None, {})
        by_reference = {name for name, value in payload["variables"].items() if isinstance(value, CallRef)}
        should_skip, skip_reason, prediction = self._refusal(
            run, execution, captured_vars, miss_guarded, by_reference=by_reference
        )
        cost_fields = _cost_fields(prediction)
        if should_skip:
            return self._store_metadata_only(run, execution, skip_reason, cost_fields)
        kept_out = self._copy_dearer_than_compute(run, execution, payload["variables"])
        if kept_out is not None:
            return self._store_metadata_only(run, execution, kept_out, cost_fields)

        metadata = StatementCacheMetadata(
            timestamp=time.time(),
            inputs=list(run.inputs),
            outputs=list(run.outputs),
            execution_time=execution.cost,
            # An entry that refers to call entries holds no second copy of
            # their results, and restoring it saves the whole statement: the
            # calls it refers to weigh in (``call_ref_bytes``) with what they cost.
            store_time=execution.cost if referenced else execution.store_cost,
            source_hash=run.source_hash,
            code=run.code,
            key=run.cache_key,
            file_dependencies=_snapshot_with_inherited(
                file_dependencies,
                execution.accessed_remote,
                inherited_snapshots,
                execution.file_tracker.known_reads() if execution.file_tracker is not None else None,
                execution.probes,
                attribute_or(execution.file_tracker, "take_checked_stats", dict)(),
            ),
            force_persist=run.force_persist,
            output_lineages=self._lineage_builder.build_output_lineages(self.tracking_state, run.outputs),
            output_digests=self._lineage_builder.build_output_digests(self.tracking_state, run.outputs) or None,
            output_sizes=_output_sizes(captured_vars, run.outputs),
            input_lineages=self._lineage_builder.build_input_lineages(
                self.tracking_state, run.inputs, self.tracking_state.statement_entry_lineages.get(run.code)
            ),
            frozen_tags=_frozen_tags(captured_vars, run.outputs),
            carriers_advanced=_carriers_advanced(run),
            sparse_ordered=run.sparse_ordered or None,
            env_reads=run.env_reads,
            holders=run.holders or None,
            holders_moved=True if run.holders and run.moves_holders else None,
            anchors=run.anchors or None,
            anchor_links=sorted(run.anchor_links) if run.anchors else None,
            anchor_holds=run.anchor_holds or None,
            share_edges=sorted([src, dst] for src, dst in run.linked_edges) if run.linked_edges is not None else None,
            ttl=run.effective_ttl,
            version_slot=_version_slot(run.source_hash, run.outputs),
            replay=replay_record(execution.captured),
            **cost_fields,
        )
        wire = self._wire(run, metadata, referenced)
        if any(type(v) is ClosedStream for v in payload["variables"].values()):
            # The RAM tier's copy of a stored closed file is the closed file
            # again, which does not pickle: the disk gets this payload.
            wire[memory_backend.NO_PRIVATE_COPY] = True
        self._write(run, payload, wire, prediction)

        store_time = _perf_counter() - t_store
        total_time = _perf_counter() - run.process_start
        logger.debug("[TIMING] Store: %.1fms | OVERALL: %.1fms", store_time * 1000, total_time * 1000)
        logger.debug("[CACHE DEBUG] Stored in cache: %s", run.cache_key)

        stored = StatementCacheMetadata.from_dict(wire)
        self.tracking_state.entry_file_states[run.cache_key] = (stored.file_dependencies,)
        return stored

    @staticmethod
    def _reads_files(execution: StatementExecution, file_dependencies: set[str]) -> bool:
        """Whether *file_dependencies* or the URLs the statement read hold any
        read outside the calls the cache holds: those calls' entries keep
        their own reads, and restore them."""
        cached = execution.cached_call_reads
        return bool(set(file_dependencies) - cached or set(execution.accessed_remote) - cached)

    def _too_cheap_to_store(
        self, run: StatementRun, execution: StatementExecution, file_dependencies: set[str], seconds: float
    ) -> bool:
        """Whether *run*'s value, worth *seconds*, is not worth an entry.

        Asked of the statement's whole cost first: under the floor, it gets no
        entry at all, not even a metadata-only one. Checked apart from the
        refusals below so that nothing is written: a notebook with many
        trivial statements (100 ``a_i = i + 1``) would otherwise write 100
        metadata-only files on its first run, and every later run would pay
        ~1ms a statement reading them only to find skipped entries. Writing
        nothing makes the next lookup a fast clean miss. Then of its own
        work, the calls the cache holds left out (``store_cost``).

        The floor is waived for a statement with any file dependency, its
        inputs' included, except one read inside a call the cache holds. A
        cheap reader of a file HANDLE (``lines = [l for l in fh]``) must still
        be stored: re-run on a second Run All, it reads the handle its
        skipped producer left at EOF and gets [] (test_file_handle_iteration_second_run_all).
        """
        if run.force_persist or self._reads_files(execution, file_dependencies):
            return False
        if _only_shows_a_name(run.tree):
            # ``data`` alone: its cost is the display, which a hit repeats.
            # Stored, the value was copied and pickled again, 0.6 s for a
            # list of 87,000 sessions, then refused as not worth its bytes.
            return True
        # On a contended machine a trivial statement can measure tens of ms
        # and clear the floor, so nothing may assume this branch is taken
        # for a given statement (the floor-exit test pins the threshold
        # rather than trusting the machine to be fast).
        policy = self.policy()
        execution_time = seconds
        if not policy.too_cheap_to_store(execution_time) or self._rebuild_cost.final_over_costly_inputs(
            run.inputs, run.outputs, in_loop=self._calls.in_loop, written_later=self.written_later_in_cell
        ):
            return False
        logger.debug(
            "[SIZE_AWARE] Compute took only %.1fms, below %.0fms floor — not writing cache entry",
            execution_time * 1000,
            policy.store_floor_s * 1000,
        )
        return True

    def _copy_dearer_than_compute(
        self, run: StatementRun, execution: StatementExecution, variables: dict[str, Any]
    ) -> str | None:
        """Why *run*'s value is kept out of the RAM tier, or None to store it.

        Only for an intermediate of the cell -- every output rebound later
        in it (``written_later_in_cell``), so only a re-run of part of the
        cell in this session could restore it -- whose copy into the RAM
        tier costs more than computing it did (`memory_backend.store_copy_seconds`).
        Running it again on such a re-run is then no slower than restoring
        it was, and the copy is paid on every miss: a cleaning cell
        filtering a 2.9-million-row table five times (``df = df[...]``)
        spent 9 s of a 4 s cell copying versions nothing restored. Its
        disk write is deferred to the end of the cell already, and the
        cell's last version is stored as before. Not in a loop, not under
        ``@cash:persist``, and not for a value whose copy the tier cannot
        price."""
        later = self.written_later_in_cell
        if run.force_persist or self._calls.in_loop or not run.outputs or not later or not set(run.outputs) <= later:
            return None
        if restore_kind(attribute_or(self.cash_instance, "backend", None)) != "ram":
            return None
        copy = 0.0
        for value in variables.values():
            seconds = memory_backend.store_copy_seconds(value)
            if seconds is None:
                return None
            copy += seconds
        compute = execution.store_cost
        if copy <= compute:
            return None
        logger.debug("[SIZE_AWARE] intermediate kept out of RAM: copy %.2fs > compute %.2fs", copy, compute)
        return f"{KEPT_OUT_OF_RAM}: copying it (~{copy:.2f}s) costs more than computing it ({compute:.2f}s)"

    def _refusal(
        self,
        run: StatementRun,
        execution: StatementExecution,
        captured_vars: dict[str, Any],
        miss_guarded: bool,
        by_reference: set[str] = frozenset(),
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Whether *run*'s value is kept as metadata only, and why.

        Returns ``(should_skip, reason, prediction)``: *reason* may be None
        for a skip nothing needs to report, and *prediction* is the cost
        model's for the largest output. The gates run in order, each only if
        the ones before let the value through. The outputs named in
        *by_reference* are stored as references to call entries, which the
        restore-cost check leaves out: restoring one is restoring the call's
        result, which running the statement again would do too.
        """
        code, force_persist = run.code, run.force_persist
        # The restore-cost check is waived only for a statement that READS a
        # file itself: reading is the expensive part then. Waiving it for every
        # file the inputs were built from exempted everything downstream of a
        # load: ~400 MiB frames restoring slower than they computed, served as
        # hits. Nor for a file read inside a call the cache holds.
        reads_files = self._reads_files(execution, set(execution.accessed_files or ()))
        # A final value is weighed against what a restart would re-run to
        # get it back, not only its own statement (`final_rebuild_seconds`).
        rebuild = self._rebuild_cost.final_rebuild_seconds(
            run.inputs,
            run.outputs,
            execution.store_cost,
            in_loop=self._calls.in_loop,
            written_later=self.written_later_in_cell,
        )
        should_skip, skip_reason, prediction = self.should_skip_large_object_caching(
            captured_vars,
            rebuild,
            force_persist,
            has_file_dependencies=reads_files,
            exempt=by_reference,
        )

        # Statements whose outputs include a __main__-defined function or
        # class are never VALUE-cached: those values pickle BY
        # REFERENCE to a binding that won't exist in the next session, so a
        # value entry would either crash the lookup (dangling find_class ->
        # AttributeError) or "restore" nothing and wrongly skip the defining
        # statement. Route them through the metadata-only path (same as the
        # size-aware skip): output lineages persist for the upstream
        # simulation, while the value lookup misses cleanly and the cheap
        # statement (lambda assign, `g = f` alias) re-executes.
        if not should_skip:
            _unrestorable = sorted(
                _name
                for _name, _v in captured_vars.items()
                if (inspect.isfunction(_v) or inspect.isclass(_v))
                and attribute_or(_v, "__module__", None) == "__main__"
            )
            if _unrestorable:
                should_skip = True
                skip_reason = (
                    f"__main__ function/class output(s) "
                    f"{', '.join(_unrestorable)} are unrestorable by value; "
                    f"statement re-executes (lineage persists)"
                )
                if _is_only_definitions(code):
                    # Nothing to report for a ``def``: it always re-runs at no
                    # cost. The reason is for ``f = make_fn()``. A def reading
                    # file-loaded data got here past the too-cheap floor and
                    # its badge row said NOT CACHED.
                    skip_reason = None
        # The same route for a value holding a closure that keeps state --
        # `add = make_log()` with the factory imported from a module, so the
        # rule above does not see it. A function is kept by reference, so the
        # restore handed back the closure the last run made, still holding
        # what it had appended.
        if not should_skip:
            _stateful = sorted(_name for _name, _v in captured_vars.items() if holds_a_closure_with_state(_v))
            if _stateful:
                should_skip = True
                skip_reason = (
                    f"output(s) {', '.join(_stateful)} hold a function that changes its closure; "
                    f"statement re-executes (lineage persists)"
                )
        # A function kept by reference whose closure holds an object the
        # other outputs hold: ``a, getter = mk()`` with ``mk`` imported. The
        # restore gives ``a`` back as a copy and ``getter`` as the live
        # function, so they no longer share the array, and a write through
        # ``a`` did not reach ``getter()`` (21 for 27).
        if not should_skip:
            _budget = closure_check_budget(execution.cost)
            _closures = closures_sharing(captured_vars, _budget)
            _sharing = _closures.names
            if not _closures.finished:
                should_skip = True
                skip_reason = (
                    "cash could not look through the outputs in time; statement re-executes (lineage persists)"
                )
                _warn_closures_unchecked(run.outputs, _budget)
            elif _sharing:
                should_skip = True
                skip_reason = (
                    f"output(s) {', '.join(_sharing)} hold a function whose closure shares an object "
                    f"with the other outputs; statement re-executes (lineage persists)"
                )
        # Perpetual-miss guard. After the gates above so it can override their
        # exemptions: ``has_file_dependencies`` waives the whole size-aware
        # cost model, and a fit on a CSV-derived frame inherits the read's file
        # deps, so without this the frame is re-serialised every run for a
        # cache that can never hit. An unstable key does not become stable
        # because the statement touched a file. ``force_persist`` is checked by
        # the caller and is the one thing that outranks this.
        if not should_skip and miss_guarded:
            should_skip = True
            skip_reason = GUARD_SKIP_REASON

        # Loop-persist amplification guard. Placed after every other
        # gate so it only accounts writes that would ACTUALLY have happened, and
        # so it can override ``force_persist`` -- which is the whole point: a
        # user asking to persist one value must not silently get every
        # intermediate state of it written to their disk. It is the last word
        # because it is a disk-safety guard, not a cost heuristic.
        if not should_skip:
            amplified, amplified_reason = self._amplification.check(
                code,
                prediction,
                annotated=force_persist,
            )
            if amplified:
                should_skip = True
                skip_reason = amplified_reason
        return should_skip, skip_reason, prediction

    def _store_metadata_only(
        self,
        run: StatementRun,
        execution: StatementExecution,
        skip_reason: str | None,
        cost_fields: dict[str, Any],
    ) -> StatementCacheMetadata:
        """Record a refused value's metadata, lineages included, without the value."""
        skip_metadata = StatementCacheMetadata(
            timestamp=time.time(),
            inputs=list(run.inputs),
            outputs=list(run.outputs),
            execution_time=execution.cost,
            store_time=execution.store_cost,
            source_hash=run.source_hash,
            code=run.code,
            key=run.cache_key,
            skipped_reason=skip_reason,
            metadata_only=True,
            output_lineages=self._lineage_builder.build_output_lineages(self.tracking_state, run.outputs),
            input_lineages=self._lineage_builder.build_input_lineages(
                self.tracking_state, run.inputs, self.tracking_state.statement_entry_lineages.get(run.code)
            ),
            carriers_advanced=_carriers_advanced(run),
            replay=replay_record(execution.captured),
            **cost_fields,
        )
        try:
            backend = self.cash_instance.backend if self.cash_instance else None
            if backend is not None:
                backend.set_metadata_only(run.cache_key, skip_metadata.to_dict())
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("[PROCESSOR] Best-effort metadata persistence failed")
        return skip_metadata

    def _payload(
        self,
        run: StatementRun,
        execution: StatementExecution,
        captured_vars: dict[str, Any],
        seed_epochs: Mapping[str, str],
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """The value entry for *run*, and the call entries it refers to
        (key -> bytes) in place of the results they hold."""
        captured_output = execution.captured
        variables = self._filter_safe_vars(captured_vars)
        referenced: dict[str, int] = {}
        decorated = False
        if not run.holders and not run.anchors:
            # A reference restores a call's result on its own: with holders
            # or anchors it would no longer be the object they hold.
            variables = self._calls.with_call_refs(variables, run.code, referenced)
            variables, decorated = self._with_decorator_ref(variables, run.offered_as)
        payload = {
            "variables": variables,
            "stdout": captured_output.stdout,
            "stderr": captured_output.stderr,
            # Rich-display output (RichOutput objects). The 'outputs' key in
            # the sibling ``metadata`` dict holds variable NAMES — two distinct
            # concepts; keep them on different keys here too.
            "rich_outputs": captured_output.outputs,
            # Only the streams the statement moved or seeded: a hit leaves
            # every other stream where it stands, as a run would.
            "rng_state": dict(execution.rng_moved),
            # The streams it seeded as it ran: a hit adopts their seed as
            # the run did (`adopt_seeds_of_a_hit`).
            "rng_reseeded": sorted(execution.rng_reseeded),
            # The seeding regime this state was captured under, so a later
            # restore can tell whether replaying it would clobber a re-seed
            # rather than continue the stream.
            "rng_epochs": dict(seed_epochs),
        }
        if referenced or decorated:
            payload[REFS_FIELD] = True  # `has_call_refs`
        # The value the statement echoed as its cell's result, which a hit
        # hands to IPython's output history as a run does.
        if execution.echo:
            payload[ECHO_FIELD] = execution.echo[0]
        if run.anchors:
            # The objects of the variables the outputs hold as they were, as a
            # reference each (`anchors`): one round trip for the values and
            # the echo, which hold the same objects.
            user_ns = self.shell.user_ns
            held = {"variables": variables, ECHO_FIELD: payload.get(ECHO_FIELD)}
            try:
                held = with_anchor_refs(held, {name: user_ns[name] for name in run.anchors})
            except AnchorUnrebuilt as exc:
                cause = exc.__cause__ or exc
                raise _AnchorsUnstorable(f"{type(cause).__name__}: {cause}", [exc.name] if exc.name else ()) from cause
            except Exception as exc:  # the value's own pickling: STORE-FAILED, as without anchors
                raise _AnchorsUnstorable(f"{type(exc).__name__}: {exc}", ()) from exc
            payload["variables"] = held["variables"]
            if ECHO_FIELD in payload:
                payload[ECHO_FIELD] = held[ECHO_FIELD]

        # the module-global RNG post-state above misses generators the
        # user holds in a variable (``rng = np.random.default_rng(42)``).
        # Capture those too, scoped to this statement's inputs so the cost
        # stays proportional to what the statement actually reads.  Omitted
        # entirely when there are none, keeping the payload shape unchanged for
        # the overwhelming majority of statements. A generator a called
        # function draws from through a global (``def model(x): return
        # x + rng.randn(len(x))``) is no input, yet the run moved it: a hit
        # that left it behind would hand the next draw numbers already drawn.
        try:
            read = set(run.inputs) | (run.carriers_advanced or set())
            object_rng_states = capture_object_rng_states(read, self.shell.user_ns)
            if object_rng_states:
                payload["rng_object_states"] = object_rng_states
        except (TypeError, AttributeError) as e:
            logger.debug("[RANDOMNESS] Object RNG capture skipped: %s", e)
        # Which of them it drew from, so a hit gives them the lineage the run
        # gave them (`carrier_advances`).
        if run.carriers_advanced is not None:
            payload[CARRIERS_FIELD] = sorted(run.carriers_advanced)
        return payload, referenced

    def _with_decorator_ref(self, variables: dict[str, Any], offered_as: str | None) -> tuple[dict[str, Any], bool]:
        """*variables* with the result of the ``@cash.cache`` call the
        statement is (``x = f(a)``) stored as a reference to that call's own
        entry (`handed_digests.stored_result`), and whether one was made.

        The decorator's entry already holds that value: a second copy cost a
        second RAM copy and a second disk write of a 100-900 MB result on
        every compute. Only into the backend the statement is stored in, and
        only when the entry is persisted wherever the statement may be. The
        call's key is not added to the statement's ``call_refs``: those are
        the notebook's own call entries, which pruning a statement version
        may delete, and a decorator's entry is the user's.
        """
        stored = None if offered_as is None else handed_digests.stored_result(offered_as)
        if stored is None or not stored.persisted or stored.backend is not self.cash_instance.backend:
            return variables, False
        names = [name for name, value in variables.items() if stored.holds(value)]
        if len(names) != 1:
            return variables, False
        return {**variables, names[0]: CallRef(stored.key, DECORATOR_ENTRY)}, True

    def _wire(self, run: StatementRun, metadata: StatementCacheMetadata, referenced: dict[str, int]) -> dict[str, Any]:
        """*metadata* as the dict the backend stores.

        Dict-on-the-wire: the backend round-trips a plain dict and may inject
        the resolved ``storage`` destinations back into it, so the caller
        re-wraps the mutated dict to carry the storage info on to the badge.
        """
        wire = metadata.to_dict()
        if referenced:
            wire[REFS_FIELD] = sorted(referenced)
            wire[REF_BYTES_FIELD] = sum(referenced.values())
        # The ``@cash.cache`` functions the statement called, so a notice about
        # its value does not advise decorating a function that already is.
        decorated = _user_decorated_calls(run.metrics.get("decorator_calls"))
        if decorated:
            wire[DECORATED_CALLS_FIELD] = decorated
        # An intermediate of this cell (``cell_runs.written_later_in_cell``)
        # stays in RAM; the cell's final version is persisted at its end.
        later = self.written_later_in_cell
        if not run.force_persist and run.outputs and later and set(run.outputs) <= later:
            wire["defer_persist"] = True
        return wire

    def _write(
        self, run: StatementRun, payload: dict[str, Any], wire: dict[str, Any], prediction: dict[str, Any] | None
    ) -> None:
        """Write the entry, and the metadata-only record a RAM-only value needs."""
        cache_key = run.cache_key
        if prediction and prediction.get("class_key") is not None:
            # A RAM tier that keeps the entry as pickle bytes credits its first
            # hit's reading to this class (`InMemoryBackend._read_pickled`).
            wire[memory_backend.ROUND_TRIP_CLASS] = prediction["class_key"]
        nestable = _nestable(payload)
        if nestable is not None:
            stored_pickle.nest(payload, *nestable)
        try:
            self.cash_instance.backend.set(cache_key, payload, wire)
        except Exception as e:  # noqa: BLE001 - copying runs the value's own code (``__deepcopy__``, ``__reduce__``)
            _warn_not_stored(e)
        else:
            # Charge this write to its statement's amplification budget, now
            # that the backend has reported which tiers actually took it.
            # Only durable destinations count.
            self._amplification.account(run.code, prediction, wire)
            _learn_round_trip(prediction, wire)

        # The metadata-only record keeps a RAM-only value's lineage across a
        # restart. A value written to a persistent tier carries its metadata
        # already, and asking for the record made the disk tier wait for that
        # write (``FileBackend.set_metadata_only``) only to skip it: a
        # cleaning cell spent 5.8 s of its cold run there, on ~500 MB frames
        # whose write was meant to happen in the background.
        persisted = any(d != "RAM" for d in (wire.get("storage") or ()))
        try:
            backend = self.cash_instance.backend
            if backend is not None and not persisted:
                backend.set_metadata_only(cache_key, wire)
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("[PROCESSOR] Best-effort metadata persistence failed")


#: The start of the skip reason of an intermediate the RAM tier does not
#: keep (`StatementStore._copy_dearer_than_compute`).
KEPT_OUT_OF_RAM = "kept out of RAM: copy dearer than compute"


def _nestable(payload: dict[str, Any]) -> tuple[Any, list[Any]] | None:
    """A variable of *payload* whose pickle its digest kept
    (`stored_pickle.kept`), when the entry may hold that pickle as it is
    (`stored_pickle.nest`): every other variable, and the echoed value, is
    that value, immutable data, or found by that digest to share no object
    with it (`stored_pickle.apart`) -- two pickles would make two objects
    of one. The rest of the entry is cash's own (printed text, RNG
    states). Else one whose digest ran out of its budget, leaving its
    pickle to the store (`stored_pickle.pending`), which proves the others
    apart as it makes it. ``(value, the parts still to prove apart)``, or
    None."""
    parts = [value for value in payload["variables"].values() if not _immutable(value)]
    echo = payload.get(ECHO_FIELD)
    if echo is not None and not _immutable(echo):
        parts.append(echo)
    for found in parts:
        if stored_pickle.kept(found) is not None and all(
            other is found or stored_pickle.apart(found, other) for other in parts
        ):
            return found, []
    for found in parts:
        if stored_pickle.pending(found):
            return found, [other for other in parts if other is not found]
    return None


#: The most items `_immutable` looks through.
_IMMUTABLE_ITEMS = 4096


def _immutable(value: Any) -> bool:
    """Is *value* data no object can be part of: an immutable leaf, a call
    reference, or tuples and frozensets of those (at most
    `_IMMUTABLE_ITEMS` in all)?"""
    budget = [_IMMUTABLE_ITEMS]

    def walk(item: Any) -> bool:
        kind = type(item)
        if kind in _LEAVES:
            return True
        if kind is tuple or kind is frozenset:
            budget[0] -= len(item)
            return budget[0] >= 0 and all(map(walk, item))
        return False

    return walk(value)


_LEAVES = frozenset((*IMMUTABLE_LEAF_TYPES, CallRef))


def _learn_round_trip(prediction: dict[str, Any] | None, wire: dict[str, Any]) -> None:
    """Credit what unpickling the RAM tier's copy of the entry took to the
    class of its biggest value, when that is an object of a class of its own: the next
    store of one is then judged on it before anything is copied
    (`memory_backend.round_trip_seconds`). ``pooled`` (a statsmodels result
    holding its 145 MiB design matrix) was copied on every run, for a hit
    as slow as its 0.8 s fit."""
    seconds = wire.get(memory_backend.MEASURED_ROUND_TRIP)
    if not seconds or not prediction or prediction.get("class_key") is None:
        return
    memory_backend.note_round_trip(prediction["class_key"], int(prediction["size_bytes"]), float(seconds))


def _only_shows_a_name(tree: ast.Module | None) -> bool:
    """Whether the statement is a bare name, which only shows its value."""
    return (
        tree is not None
        and len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Name)
    )


def _user_decorated_calls(events: Any) -> list[str]:
    """The names of the ``@cash.cache`` functions in *events* (the statement's
    ``decorator_calls``), leaving out the calls cash wrapped itself."""
    names: dict[str, None] = {}
    for event in events or ():
        if isinstance(event, dict) and not event.get("intercepted"):
            name = event.get("func_name")
            if isinstance(name, str) and name:
                names[name] = None
    return list(names)
