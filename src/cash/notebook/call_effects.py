"""What an intercepted call does besides return: observed on a miss, replayed on a hit.

A call served from the cache does not run, so nothing it would have done
happens unless it is put back: the files it read go onto the statement's
tracker (:func:`replay_deps`), what it printed goes onto the live stream
(:func:`replay_output`), and the globals it wrote go back into its namespace
(:func:`restore_globals`). On a miss the same effects are recorded for that
later hit (:func:`call_capturing_output`, :func:`capture_globals`), and the
arguments are hashed around the call (:func:`hash_args`) so a callee that
mutates one is never cached.
"""

from __future__ import annotations

import copy as _copy
import logging
import sys
from collections.abc import Mapping
from typing import Any

from cash.notebook._tee import TeeWriter
from cash.object_hashing import compute_hash, is_identity_fallback_hash
from cash.tracking.file_dep_snapshot import dep_path_for_this_process
from cash.tracking.tracker_context import active_tracker

logger = logging.getLogger(__name__)

__all__ = [
    "UNWRAP_FAILED",
    "call_capturing_output",
    "capture_globals",
    "hash_args",
    "replay_deps",
    "replay_output",
    "restore_globals",
    "unwrap_callee_globals",
]


#: Returned by :func:`unwrap_callee_globals` when an entry claims to carry a
#: callee's captured globals but does not have the shape to prove it. A unique
#: sentinel rather than ``None``, because ``None`` is a perfectly good cached
#: value and must stay distinguishable from a broken entry.
UNWRAP_FAILED = object()


def unwrap_callee_globals(value, metadata: Mapping[str, Any]):
    """Split a stored value into ``(result, captured_globals)``.

    Entries that captured a callee's writes to globals store
    ``(result, {name: value})`` and set a plain ``has_callee_globals`` bool in
    metadata; every other entry stores the bare result. The flag makes the
    shape self-describing rather than something the reader has to guess from
    the value's type -- a cached call that legitimately returns a 2-tuple would
    otherwise be indistinguishable from a wrapped one.

    Returns ``(UNWRAP_FAILED, None)`` when the flag is set and the shape does
    not match. That can only mean a corrupt or hand-edited entry, and handing
    back the tuple as though it were the result would be a silently wrong
    value where a miss merely costs a recompute.
    """
    if not metadata.get("has_callee_globals"):
        return value, None
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        return value[0], value[1]
    logger.debug("call unit: entry claims captured globals but has the wrong shape")
    return UNWRAP_FAILED, None


def call_capturing_output(fn, args: tuple, kwargs: dict) -> tuple[Any, str, str]:
    """Run *fn*, returning ``(result, stdout_text, stderr_text)``.

    Tees ``sys.stdout``/``sys.stderr`` through a recorder that still
    forwards every byte to the stream that was live going in -- which,
    during a real statement execution, IS the statement's own ambient
    capture (a ``StringIO``, a ``TeeWriter``, or the real terminal
    outside any capture). So a genuine miss looks exactly as it did
    before this method existed: the callee's output reaches the
    statement's capture "for free", untouched.
    The recording exists only so a LATER hit (:func:`replay_output`) has
    something to write back onto the live stream -- otherwise that
    output is simply gone, since the callee does not run at all on a hit.
    """
    __tracebackhide__ = True
    old_stdout, old_stderr = sys.stdout, sys.stderr
    tee_out = TeeWriter(old_stdout)
    tee_err = TeeWriter(old_stderr)
    sys.stdout, sys.stderr = tee_out, tee_err
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return result, tee_out.getvalue(), tee_err.getvalue()


def replay_deps(metadata: Mapping[str, Any]) -> None:
    """Re-declare a hit entry's recorded file/remote deps as though THIS
    call had just read them, onto the statement's ambient tracker.

    Attribution AND propagation: the call unit already validated these
    deps before serving the hit (they are checked as part of the
    entry's own freshness -- a stale file behind ``key`` simply misses,
    same as the statement path), and the enclosing statement still needs
    them registered because its own cached value transitively depends on
    the same files. Without this, a statement that only reaches a file
    through a now-cached sub-call would lose that dependency the moment
    the sub-call started hitting -- exactly the regression this
    closes.
    """
    snap = metadata.get("auto_file_deps")
    if not snap:
        return
    try:
        tracker = active_tracker.get()
    except Exception:  # noqa: BLE001 - tracking is best-effort
        return
    if tracker is None:
        return
    for path, recorded in snap.items():
        try:
            # A remote entry must go back onto the remote channel --
            # routed to ``add_tracked`` it would enter the file set, be
            # stat'ed, and be dropped, same reasoning as
            # ``propagate_file_deps_to_active_tracker`` in decorator/file_deps.py.
            if isinstance(recorded, dict) and recorded.get("remote"):
                tracker.add_tracked_remote(path)
            else:
                # The file THIS process reads, as the decorator replays it.
                tracker.add_tracked(dep_path_for_this_process(path, recorded))
        except Exception:  # noqa: BLE001 - a dep that cannot be replayed is left out
            logger.debug("call unit: could not replay dep %r", path)


def replay_output(metadata: Mapping[str, Any]) -> None:
    """Write a hit entry's recorded stdout/stderr onto the LIVE stream.

    The callee did not run this time, so its prints never happened;
    writing the recorded text to ``sys.stdout``/``sys.stderr`` puts it
    back wherever the statement's ambient capture currently points (a
    buffer during a real run, the real terminal in a bare unit test),
    which is what reconstructs ``print(a); f(x); print(b)``'s
    interleaving without the statement path needing to know sub-call
    caching exists at all.
    """
    stdout_text = metadata.get("stdout") or ""
    stderr_text = metadata.get("stderr") or ""
    if stdout_text:
        try:
            sys.stdout.write(stdout_text)
        except (OSError, ValueError, TypeError, AttributeError):  # a closed or foreign stream
            logger.debug("call unit: could not replay stdout", exc_info=True)
    if stderr_text:
        try:
            sys.stderr.write(stderr_text)
        except (OSError, ValueError, TypeError, AttributeError):  # a closed or foreign stream
            logger.debug("call unit: could not replay stderr", exc_info=True)


def hash_args(args: tuple, kwargs: dict) -> tuple:
    """Content hashes of the live arguments, for mutation detection.

    Uses the sampling hash (`compute_hash`) deliberately, not
    `compute_hash_full`: it is the same one the statement path's own
    content observation uses, and for a large frame a full hash per call
    would cost more than the call being cached is worth.

    Two DIFFERENT ways this can under-report a mutation, and they get
    different treatment:

    1. **Sampling.** `compute_hash` samples large objects (ndarray: first
       100 elements, DataFrame: first 5 rows, collections >200: head/tail).
       A same-size in-place edit outside the sampled region is invisible
       here. This is a known, accepted trade -- it errs toward CACHING for
       objects that still hash BY CONTENT, and the identity check in
       `CallEntries.storable` stays as a second line of defence for the one shape it
       fully covers (`return arg`).

    2. **Identity fallback.** `compute_hash`'s tier 3
       (`object_hashing.identity_hash`) hashes `id(obj)`, not the object's
       data, once pickling itself has failed (a `threading.Lock`, a socket,
       an open file, anything with an unpicklable `__reduce__`). `id(obj)`
       is invariant across an in-place mutation of that SAME object, so
       this is not "a coarser content hash" the way sampling is -- it is
       BLIND to every mutation of that argument, always, for the entire
       unpicklable-object class. Comparing two such hashes before/after a
       call would silently read as "unchanged" even when the callee
       mutated the object, which directly contradicts "fail closed": a
       value flagged via `is_identity_fallback_hash` is therefore replaced
       with a fresh, single-use sentinel (`object()`) instead of the hash
       string. Two distinct `object()` instances are never `==`, so the
       before/after comparison (`CallUnit._did_what_a_hit_cannot`) always
       reads as "changed" for that argument -- i.e. "cannot prove this argument is clean" is
       treated the same as "proved it changed", which is the fail-closed
       direction the task requires.
    """
    out = []
    for value in (*args, *kwargs.values()):
        try:
            h = compute_hash(value)
        except Exception:  # noqa: BLE001 - see the comment below
            # This branch IS live, on every Python before 3.14: hashing an
            # instance of a locally-defined class raises
            # `AttributeError: Can't pickle local object '<f>.<locals>.C'`
            # rather than reaching `compute_hash`'s identity fallback. A
            # class defined inside a function is ordinary in a notebook and
            # ubiquitous in tests.
            #
            # It must append the SAME single-use sentinel as the
            # identity-fallback case below, and for the same reason. This
            # used to append `None`, on the reasoning that "'cannot prove
            # unmutated' is exactly what a `None` here already means to the
            # caller" -- but `None == None`, so two unknowable snapshots
            # compared EQUAL and read as "argument unchanged". That is
            # fail-OPEN: a callee mutating an unpicklable argument was
            # cached and its mutation silently skipped, on 3.10-3.13.
            out.append(object())
            continue
        if is_identity_fallback_hash(value, h):
            out.append(object())
        else:
            out.append(h)
    return tuple(out)


def capture_globals(fn, names: tuple[str, ...]) -> dict[str, Any] | None:
    """Post-call values of the globals *fn* writes, or ``None`` to refuse.

    ``None`` is returned when any watched name cannot be captured soundly:

    * **absent** — it was there when the watch list was filtered and is not
      now, so this call's effect on it cannot be described;
    * **identity-fallback hash** — ``compute_hash`` fell through to
      ``sha256(str(id(obj)))`` because the object does not pickle (a lock,
      a socket, an open file). ``id`` is invariant across an in-place
      mutation, so a later key comparison on this name is BLIND, always,
      for that whole class. Storing an entry whose pre-state cannot be
      told apart is exactly the partial-accumulator hazard.

    Returning ``None`` costs a permanently-uncached site. Storing anyway
    would cost a silently wrong restore, and this method exists to prefer
    the former.

    **Deep-copied, not referenced.** The whole point of this capture is a
    POST-CALL snapshot, and the object being snapshotted is by construction
    one that gets mutated in place -- so keeping a reference does not
    capture a state at all, it captures a live handle that keeps changing.
    The RAM tier stores metadata as given, so a later call mutating the
    same object silently rewrites an already-stored entry's recorded
    "post-state". Measured::

        cell 3   a = next_seq()            stores N -> [1]
        cell 5   seen.append(next_seq())   mutates the SAME list to [2]
        rerun    cell 3 hits, restores N -> [2]   (not [1])

    which then re-keyed cell 5's call against a pre-state that had never
    existed, so it missed forever -- and the two spellings diverged.
    A copy failure is treated like any other
    "cannot capture this soundly": refuse.
    """
    if not names:
        return {}
    globals_dict = getattr(fn, "__globals__", None) or {}
    captured: dict[str, Any] = {}
    for name in names:
        if name not in globals_dict:
            return None
        value = globals_dict[name]
        try:
            if is_identity_fallback_hash(value, compute_hash(value)):
                return None
            captured[name] = _copy.deepcopy(value)
        except Exception:  # noqa: BLE001 - cannot prove it is capturable
            return None
    return captured


def restore_globals(fn, names: tuple[str, ...], recorded: Mapping[str, Any] | None) -> None:
    """Land a hit entry's recorded post-call globals back into *fn*'s own
    namespace, so the callee's write survives a call it did not make.

    The counterpart of :func:`replay_output` for state rather than text,
    and the call-level twin of what the statement path does by listing the
    same names in its ``outputs``.

    A rebind, not an in-place transfer. That matches the statement path's
    default restore (``StatementRestorer._write_restored_value`` only
    transfers in place for an explicitly-listed estimator-fit receiver), so
    the two spellings of the same code land the value the same way. An
    alias taken BEFORE the restore therefore keeps pointing at the old
    object — a real limitation, and the same one the statement path has
    always had for every restored variable.

    Only names in *names* are written. The entry could carry a stale name
    from a since-edited callee, and honouring it would resurrect a variable
    the current source never mentions.
    """
    if not names:
        return
    if not isinstance(recorded, Mapping) or not recorded:
        return
    globals_dict = getattr(fn, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return
    for name in names:
        if name in recorded:
            # A COPY, for the mirror of the reason `capture_globals`
            # copies: handing back the stored object would make the live,
            # about-to-be-mutated variable and the cache entry the same
            # object, so the next call would rewrite the entry it was just
            # served from.
            try:
                globals_dict[name] = _copy.deepcopy(recorded[name])
            except Exception:  # noqa: BLE001 - a restore must never crash
                logger.debug("call unit: could not restore global %r", name)
