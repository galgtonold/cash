"""What an intercepted call does besides return: observed on a miss, replayed on a hit.

A call served from the cache does not run, so nothing it would have done
happens unless it is put back: the files it read go onto the statement's
tracker (:func:`replay_deps`), what it printed goes onto the live stream
(:func:`replay_output`), and the globals it wrote go back into its namespace
(:func:`restore_globals`), as do the objects in its closure it changed in
place. On a miss the same effects are recorded for that
later hit (:func:`call_capturing_output`, :func:`capture_globals`), and the
arguments are hashed around the call (:func:`hash_args`) so a callee that
mutates one is never cached, as is one that rebinds a variable of the
function it was made in (:func:`rebinds_its_closure`).
"""

from __future__ import annotations

import ast
import collections
import copy as _copy
import dis as _dis
import functools
import inspect as _inspect
import logging
import sys
import types as _types
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from cash.decorator.arg_hashing import (
    exposure_mark,
    frame_borrows_its_data,
    frame_signature,
    is_cow_pandas,
    watch_array_handles,
)
from cash.notebook._tee import TeeWriter
from cash.sizing import pandas_nbytes
from cash.tracking.file_dep_snapshot import dep_path_for_this_process
from cash.tracking.tracker_context import active_tracker
from cash.value_hash import compute_hash, is_identity_fallback_hash

logger = logging.getLogger(__name__)

__all__ = [
    "UNWRAP_FAILED",
    "ArgFingerprints",
    "DigestHandoff",
    "call_capturing_output",
    "capture_globals",
    "closure_cells",
    "hash_args",
    "rebinds_its_closure",
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


def replay_deps(metadata: Mapping[str, Any]) -> frozenset[str]:
    """Re-declare a hit entry's recorded file/remote deps as though THIS
    call had just read them, onto the statement's ambient tracker, and
    return the paths and URLs declared.

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
        return frozenset()
    try:
        tracker = active_tracker.get()
    except Exception:  # noqa: BLE001 - tracking is best-effort
        tracker = None
    declared: set[str] = set()
    for path, recorded in snap.items():
        try:
            # A remote entry must go back onto the remote channel --
            # routed to ``add_tracked`` it would enter the file set, be
            # stat'ed, and be dropped, same reasoning as
            # ``propagate_file_deps_to_active_tracker`` in decorator/file_deps.py.
            if isinstance(recorded, dict) and recorded.get("remote"):
                declared.add(path)
                if tracker is not None:
                    tracker.add_tracked_remote(path)
            else:
                # The file THIS process reads, as the decorator replays it.
                local = dep_path_for_this_process(path, recorded)
                declared.add(local)
                if tracker is not None:
                    tracker.add_tracked(local)
        except Exception:  # noqa: BLE001 - a dep that cannot be replayed is left out
            logger.debug("call unit: could not replay dep %r", path)
    return frozenset(declared)


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


def closure_cells(fn) -> dict[str, _types.CellType]:
    """*fn*'s closure cells, by the name its code reads each one under.

    Those of the function under any ``functools.wraps`` wrapper, whose
    source is the one analysed for what the call writes.
    """
    try:
        target = _inspect.unwrap(fn)
    except ValueError:  # a __wrapped__ cycle
        return {}
    code = getattr(target, "__code__", None)
    closure = getattr(target, "__closure__", None)
    if not isinstance(code, _types.CodeType) or not closure:
        return {}
    return dict(zip(code.co_freevars, closure))


#: What a closure's object can be to be put back in place on a hit: a
#: container whose whole content one assignment or ``clear`` + ``update``
#: replaces. Anything else -- a user object, a frame, an array that grew --
#: cannot be written back into the live object soundly.
_IN_PLACE_TYPES = (list, dict, set, bytearray, collections.deque)


def _put_back_in_place(cell: _types.CellType, value) -> None:
    """Make the object in *cell* hold *value*'s content, keeping the object."""
    try:
        live = cell.cell_contents
    except ValueError:  # emptied since: nothing holds the old object
        cell.cell_contents = value
        return
    if type(live) is not type(value):
        cell.cell_contents = value
    elif isinstance(live, (list, bytearray)):
        live[:] = value
    elif isinstance(live, collections.deque):
        live.clear()
        live.extend(value)
    else:
        live.clear()
        live.update(value)


def rebinds_its_closure(fn) -> bool:
    """Whether calling *fn* rebinds a variable of the function it was made
    in: ``nonlocal count; count += 1``.

    A hit cannot put that back, and the key never sees it: the cell is not
    in the callee's lineage, and an int in a cell passes as plain data. So
    ``val1 = counter(); val2 = counter()`` read ``1, 1`` whenever the first
    call was slow enough to be stored. The globals a callee writes are
    captured and restored (:func:`capture_globals`); a cell is not a global,
    so such a callee is not cached.
    """
    try:
        code = getattr(_inspect.unwrap(fn), "__code__", None)
    except ValueError:  # a __wrapped__ cycle
        return False
    if not isinstance(code, _types.CodeType) or not code.co_freevars:
        return False
    return _rebinds(code, frozenset(code.co_freevars))


@functools.lru_cache(maxsize=4096)
def _rebinds(code: _types.CodeType, cells: frozenset[str]) -> bool:
    """Whether *code*, or a function defined in it, stores to or deletes one
    of *cells*, the enclosing function's variables it reaches."""
    for ins in _dis.get_instructions(code):
        if ins.opname in ("STORE_DEREF", "DELETE_DEREF") and ins.argval in cells:
            return True
    for const in code.co_consts:
        if isinstance(const, _types.CodeType):
            # Its own variable of the same name shadows the outer one.
            reached = (cells & frozenset(const.co_freevars)) - frozenset(const.co_cellvars)
            if reached and _rebinds(const, reached):
                return True
    return False


#: A frame smaller than this is hashed again rather than remembered: its
#: hash costs about what checking it costs.
_FINGERPRINT_MIN_BYTES = 1 << 20


class ArgFingerprints:
    """The content hashes of the pandas frames calls received, kept while
    the frame provably has not changed since it was hashed.

    ``scores = [evaluate(df, a) for a in alphas]`` hashed ``df`` in full
    before and after every call, to see whether the callee changed it: 0.7 s
    a call for an 80 MB frame. Under copy-on-write a frame is checked
    instead, as the decorator's arguments are: the hash is kept with a
    shallow copy of the frame, which makes every write through pandas give
    the frame new arrays, and with the frame's signature (`frame_signature`:
    the identities of its manager, blocks and axes, its axis names and
    ``attrs``). A write past pandas -- through a handle it gave out
    (``.array``, a read-only view made writable), or into an array the
    frame was built over -- makes the frame borrowed
    (`frame_borrows_its_data`), and it is hashed again. Anything that is
    not such a frame is hashed every time.

    Kept for one cell (`clear`), and an entry only while its frame lives.
    """

    def __init__(self) -> None:
        #: ``id(frame) -> (weak reference, shallow copy, signature, hash,
        #: exposure mark)``.
        self._memo: dict[int, tuple] = {}

    def clear(self) -> None:
        self._memo.clear()

    def digest(self, value: Any) -> str:
        """`compute_hash` of *value*, from the memo when it is a frame
        that has not changed since it was hashed."""
        if not is_cow_pandas(value):
            return compute_hash(value)
        found = self._lookup(value)
        if found is not None:
            return found
        since = exposure_mark()
        digest = compute_hash(value)
        if not is_identity_fallback_hash(value, digest) and (pandas_nbytes(value) or 0) >= _FINGERPRINT_MIN_BYTES:
            self._store(value, digest, since)
        return digest

    def _lookup(self, value: Any) -> str | None:
        entry = self._memo.get(id(value))
        if entry is None:
            return None
        ref, held, signature, digest, since = entry
        try:
            if (
                ref() is value
                and not frame_borrows_its_data(value, held, since)
                and frame_signature(value) == signature
            ):
                return digest
        except Exception:  # noqa: BLE001 - a pandas internals change: hash again
            logger.debug("call unit: could not check a remembered frame", exc_info=True)
        self._memo.pop(id(value), None)
        return None

    def _store(self, value: Any, digest: str, since: int) -> None:
        """Remember *value*'s hash, *since* being the `exposure_mark` taken
        before it was hashed; nothing for a frame something outside pandas
        can write."""
        try:
            watch_array_handles()
            held = value.copy(deep=False)
            if frame_borrows_its_data(value, held, since):
                return
            signature = frame_signature(value)
            memo = self._memo
            key = id(value)
            ref = weakref.ref(value, lambda _ref, key=key, memo=memo: memo.pop(key, None))
        except Exception:  # noqa: BLE001 - the memo is a speedup: hash every time
            logger.debug("call unit: could not remember a frame", exc_info=True)
            return
        self._memo[key] = (ref, held, signature, digest, since)


def _call_shape(tree: ast.Module | None) -> tuple[bool, bool]:
    """``(nothing_runs_before, nothing_runs_after)`` the one call *tree* is.

    *nothing_runs_before*: the statement is one call of a bare name on bare
    names and constants (``y = f(x, 3, k=z)``, or ``f(x)`` alone), so until
    that call starts only names are read. *nothing_runs_after*: besides,
    its result is bound to one name and nothing else, so once it returns
    only the binding happens -- not a display, not an unpacking that
    iterates the result, not an attribute or item store, not an annotation.
    """
    if tree is None or len(tree.body) != 1:
        return False, False
    node = tree.body[0]
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        call, after = node.value, True
    elif isinstance(node, ast.Expr):
        call, after = node.value, False
    else:
        return False, False
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return False, False
    simple = (ast.Name, ast.Constant)
    if not all(isinstance(a, simple) for a in call.args):
        return False, False
    if not all(k.arg is not None and isinstance(k.value, simple) for k in call.keywords):
        return False, False
    return True, after


class DigestHandoff:
    """The content digests of one statement's arguments, handed from one
    check that reads them to the next, so a big argument is hashed once
    before its call and once after.

    ``y = normalize(x)`` hashed ``x`` four times on a first run: the
    statement's in-place-change fingerprint before it ran
    (`MutationClassifier.classify`), the call's argument hash before and
    after the call (`hash_args`), and the statement's fingerprint after
    (`MutationClassifier.observed_mutations`) -- 19.1 s where plain Python
    took 0.89 s over a 257 MB frame. Between the first two only names are
    read, and between the last two only a name is bound, when the statement
    is nothing but that call (`_call_shape`); there the digest taken by one
    is the digest the other would take. Anywhere else each takes its own.

    Every digest goes through *fingerprints* (`ArgFingerprints`), so a
    frame under copy-on-write that provably has not changed since it was
    hashed is not read again, whichever check asks.

    Held per statement (`begin_statement`); a statement that runs another
    statement inside its call (a nested ``run_cell``) starts a new
    generation, and nothing from the older one is handed on.
    """

    def __init__(self) -> None:
        self.fingerprints = ArgFingerprints()
        self._generation = 0
        self._before_ok = False
        self._after_ok = False
        #: ``id(value) -> (value, digest)``: what the statement's fingerprint
        #: read before it ran, for its call's hash before the call; and what
        #: the call's hash read after it, for the statement's fingerprint.
        #: Each holds its values, so an id cannot be reused meanwhile.
        self._before: dict[int, tuple[Any, str]] = {}
        self._after: dict[int, tuple[Any, str]] = {}
        #: Calls through the call cache started in this generation, and their
        #: count when `_after` was filled.
        self._started = 0
        self._after_at = -1

    def begin_cell(self) -> None:
        self.fingerprints.clear()

    def begin_statement(self) -> None:
        """A statement starts: nothing earlier is handed on."""
        self._generation += 1
        self._before_ok = self._after_ok = False
        self._before.clear()
        self._after.clear()
        self._started = 0
        self._after_at = -1

    def watch(self, tree: ast.Module | None) -> None:
        """The statement *tree* is about to have its arguments fingerprinted."""
        self._before_ok, self._after_ok = _call_shape(tree)

    def digest(self, value: Any) -> str:
        """`compute_hash` of *value*, kept for the call's hash before the
        call when nothing can run in between."""
        digest = self.fingerprints.digest(value)
        if self._before_ok and self._started == 0 and not is_identity_fallback_hash(value, digest):
            self._before[id(value)] = (value, digest)
        return digest

    def call_started(self) -> tuple[int, int]:
        """A call through the call cache starts; its ticket for
        `digest_before` and `note_after`."""
        self._started += 1
        if self._started > 1:
            self._before.clear()
        return self._generation, self._started

    def digest_before(self, ticket: tuple[int, int]) -> Callable[[Any], str]:
        """How the call holding *ticket* hashes an argument before it runs."""
        if ticket != (self._generation, 1) or not self._before:
            return self.fingerprints.digest
        before = self._before

        def digest(value: Any) -> str:
            found = before.get(id(value))
            if found is not None and found[0] is value:
                return found[1]
            return self.fingerprints.digest(value)

        return digest

    def note_after(self, ticket: tuple[int, int], values: tuple, digests: tuple) -> None:
        """The call holding *ticket* returned and hashed *values* to
        *digests*: kept for the statement's fingerprint after it, when the
        call is the whole statement and only its result's binding follows."""
        if ticket[0] != self._generation or ticket[1] != 1 or not self._after_ok:
            return
        self._after = {id(v): (v, d) for v, d in zip(values, digests) if isinstance(d, str)}
        self._after_at = self._started

    def digest_after(self, value: Any) -> str:
        """`compute_hash` of *value* for the statement's fingerprint after
        it ran: the call's own when no other call started since."""
        if self._after and self._after_at == self._started:
            found = self._after.get(id(value))
            if found is not None and found[0] is value:
                return found[1]
        return self.fingerprints.digest(value)


def hash_args(
    args: tuple,
    kwargs: dict,
    fingerprints: ArgFingerprints | None = None,
    digest: Callable[[Any], str] | None = None,
) -> tuple:
    """Content hashes of the live arguments, for mutation detection.

    Every byte of every argument (`compute_hash`): a callee that edits a
    frame's 500th row in place and returns something else must read as
    changed, or its result is stored and a hit skips the edit. Paid only on
    the miss path, twice per argument; a site whose keying and hashing cost
    more than the call is run plain by the call-site guard. With
    *fingerprints*, a pandas frame that provably has not changed since it
    was last hashed is not read again (`ArgFingerprints`). *digest*, when
    given, hashes each value in their place (`DigestHandoff.digest_before`).

    An argument whose content cannot be read at all fails closed:

    **Identity fallback.** `compute_hash`'s tier 3
    (`value_hash.identity_hash`) hashes `id(obj)`, not the object's
    data, once pickling itself has failed (a `threading.Lock`, a socket,
    an open file, anything with an unpicklable `__reduce__`). `id(obj)`
    is invariant across an in-place mutation of that SAME object, so
    this is not a content hash at all -- it is
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
            if digest is not None:
                h = digest(value)
            else:
                h = compute_hash(value) if fingerprints is None else fingerprints.digest(value)
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

    * **a closure's object that cannot be put back in place** -- see
      :data:`_IN_PLACE_TYPES`. A name the callee reads from its closure is
      captured from the cell, not from its globals.

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
    cells = closure_cells(fn)
    captured: dict[str, Any] = {}
    for name in names:
        if name in cells:
            try:
                value = cells[name].cell_contents
            except ValueError:  # an empty cell
                return None
            if not isinstance(value, _IN_PLACE_TYPES):
                return None
        elif name in globals_dict:
            value = globals_dict[name]
        else:
            return None
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

    A name the callee reads from its closure is different: the content goes
    back INTO the object the cell holds (:func:`_put_back_in_place`), so the
    list a factory handed out alongside the closure, and every other holder
    of it, sees what the calls appended.

    Only names in *names* are written. The entry could carry a stale name
    from a since-edited callee, and honouring it would resurrect a variable
    the current source never mentions.
    """
    if not names:
        return
    if not isinstance(recorded, Mapping) or not recorded:
        return
    globals_dict = getattr(fn, "__globals__", None)
    cells = closure_cells(fn)
    for name in names:
        if name in recorded:
            # A COPY, for the mirror of the reason `capture_globals`
            # copies: handing back the stored object would make the live,
            # about-to-be-mutated variable and the cache entry the same
            # object, so the next call would rewrite the entry it was just
            # served from.
            try:
                value = _copy.deepcopy(recorded[name])
                if name in cells:
                    _put_back_in_place(cells[name], value)
                elif isinstance(globals_dict, dict):
                    globals_dict[name] = value
            except Exception:  # noqa: BLE001 - a restore must never crash
                logger.debug("call unit: could not restore global %r", name)
