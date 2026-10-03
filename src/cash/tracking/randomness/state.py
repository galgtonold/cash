"""Capturing and restoring RNG state, so a cache hit leaves the stream where
the computation did.

Two channels: the module globals (``random``, ``numpy.random``, ``torch``) and
per-object carriers a statement reads (``rng = np.random.default_rng()``).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import random
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

from .detect import KIND_NP_GENERATOR, KIND_NP_RANDOMSTATE, KIND_PY_RANDOM

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# =============================================================================
# RNG State Capture and Restore
# =============================================================================


def capture_rng_state() -> dict:
    """
    Capture the current state of the global RNGs cash can restore.

    Those are ``random``, ``numpy.random`` and ``torch`` (plus ``torch.cuda``);
    TensorFlow has no readable state, see below.

    Returns:
        Dict mapping module name to its RNG state (picklable).
        Only includes modules that are currently imported.
    """

    state = {}

    # Standard library random
    if "random" in sys.modules:
        try:
            state["random"] = random.getstate()
        except (TypeError, AttributeError) as e:
            logger.debug("[RANDOMNESS] Failed to capture random state: %s", e)

    # NumPy random
    if "numpy" in sys.modules or "numpy.random" in sys.modules:
        try:
            import numpy as np

            state["numpy.random"] = np.random.get_state()
        except (ImportError, AttributeError) as e:
            logger.debug("[RANDOMNESS] Failed to capture numpy random state: %s", e)

    # PyTorch (if available)
    if "torch" in sys.modules:
        try:
            import torch

            state["torch"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["torch.cuda"] = torch.cuda.get_rng_state_all()
        except (ImportError, RuntimeError) as e:
            logger.debug("[RANDOMNESS] Failed to capture torch random state: %s", e)

    # TensorFlow is detected (``RANDOM_FUNCTIONS``, ``SEED_FUNCTIONS``) but not
    # captured, by design: ``tf.random.uniform`` and the other op-level draws
    # keep their state inside the ops, and TF has no API to read it back or set
    # it. A hit on a TF draw therefore leaves TF's stream where it was. Stated
    # in known-limitations ("TensorFlow draws are flagged but not replayed").

    return state


def restore_rng_state(state: dict, displaced: dict | None = None) -> None:
    """
    Restore RNG state from a previously captured state dict.

    Args:
        state: Dict mapping module name to its RNG state.
        displaced: When given, receives the state each restored module is
            moved away from, unless it already holds one for that module, so
            it keeps the earliest position a run of restores moved away from.
    """

    if not state:
        return

    if displaced is not None and any(module not in displaced for module in state):
        current = capture_rng_state()
        for module in state:
            if module in current:
                displaced.setdefault(module, current[module])

    # Standard library random
    if "random" in state and "random" in sys.modules:
        try:
            random.setstate(state["random"])
        except (TypeError, ValueError) as e:
            logger.debug("[RANDOMNESS] Failed to restore random state: %s", e)

    # NumPy random
    if "numpy.random" in state and ("numpy" in sys.modules or "numpy.random" in sys.modules):
        try:
            import numpy as np

            np.random.set_state(state["numpy.random"])
        except (ImportError, TypeError, ValueError) as e:
            logger.debug("[RANDOMNESS] Failed to restore numpy random state: %s", e)

    # PyTorch
    if "torch" in state and "torch" in sys.modules:
        try:
            import torch

            torch.set_rng_state(state["torch"])
            if "torch.cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch.cuda"])
        except (ImportError, RuntimeError) as e:
            logger.debug("[RANDOMNESS] Failed to restore torch random state: %s", e)


# -----------------------------------------------------------------------------
# Per-object RNG carriers
# -----------------------------------------------------------------------------
#
# ``capture_rng_state`` / ``restore_rng_state`` above cover the RNG *module
# globals* (``random``, ``np.random``, ``torch``).  They cannot see a generator
# a user holds in a variable — ``rng = np.random.default_rng(42)`` — because
# that object has no module-level home.
#
# The asymmetry that motivates this: a statement drawing from the
# global channel (``np.random.randint``) replays correctly across a cache hit,
# because the post-state is captured and re-injected.  A statement drawing from
# an object-held generator (``rng.integers``) HITS and restores its output, but
# the live generator is never advanced — so the next draw repeats the values
# the cached statement already consumed.
#
# The fix mirrors the global channel exactly: capture the carrier's post-state
# at store time and inject it back on a hit.  Capture is scoped to the
# statement's declared inputs (not all of ``user_ns``) and gated on an
# ``isinstance`` allowlist, which bounds the cost to the handful of variables a
# statement actually reads.

# The carrier kinds (``KIND_*``) and the source-level constructor tables live
# in :mod:`.detect` — the detector needs them too, and both channels must agree
# on what an RNG is.  See the "RNG carriers" section there.


def rng_carrier_kind(obj: object) -> str | None:
    """Return the carrier kind for ``obj``, or ``None`` if it isn't one.

    Objects owned by the RNG *module globals* are deliberately excluded: the
    module channel in :func:`capture_rng_state` already replays those, and
    capturing them twice under a variable name would let a stale alias fight
    with the authoritative global state.
    """

    if "numpy" in sys.modules or "numpy.random" in sys.modules:
        try:
            import numpy as np

            if isinstance(obj, np.random.Generator):
                return KIND_NP_GENERATOR
            if isinstance(obj, np.random.RandomState):
                # ``np.random.*`` module functions delegate to this singleton;
                # the global channel owns it.
                if obj is not np.random.mtrand._rand:
                    return KIND_NP_RANDOMSTATE
                return None
        except (ImportError, AttributeError):
            pass

    if isinstance(obj, random.Random):
        # ``random.*`` module functions delegate to this singleton.
        if obj is not getattr(random, "_inst", None):
            return KIND_PY_RANDOM
        return None

    return None


def capture_object_rng_states(
    names: "Iterable[str]",
    user_ns: dict[str, object],
) -> dict[str, dict]:
    """Capture the post-state of any RNG carrier bound to one of ``names``.

    Args:
        names: Variable names to consider — the statement's inputs.  Scoping to
            inputs is what bounds the cost: no full ``user_ns`` walk.
        user_ns: The shell namespace to resolve names against.

    Returns:
        ``{var_name: {'kind': <carrier kind>, 'state': <picklable state>}}``.
        Empty when the statement reads no RNG carriers, which is the common
        case — callers should omit the payload key entirely when empty so
        non-RNG statements keep their existing payload shape.
    """
    states: dict[str, dict] = {}

    for name in names:
        try:
            obj = user_ns.get(name)
        except (TypeError, AttributeError):
            continue
        if obj is None:
            continue

        try:
            kind = rng_carrier_kind(obj)
            if kind is None:
                continue
            if kind == KIND_NP_GENERATOR:
                # Foreign / third-party bit generators may raise or hand back
                # something unpicklable here; the except below drops them.
                state = obj.bit_generator.state
            elif kind == KIND_NP_RANDOMSTATE:
                state = obj.get_state()
            else:
                state = obj.getstate()
        except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
            # e.g. random.SystemRandom.getstate() raises NotImplementedError.
            logger.debug("[RANDOMNESS] Failed to capture RNG state for %r: %s", name, e)
            continue

        states[name] = {"kind": kind, "state": state}

    return states


def restore_object_rng_states(
    states: dict[str, dict] | None,
    user_ns: dict[str, object],
) -> None:
    """Inject captured per-object RNG states back onto the live carriers.

    Guarded on presence (the name still resolves) and on type match (the live
    object is still the same kind of carrier).  A name that now holds something
    else is skipped rather than forced.

    Args:
        states: Mapping produced by :func:`capture_object_rng_states`;
            ``None`` or empty when nothing was captured, which restores
            nothing.
        user_ns: The shell namespace to resolve names against.
    """
    if not states:
        return

    for name, entry in states.items():
        try:
            kind = entry["kind"]
            state = entry["state"]
        except (TypeError, KeyError):
            continue

        obj = user_ns.get(name)
        if obj is None:
            continue

        # Type match: only write the state back onto the same carrier kind.
        if rng_carrier_kind(obj) != kind:
            continue

        try:
            if kind == KIND_NP_GENERATOR:
                obj.bit_generator.state = state
            elif kind == KIND_NP_RANDOMSTATE:
                obj.set_state(state)
            else:
                obj.setstate(state)
        except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
            logger.debug("[RANDOMNESS] Failed to restore RNG state for %r: %s", name, e)


def _carrier_state(obj: object, kind: str) -> object:
    if kind == KIND_NP_GENERATOR:
        return obj.bit_generator.state
    if kind == KIND_NP_RANDOMSTATE:
        return obj.get_state()
    return obj.getstate()


def capture_reachable_carrier_states(fn: object) -> list[tuple[object, object, tuple[str, str] | None]]:
    """The live RNG carriers *fn* can reach, each with its current state.

    A helper that draws from a generator held in a global (``rng`` built in a
    cell, then ``rng.integers(...)`` inside ``boot(x)``) moves that
    generator's stream, and :func:`capture_rng_state` -- the module channel --
    cannot see it. Followed through the names *fn*'s code reads (nested code
    too), its closure, and every user function those reach, so a draw two
    helpers down is found as well.

    Each entry is ``(carrier, state, where)``: *where* is ``(module, name)``
    for a carrier bound to a module global, which a later process can find
    again, and ``None`` for one only a closure holds. Compare with
    :func:`moved_carriers`.
    """
    found: dict[int, tuple[object, object, tuple[str, str] | None]] = {}
    seen_fns: set[int] = set()
    stack = [fn]
    while stack:
        f = stack.pop()
        f = getattr(f, "__func__", f)  # a bound method's function
        code = getattr(f, "__code__", None)
        if code is None or id(f) in seen_fns:
            continue
        seen_fns.add(id(f))
        f_globals = getattr(f, "__globals__", None) or {}
        module = f_globals.get("__name__")
        named: list[tuple[object, tuple[str, str] | None]] = [
            (cell.cell_contents, None) for cell in (getattr(f, "__closure__", None) or ()) if _cell_filled(cell)
        ]
        codes = [code]
        while codes:
            c = codes.pop()
            named.extend(
                (f_globals[name], (module, name) if isinstance(module, str) else None)
                for name in c.co_names
                if name in f_globals
            )
            codes.extend(const for const in c.co_consts if hasattr(const, "co_names"))
        for value, where in named:
            kind = rng_carrier_kind(value)
            if kind is not None:
                if id(value) not in found or found[id(value)][2] is None:
                    try:
                        state = _carrier_state(value, kind)
                    except (TypeError, ValueError, AttributeError, NotImplementedError):
                        # A carrier whose state cannot be read cannot be shown
                        # unmoved either: record it as always moved.
                        state = _UNREADABLE
                    found[id(value)] = (value, state, where)
            elif _is_user_function(value):
                stack.append(value)
    return list(found.values())


def moved_carriers(
    before: list[tuple[object, object, tuple[str, str] | None]],
) -> list[tuple[tuple[str, str] | None, object, object]]:
    """The carriers from :func:`capture_reachable_carrier_states` whose stream
    moved since, as ``(where, state before, state now)``."""
    moved = []
    for obj, state, where in before:
        try:
            after = _carrier_state(obj, rng_carrier_kind(obj))
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            moved.append((where, state, _UNREADABLE))
            continue
        if state is _UNREADABLE or not _rng_states_equal(_flatten_state(state), _flatten_state(after)):
            moved.append((where, state, after))
    return moved


def replayable(moved: list[tuple[tuple[str, str] | None, object, object]]) -> bool:
    """Whether a later process can put every one of *moved* back where the
    call left it: each bound to a module global, both states readable."""
    return all(where is not None and _UNREADABLE not in (pre, post) for where, pre, post in moved)


def carrier_states_changed(before: list[tuple[object, object, tuple[str, str] | None]]) -> bool:
    """Whether any carrier from :func:`capture_reachable_carrier_states` moved."""
    return bool(moved_carriers(before))


def _set_carrier_state(obj: object, kind: str, state: object) -> None:
    if kind == KIND_NP_GENERATOR:
        obj.bit_generator.state = state
    elif kind == KIND_NP_RANDOMSTATE:
        obj.set_state(state)
    else:
        obj.setstate(state)


def capture_argument_carrier_states(named: list[tuple[str, object]]) -> list[tuple[object, object, tuple[str, str]]]:
    """The RNG carriers among a call's arguments, each with its current state.

    *named* is ``(parameter, value)`` for each argument. An entry's *where* is
    ``("arg", parameter)``: a later call finds the generator again among its
    own arguments. Compare with :func:`moved_carriers`.
    """
    found: dict[int, tuple[object, object, tuple[str, str]]] = {}
    for name, value in named:
        kind = rng_carrier_kind(value)
        if kind is None or id(value) in found:
            continue
        try:
            state = _carrier_state(value, kind)
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            state = _UNREADABLE
        found[id(value)] = (value, state, ("arg", name))
    return list(found.values())


def carrier_positions(names: "Iterable[str]", user_ns: dict[str, object]) -> dict[str, tuple[object, object]]:
    """``{name: (generator, its state)}`` for each of *names* bound to an RNG
    carrier in *user_ns*. Compare with :func:`moved_carrier_names`.

    Keyed by name, not by object: two names for one generator are both
    reported when it moves. A state that cannot be read is recorded as
    unreadable, which :func:`moved_carrier_names` reports as moved.
    """
    positions: dict[str, tuple[object, object]] = {}
    for name in names:
        try:
            obj = user_ns.get(name)
        except (TypeError, AttributeError):
            continue
        kind = rng_carrier_kind(obj)
        if kind is None:
            continue
        try:
            state = _carrier_state(obj, kind)
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            state = _UNREADABLE
        positions[name] = (obj, state)
    return positions


def moved_carrier_names(positions: dict[str, tuple[object, object]], user_ns: dict[str, object]) -> set[str]:
    """The names from :func:`carrier_positions` whose generator moved since,
    and that still hold that same generator (a rebound name is an output of
    the statement, with a lineage of its own)."""
    moved: set[str] = set()
    for name, (obj, before) in positions.items():
        if user_ns.get(name) is not obj:
            continue
        if before is _UNREADABLE:
            moved.add(name)
            continue
        try:
            after = _carrier_state(obj, rng_carrier_kind(obj))
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            moved.add(name)
            continue
        if not _rng_states_equal(_flatten_state(before), _flatten_state(after)):
            moved.add(name)
    return moved


@contextlib.contextmanager
def carriers_put_back(before: list[tuple[object, object, object]]) -> Iterator[None]:
    """Inside the block, each carrier of *before* stands where it was then;
    afterwards, where it is now. For a check that must not see a draw the
    caller will see replayed (the argument-mutation check)."""
    now = []
    for obj, state, _where in before:
        kind = rng_carrier_kind(obj)
        if kind is None or state is _UNREADABLE:
            continue
        try:
            now.append((obj, kind, _carrier_state(obj, kind)))
            _set_carrier_state(obj, kind, state)
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            continue
    try:
        yield
    finally:
        for obj, kind, state in now:
            _set_carrier_state(obj, kind, state)


def replay_argument_carriers(moved: list, named: dict[str, object]) -> None:
    """:func:`replay_carriers` for generators handed in as arguments: *named*
    maps each parameter of the hit's call to its value."""
    for where, pre, post in moved:
        try:
            _scope, name = where
            obj = named.get(name)
            kind = rng_carrier_kind(obj)
            if kind is None:
                continue
            if not _rng_states_equal(_flatten_state(_carrier_state(obj, kind)), _flatten_state(pre)):
                continue
            _set_carrier_state(obj, kind, post)
        except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
            logger.debug("[RANDOMNESS] Failed to replay RNG state for %r: %s", where, e)


def replay_carriers(moved: list) -> None:
    """Move each recorded carrier to where the computed call left it -- only
    while it is where that call found it, as the module channel does.

    *moved* is :func:`moved_carriers`' output, every *where* set.
    """
    for where, pre, post in moved:
        try:
            module, name = where
            obj = getattr(sys.modules.get(module), "__dict__", {}).get(name)
            kind = rng_carrier_kind(obj)
            if kind is None:
                continue
            if not _rng_states_equal(_flatten_state(_carrier_state(obj, kind)), _flatten_state(pre)):
                continue
            _set_carrier_state(obj, kind, post)
        except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
            logger.debug("[RANDOMNESS] Failed to replay RNG state for %r: %s", where, e)


_UNREADABLE = object()


def _is_user_function(value: object) -> bool:
    """A function the user wrote: walked into. Library code is not, both for
    cost and because a library's own generators are its business."""
    from ...install_paths import is_user_code_file

    code = getattr(getattr(value, "__func__", value), "__code__", None)
    return code is not None and is_user_code_file(code.co_filename)


def _cell_filled(cell) -> bool:
    try:
        cell.cell_contents
    except ValueError:  # an empty cell: the closure variable is not bound yet
        return False
    return True


def _flatten_state(state: object) -> object:
    """A numpy Generator's state is a nested dict; compare it as a tuple."""
    if isinstance(state, dict):
        return tuple((k, _flatten_state(v)) for k, v in sorted(state.items()))
    return state


def rng_modules_changed(before: dict, after: dict) -> set[str]:
    """Modules whose captured RNG state differs between *before* and *after*.

    Used to observe a draw that static analysis cannot see because it happens
    inside a called function. Compares the per-module states from two
    :func:`capture_rng_state` snapshots. A module present in only one snapshot
    (e.g. numpy imported mid-cell) counts as changed. Arrays are compared by
    bytes, never ``repr``, so display options cannot mask a difference.
    """
    changed: set[str] = set()
    for module in set(before) | set(after):
        b, a = before.get(module), after.get(module)
        if b is None or a is None:
            changed.add(module)
        elif not _rng_states_equal(b, a):
            changed.add(module)
    return changed


def _rng_states_equal(before: object, after: object) -> bool:
    """True when two captured per-module RNG states are identical.

    Compares the states DIRECTLY rather than digesting both sides. This is on the
    hot path -- the observer runs on every statement -- and digesting dominated
    it: the stdlib state is MT19937's 624 state words plus a position, carried as
    a 625-element tuple of Python ints, and feeding those ints one at a time into
    a hash is far slower than one tuple comparison.

    Arrays are still compared by BYTES, never ``repr``, so display truncation can
    never mask a difference -- and torch tensors now go by bytes too (via
    ``.numpy()``), where the digest fallback would have used a TRUNCATED repr and
    could have called two different states equal.
    """
    if before is after:
        return True
    b_tobytes, a_tobytes = getattr(before, "tobytes", None), getattr(after, "tobytes", None)
    if callable(b_tobytes) and callable(a_tobytes):
        return b_tobytes() == a_tobytes()  # numpy ndarray
    if isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)):
        if len(before) != len(after):
            return False
        try:
            # Fast path: an all-scalar sequence (the stdlib's 625 ints) compares
            # at C speed in one call. Raises when an element is an ndarray or
            # tensor -- whose ``==`` yields an array, making the tuple compare's
            # truthiness check ambiguous -- so those fall through to elementwise.
            return bool(before == after)
        except (ValueError, RuntimeError):
            return all(_rng_states_equal(x, y) for x, y in zip(before, after))
    if isinstance(before, (int, float, complex, str, bytes, bool, type(None))):
        return type(before) is type(after) and before == after
    b_np, a_np = getattr(before, "numpy", None), getattr(after, "numpy", None)
    if callable(b_np) and callable(a_np):
        try:
            return b_np().tobytes() == a_np().tobytes()  # torch tensor
        except (TypeError, ValueError, RuntimeError):
            pass
    # Anything exotic: fall back to the byte digest rather than risk a
    # container's ambiguous ``==`` (an ndarray/tensor compare returns an array).
    try:
        return _digest_rng_state(before) == _digest_rng_state(after)
    except (TypeError, ValueError, RuntimeError):
        return False


def _digest_rng_state(value: object) -> str:
    """Byte-digest of one module's RNG state (arrays by bytes, not repr)."""
    h = hashlib.sha256()

    def feed(item: object) -> None:
        tobytes = getattr(item, "tobytes", None)
        if callable(tobytes):
            h.update(tobytes())
        elif isinstance(item, (tuple, list)):
            for sub in item:
                feed(sub)
        else:
            h.update(repr(item).encode("utf-8"))

    feed(value)
    return h.hexdigest()
