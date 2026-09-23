"""Capturing and restoring RNG state, so a cache hit leaves the stream where
the computation did.

Two channels: the module globals (``random``, ``numpy.random``, ``torch``) and
per-object carriers a statement reads (``rng = np.random.default_rng()``).
"""

from __future__ import annotations

import hashlib
import logging
import random
import sys
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


def restore_rng_state(state: dict) -> None:
    """
    Restore RNG state from a previously captured state dict.

    Args:
        state: Dict mapping module name to its RNG state.
    """

    if not state:
        return

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
        states: Mapping produced by :func:`capture_object_rng_states`.  Cache
            entries written before this field existed pass ``None`` here and
            restore unchanged.
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
    sha256 cost ~155us per comparison versus ~3.5us for numpy's ndarray (one
    ``tobytes``). Direct comparison does the same job ~100x faster (measured
    484us -> 4.6us for a two-module snapshot pair).

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
