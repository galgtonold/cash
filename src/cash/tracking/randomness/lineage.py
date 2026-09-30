"""RNG state as lineage: virtual variables, seed epochs and seed cells.

A seed WRITES a hidden variable per module and a draw READS it, through the
ordinary variable-lineage machinery, so a re-seed re-keys every draw below it.
"""

from __future__ import annotations

import hashlib
import secrets
import sys
from collections.abc import Iterable, Mapping

from .detect import get_drawing_rng_modules, get_seeding_rng_modules

# -----------------------------------------------------------------------------
# Hidden lineage dependencies (RNG state as a lineage-tracked virtual variable)
# -----------------------------------------------------------------------------
#
# A *hidden dependency* is global mutable state a statement reads or writes
# without binding a user-visible name -- the RNG generator's state is the first
# instance (``os.environ``, ``plt.rcParams`` are the same shape). Modelled as a
# virtual variable that rides the ORDINARY variable-lineage machinery: a
# producer (a ``seed()``) writes its lineage, a consumer (a draw) reads it, and
# every downstream consumer folds that lineage into its own output lineage
# through the same input-lineage code every real variable uses. This is what
# makes a seed change propagate to a cached downstream statement -- the half the
# epoch (a key-only side channel) never did.
#
# The engines (runtime ``StatementProcessor`` and the ``VirtualLineage``
# simulator) integrate at a single seam each via these three helpers; there is
# no RNG-specific branching elsewhere. Adding a second hidden dependency later
# means teaching these three functions about it, nothing more.


def rng_virtual_var(module: str) -> str:
    """Name of the virtual lineage variable modelling *module*'s global RNG state.

    Deliberately not a valid attribute path and never written to ``user_ns`` --
    it exists only as a key in the lineage dict, so it cannot collide with a
    user variable or be mistaken for one.
    """
    return f"__cash_rng__{module}"


# ---------------------------------------------------------------------------
# Seed-epoch registry.
#
# The statement engine owns the ledger (a seed opens a new epoch for its module,
# valued by the seeding statement's cache key). It was reachable only from
# ``StatementProcessor``, so ``@cash.cache`` -- which consumes the same global
# stream -- had no way to see that the seed had changed, and served a model
# trained under the previous one.
#
# The processor PUBLISHES its dict here by reference, so this is a view onto the
# live ledger rather than a second copy that could drift. A fresh processor
# replaces the pointer, which also keeps tests isolated.
# ---------------------------------------------------------------------------
_ACTIVE_SEED_EPOCHS: dict[str, str] | None = None


def publish_seed_epochs(epochs: dict[str, str] | None) -> None:
    """Expose the statement engine's live seed ledger to other front-ends."""
    global _ACTIVE_SEED_EPOCHS
    _ACTIVE_SEED_EPOCHS = epochs


def seed_epochs() -> dict[str, str]:
    """Snapshot of the current seed epoch per RNG module (may be empty)."""
    return dict(_ACTIVE_SEED_EPOCHS or {})


# ---------------------------------------------------------------------------
# Seeds outside a notebook.
#
# Without a statement engine nobody published a ledger, so a seed set by the
# caller -- ``np.random.seed(0)`` at module level, or in an outer function
# before it calls a cached function that draws -- never reached that function's
# key: every seed was served the first seed's draw. ``watch_seeds`` wraps the
# global ``seed`` functions so the process knows which streams were seeded, and
# a drawing function is then keyed on where the seeded stream stands when it is
# called. That position is a pure function of the seed and the draws since, so
# it is the same in the next process and moves with every draw in between.
# ---------------------------------------------------------------------------
#: RNG modules a watched ``seed(<value>)`` seeded in this process.
_PROCESS_SEEDED: set[str] = set()
#: The seed functions wrapped so far, by module.
_WATCHED: set[str] = set()


def _note_seed(module: str, args: tuple, kwargs: dict) -> None:
    value = args[0] if args else kwargs.get("seed", kwargs.get("a"))
    if value is None:
        # ``seed()`` / ``seed(None)`` re-randomises: the stream is unseeded
        # again, and the freeze contract applies to it.
        _PROCESS_SEEDED.discard(module)
    else:
        _PROCESS_SEEDED.add(module)


def _seed_wrapper(module: str, original):
    def seed(*args, **kwargs):
        result = original(*args, **kwargs)
        _note_seed(module, args, kwargs)
        return result

    seed.__wrapped__ = original  # type: ignore[attr-defined]
    seed.__cash_seed_watch__ = True  # type: ignore[attr-defined]
    seed.__name__ = getattr(original, "__name__", "seed")
    seed.__doc__ = getattr(original, "__doc__", None)
    return seed


def watch_seeds() -> None:
    """Wrap ``random.seed`` and ``numpy.random.seed`` (once each, when imported).

    Cheap to call again: a module already wrapped is skipped. A seed set before
    the wrap -- before cash saw ``numpy`` imported -- is not seen.
    """
    if "numpy.random" not in _WATCHED and "numpy" in sys.modules and "numpy.random" not in sys.modules:
        # numpy loads `numpy.random` on first attribute access, which is the
        # user's `np.random.seed(...)` itself: load it now to watch that call.
        try:
            import numpy.random  # noqa: F401
        except Exception:  # noqa: BLE001 - a broken numpy only loses the watch
            pass
    targets = (("random", "random"), ("numpy.random", "numpy.random"))
    for module, rng in targets:
        if rng in _WATCHED:
            continue
        mod = sys.modules.get(module)
        original = getattr(mod, "seed", None) if mod is not None else None
        if original is None:
            continue
        if not getattr(original, "__cash_seed_watch__", False):
            try:
                mod.seed = _seed_wrapper(rng, original)
            except (AttributeError, TypeError):  # pragma: no cover - read-only module
                continue
        _WATCHED.add(rng)


def seeded_rng_modules() -> set[str]:
    """RNG modules seeded now: the notebook's ledger, or else this process's
    watched seeds."""
    if _ACTIVE_SEED_EPOCHS is not None:
        return set(_ACTIVE_SEED_EPOCHS)
    return set(_PROCESS_SEEDED)


def _stream_position(module: str) -> str | None:
    """Digest of *module*'s global stream as it stands, or None."""
    try:
        if module == "random":
            import random

            payload = repr(random.getstate()).encode("utf-8")
        elif module == "numpy.random":
            np_random = sys.modules.get("numpy.random")
            if np_random is None:
                return None
            name, keys, pos, has_gauss, cached = np_random.get_state()
            payload = b"%s|%d|%d|%r|" % (str(name).encode(), int(pos), int(has_gauss), float(cached))
            payload += keys.tobytes()
        else:
            return None
    except Exception:  # noqa: BLE001 - an unreadable stream keeps the plain key
        return None
    return hashlib.sha256(payload).hexdigest()[:32]


def seed_epoch_component(modules: set[str]) -> str:
    """Cache-key fragment for *modules*, or "" when none of them is seeded.

    Empty when unseeded, so a function that draws from an unseeded stream keeps
    the key it has today: the freeze contract still applies and the value is
    replayed. Only an actual seed -- or a change to one -- moves the key.
    Outside a notebook, a stream seeded in this process is keyed by its
    position at the call (see `watch_seeds`).
    """
    if not modules:
        return ""
    if _ACTIVE_SEED_EPOCHS is None:
        parts = []
        for m in sorted(modules & _PROCESS_SEEDED):
            position = _stream_position(m)
            if position is not None:
                parts.append(f"{m}@{position}")
        return ":rng:" + ":".join(parts) if parts else ""
    epochs = _ACTIVE_SEED_EPOCHS
    parts = [f"{m}:{epochs[m]}" for m in sorted(modules) if m in epochs]
    return ":rng:" + ":".join(parts) if parts else ""


def observed_rng_reads(tracking_state, code: str) -> set[str]:
    """Virtual RNG variables *code* reads by prior runtime OBSERVATION.

    ``model.fit()`` consumes the global stream while its AST says nothing, so
    the runtime records what it saw and every engine reads that one ledger.

    Deliberately shared by all three seams -- the runtime cache key, the runtime
    output lineage, and the simulation. They must agree byte-for-byte (the
    cache-key unification rule); when only the first one had it, the simulation
    computed different keys and 65 integration tests failed. One function, so
    they cannot drift apart again.
    """
    ledger = getattr(tracking_state, "observed_rng_statement_draws", None)
    if not ledger or not code:
        return set()
    try:
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    except (AttributeError, UnicodeEncodeError):
        return set()
    modules = ledger.get(digest)
    return {rng_virtual_var(m) for m in modules} if modules else set()


def hidden_lineage_reads(code: str) -> set[str]:
    """Virtual variables a statement READS: the RNG state a draw consumes.

    Folded into the statement's cache key AND its output lineage, so a re-seed
    upstream both re-keys the draw and propagates to everything cached
    downstream of it.
    """
    return {rng_virtual_var(m) for m in get_drawing_rng_modules(code)}


def hidden_lineage_writes(code: str) -> set[str]:
    """Virtual variables a statement PRODUCES: a ``seed()`` defines its RNG state."""
    return {rng_virtual_var(m) for m in get_seeding_rng_modules(code)}


def entropy_write_lineage() -> str:
    """Lineage for a stream the user explicitly re-randomised.

    ``seed(None)`` states that this run should differ from the last, so the
    virtual variable must NOT take a value derived from the statement's cache
    key -- that key is byte-identical run to run, which is exactly what let
    downstream consumers keep hitting across a genuinely new stream.

    Fresh per call, so every consumer below the reseed recomputes. The
    simulation cannot predict this value, and that is the correct outcome: it
    sees a changed input and re-runs, which is the safe direction.
    """
    return "entropy:" + secrets.token_hex(16)


def hidden_write_lineage(producing_key: str) -> str:
    """Lineage a hidden variable takes when a statement produces it.

    A ``seed()`` RESETS the generator, so the new lineage depends only on the
    seeding statement's own cache key (which already folds in its source and its
    input lineage -- so ``seed(0)`` -> ``seed(1)`` and ``seed(cfg.seed)`` are
    both caught). Overwrite, not chain: editing an *earlier* seed must not
    invalidate a draw governed by a *later* one.
    """
    return hashlib.sha256(producing_key.encode("utf-8")).hexdigest()


def seed_cells_not_yet_run(
    drawing_modules: set[str],
    notebook_cells: "list[str]",
    executed_cell_hashes: set[str],
) -> "list[tuple[str, int]]":
    """Notebook cells that SEED a drawn module but whose source has not run.

    The detection core for the bare-``seed()`` edit-without-rerun defect.
    Editing an ``np.random.seed(N)`` cell and running only a downstream draw
    leaves the draw on the *old* seed's state, silently — because a bare seed
    binds no variable, so nothing links the draw back to the seed cell.

    This finds the mismatch from the one place it is visible: the notebook. A
    cell whose source SEEDS a module that some draw consumes, but whose exact
    source hash is NOT among the cells that have executed this session, is an
    edited-but-not-rerun (or never-run) seed the draw will ignore.

    Pure and cell-granular — it matches the checker's view of the notebook and
    takes the executed-hash set as an argument, so it carries no state and is
    trivially testable. The upstream checker re-runs such a seed cell ahead
    of the draw, since a seed's effect is replayed, never restored.

    Parameters
    ----------
    drawing_modules : the RNG modules the current cell draws from.
    notebook_cells  : the notebook's cell sources, in order.
    executed_cell_hashes : ``sha256(source).hexdigest()`` for every cell source
        that has actually executed this session.

    Returns ``[(module, cell_index)]``, sorted, one per (unrun seed cell, module).
    """
    if not drawing_modules:
        return []
    out: list[tuple[str, int]] = []
    for idx, src in enumerate(notebook_cells):
        seeded = get_seeding_rng_modules(src) & drawing_modules
        if not seeded:
            continue
        digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
        if digest in executed_cell_hashes:
            continue  # this exact seed cell source has run — not stale
        for module in sorted(seeded):
            out.append((module, idx))
    return out


def rng_lineage_fingerprint(
    variable_lineage: Mapping[str, str],
    modules: Iterable[str],
) -> tuple:
    """The seeds in force for *modules* — an invalidation key for a saved position.

    A recorded RNG start position stays valid only while the seed that
    determines it is unchanged. Rather than invent a staleness rule for that,
    reuse the one every real variable already uses: the RNG state is modelled as
    a hidden lineage variable per module (``__cash_rng__<module>``), written by a
    seeding statement and read by a draw, so comparing that lineage now against
    the lineage recorded alongside the position IS the standard check.

    An UNSEEDED module contributes ``None`` on both sides and therefore stays
    valid forever. That is deliberate: an unseeded stream has no semantically
    correct position, so cash freezes the one it first saw — the same
    "non-determinism frozen, not blocked" rule the rest of the system follows.
    """
    return tuple(sorted((m, variable_lineage.get(rng_virtual_var(m))) for m in modules))
