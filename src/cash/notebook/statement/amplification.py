"""The loop-persist amplification guard.

``# @cash:persist`` inside (or on) a loop makes EVERY iteration a persist
target. When the loop grows one object -- the classic "add a column per
iteration" frame build -- each iteration snapshots the whole object at its
current width, so a 40 MB final frame costs sum(widths) on disk: 13x for 25
columns, and quadratic in the iteration count thereafter. The tier caps are
structurally blind to this: its per-object refusal compares ONE value against
half the tier cap (40 MB vs >=4 GiB -> fine) and its evict-after-write warning
needs the total to exceed the cap (520 MB vs >=8 GiB -> never evicts). Neither
looks at *cumulative writes for one statement*, which is the dimension that
actually blows up.

So track that dimension directly. Once one statement's cumulative persisted
bytes exceed both an absolute floor and a multiple of the value's CURRENT
size, stop value-persisting it (metadata-only, exactly like the size-aware
skip) and warn once. Skipping beats evict-after-write here: it also stops
paying the rising per-iteration serialisation cost, which is the "re-runs got
slower" half of the symptom.

The floor keeps the guard off small loops entirely (nobody's disk is at risk
from a few MB), and the accounting is only ever done for statements carrying
an iteration/branch context marker, so an ordinary single-statement
``# @cash:persist`` can never trip it -- it writes once, and is not in a loop.
"""

from __future__ import annotations

from typing import Any

from cash._memo import STATEMENTS, LruMemo
from cash.backends.adaptive_caps import human_bytes
from cash.control_markers import has_marker, strip_markers
from cash.diagnostics import warn_diagnostic
from cash.exceptions import CashCacheIneffectiveWarning

__all__ = ["AMPLIFICATION_SKIP_REASON", "AmplificationGuard"]

_PERSIST_AMPLIFICATION_FLOOR_BYTES = 64 * 1024 * 1024
_PERSIST_AMPLIFICATION_LIMIT = 4
# Badge/metadata reason for a write refused by the guard above. A constant so
# consumers compare identity rather than pattern-matching the wording.
AMPLIFICATION_SKIP_REASON = (
    "loop caching a growing object would store every intermediate state; "
    "further iterations kept metadata-only (see the emitted warning)"
)


class AmplificationGuard:
    """Per-statement accounting of what a looped persist has written.

    Keyed on the loop body's real source with the per-iteration marker
    stripped, so every iteration of one statement shares a counter; a
    statement warns once, not once per iteration.
    """

    def __init__(self) -> None:
        self._bytes_by_stmt: LruMemo[str, int] = LruMemo(STATEMENTS)
        self._last_size_by_stmt: LruMemo[str, int] = LruMemo(STATEMENTS)
        self._warned: LruMemo[str, bool] = LruMemo(STATEMENTS)

    @staticmethod
    def _size(prediction: dict[str, Any] | None) -> int:
        """Size of the largest output var, or 0 when it isn't usable.

        Reads the estimate the store's size-aware skip already
        computed, so the guard adds no sizing work to the write path.
        """
        if prediction is None:
            return 0
        try:
            size = int(prediction.get("size_bytes") or 0)
        except (TypeError, ValueError):
            return 0
        return max(size, 0)

    def _stmt_id(self, code: str) -> str | None:
        """Per-statement accounting key, or ``None`` if it cannot amplify.

        Only a statement replayed under a control structure writes more than
        once per run, so only those are accounted. Stripping the per-iteration
        discriminator comment makes every iteration of one loop body share a
        counter; an ordinary ``# @cash:persist`` on a single statement has no
        marker, gets ``None`` here, and is untouched by the whole mechanism.
        """
        if not has_marker(code):
            return None
        return strip_markers(code).strip()

    def check(
        self,
        code: str,
        prediction: dict[str, Any] | None,
        annotated: bool = False,
    ) -> tuple[bool, str | None]:
        """Return ``(skip, reason)`` for the loop-persist guard.

        Consulted immediately before a value-persist: refuse once this
        statement's cumulative *durably stored* bytes are out of all proportion
        to the value being stored. The counter is fed by
        :meth:`account` after the write actually lands.

        Two thresholds must BOTH be crossed, which is what keeps the guard off
        healthy notebooks: an absolute floor
        (``_PERSIST_AMPLIFICATION_FLOOR_BYTES``), so small loops never engage at
        all, and a ratio against the current value, so a loop that legitimately
        stores a lot of *distinct* results is judged on proportion rather than
        volume.

        The verdict LATCHES per statement: once a statement has demonstrated
        amplification, later iterations stay metadata-only. Without the latch the
        guard would disengage exactly when it matters -- the running total
        freezes while the object keeps growing, so ``LIMIT x size`` would
        eventually overtake it and the writes would resume mid-loop.
        """
        size = self._size(prediction)
        if size <= 0:
            return False, None
        stmt_id = self._stmt_id(code)
        if stmt_id is None:
            return False, None

        if stmt_id in self._warned:
            return True, AMPLIFICATION_SKIP_REASON

        cumulative = self._bytes_by_stmt.get(stmt_id, 0)
        if cumulative > _PERSIST_AMPLIFICATION_FLOOR_BYTES and cumulative > _PERSIST_AMPLIFICATION_LIMIT * size:
            self._warned[stmt_id] = True
            self._warn(stmt_id, cumulative, size, annotated=annotated)
            return True, AMPLIFICATION_SKIP_REASON
        return False, None

    def account(
        self,
        code: str,
        prediction: dict[str, Any] | None,
        wire: dict[str, Any],
    ) -> None:
        """Add a completed write to its statement's running total.

        Counts a write only when it reached a **persistent** tier. The backend
        reports the resolved destinations back on the metadata dict, so this is
        a read of information the write already produced.

        Excluding RAM-only writes is what makes the guard track the resource the
        user is actually losing. A loop body that misses the promotion floor is
        cached in RAM and never touches the disk at all; counting those would
        warn about "filling your disk" for a notebook whose disk cache is a few
        KB, which is both false and noisy.
        """
        stmt_id = self._stmt_id(code)
        if stmt_id is None:
            return
        size = self._size(prediction)
        if size <= 0:
            return
        destinations = wire.get("storage") or ()
        if not isinstance(destinations, (list, tuple)):
            return
        if not any(d != "RAM" for d in destinations):
            return
        # Growth only: a loop that REBINDS a same-sized value each pass stores a
        # different result every time, and nothing in it grows. Summing those
        # warned about a sweep's per-window dict.
        last = self._last_size_by_stmt.get(stmt_id)
        self._last_size_by_stmt[stmt_id] = size
        if last is not None and size <= last:
            return
        self._bytes_by_stmt[stmt_id] = self._bytes_by_stmt.get(stmt_id, 0) + size

    def _warn(
        self,
        stmt_id: str,
        cumulative: int,
        size: int,
        annotated: bool = False,
    ) -> None:
        """Warn once that a looped persist is snapshotting a growing object.

        Names the amplification in the user's own terms -- what it has already
        written versus how big the value actually is -- and points at the fix,
        which is to persist the finished object once instead of every
        intermediate state of it.
        """

        first_line = (stmt_id.splitlines() or [""])[0].strip()
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-LOOP-GROWTH",
            f"`{first_line}` runs in a loop and has already cached "
            f"{human_bytes(cumulative)} of intermediate snapshots for a value "
            f"that is currently only {human_bytes(size)} -- caching a growing "
            f"object every iteration costs the SUM of every intermediate size, "
            f"not the final one. Further iterations are not being stored.",
            # The annotation advice only for a statement that carries it: a user
            # was told to move a `# @cash:persist` they never wrote.
            (
                "move `# @cash:persist` off the loop and onto a statement that "
                "produces the finished object, so it is stored once."
                if annotated
                else "build the finished object in one statement -- a comprehension, "
                "or a function the loop's work moves into -- so it is stored once; "
                "calls inside the loop are still cached."
            ),
        )
