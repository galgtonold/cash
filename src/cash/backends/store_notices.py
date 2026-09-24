"""What a tiered store tells the user about values it kept only in RAM.

Two notices, each deduplicated so a loop does not repeat it on every
iteration: CACHE-VALUE-TOO-BIG, once per session, and CACHE-NOT-WORTH-BYTES,
once per statement. The second can be held (`StoreNotices.hold`) and said as
one notice for everything refused meanwhile -- a notebook holds it for a cell.
"""

from __future__ import annotations

from ..config import human_bytes
from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning
from .value_policy import WORTH_CEILING_BYTES_PER_SECOND

__all__ = ["StoreNotices"]


def _cap_list(caps: list[int] | None) -> str:
    """ " (cap: 512.0 MiB)" / " (caps: ...)" / "" when unknown."""
    if not caps:
        return ""
    rendered = ", ".join(human_bytes(c) for c in caps)
    return f" (cap: {rendered})" if len(caps) == 1 else f" (caps: {rendered})"


class StoreNotices:
    """The notices of one tiered backend, with their dedup state."""

    #: Statements a combined refusal names; the rest are counted.
    NAMED = 5

    def __init__(self) -> None:
        self._warned_too_big = False
        #: Statements already told CACHE-NOT-WORTH-BYTES this session.
        self._warned_not_worth: set[str] = set()
        #: Refusals held for one notice (`hold`); None when not holding.
        self._held: list[tuple[str, int, float]] | None = None

    def hold(self) -> None:
        """Hold CACHE-NOT-WORTH-BYTES refusals until `release`, to say them once."""
        self._held = []

    def release(self) -> None:
        """Say the refusals held since `hold`, as one notice."""
        held, self._held = self._held, None
        if held:
            self._say_not_worth(held)

    def too_big(self, key: str, size_bytes: int, caps: list[int] | None = None) -> None:
        """A value worth persisting fit no persistent tier's whole cap.

        *size_bytes* is the serialized size, the number `cash inspect` shows,
        and *caps* the caps it was compared with, so the message names both.
        The RAM tier is only offered the value: its own cap is usually smaller
        than the disk tier's, so the usual outcome is no caching at all. Keep
        this and docs/warnings.md#cache-value-too-big saying the same thing.
        """
        if self._warned_too_big:
            return
        self._warned_too_big = True
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-VALUE-TOO-BIG",
            f"cached value {key!r} is {human_bytes(size_bytes)} serialized, "
            f"which is more than every persistent cache tier's whole cap"
            f"{_cap_list(caps)}, so only the RAM tier was offered it -- it will "
            f"not survive a kernel restart, and if it is over the RAM tier's "
            f"own cap too it is evicted at once and nothing is cached.",
            f"raise max_cache_size above {human_bytes(size_bytes)} (a "
            f"comfortable multiple of it, so the cache can hold more than this "
            f"one entry), or cache something smaller -- the aggregate, the "
            f"sample, or the columns you actually use. The size named here is "
            f"the serialized one, the same number `cash inspect` reports.",
        )

    def not_worth_bytes(self, key: str, size_bytes: int, compute_seconds: float, code: str | None = None) -> None:
        """A value took more disk than the compute it saves is worth.

        Once per statement, named by its code, which is what a reader can find
        in their notebook; a key is not.
        """
        ident = (code or "").strip() or str(key)
        if ident in self._warned_not_worth:
            return
        self._warned_not_worth.add(ident)
        lines = [ln for ln in ident.splitlines() if ln.strip()]
        # A loop body's stored code starts with its context marker comment.
        first = next((ln for ln in lines if not ln.lstrip().startswith("#")), lines[0] if lines else str(key))
        named = f"`{first[:80]}`" if code else repr(key)
        if self._held is not None:
            self._held.append((named, size_bytes, compute_seconds))
            return
        self._say_not_worth([(named, size_bytes, compute_seconds)])

    def _say_not_worth(self, refused: list[tuple[str, int, float]]) -> None:
        ceiling = human_bytes(WORTH_CEILING_BYTES_PER_SECOND)
        if len(refused) == 1:
            named, size_bytes, compute_seconds = refused[0]
            rate = size_bytes / max(compute_seconds, 1e-9) / (1024**2)
            what = (
                f"the value of {named} is {human_bytes(size_bytes)} serialized but "
                f"only takes {compute_seconds:.2f}s to recompute -- "
                f"{rate:,.0f} MiB of cache per second saved, against the "
                f"{ceiling} per second cash is willing to spend. It was not "
                f"persisted, so it is recomputed rather than restored."
            )
        else:
            shown = ", ".join(
                f"{named} ({human_bytes(size)} for {secs:.2f}s)" for named, size, secs in refused[: self.NAMED]
            )
            more = len(refused) - self.NAMED
            what = (
                f"{len(refused)} values in this cell take more cache per second "
                f"saved than the {ceiling} cash is willing to spend: {shown}"
                + (f" and {more} more" if more > 0 else "")
                + ". They were not persisted, so they are recomputed rather "
                "than restored."
            )
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-NOT-WORTH-BYTES",
            what,
            "nothing, if the recompute is cheap enough that you had not "
            "noticed it -- that is the trade being made. To cache it anyway, "
            "say so explicitly: `@cash:persist` on the statement, or "
            "`@cash.cache` on the function, both of which cash honours without "
            "re-taking the decision. Caching something smaller -- the "
            "aggregate, the sample, the columns you use -- is usually the "
            "better answer for a value this large. `cash inspect` in a "
            "terminal lists what the cache does hold, each entry's size "
            "next to the time it saves.",
            location=("<cash>", 1),
        )
