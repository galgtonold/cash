"""What the session-inspection magics, ``%cash_stats`` and ``%cash_provenance``,
print.

:class:`~cash.notebook.ipython.magics.CashMagics` registers the magics and
hands each call to :func:`show_stats` or :func:`show_provenance` here, with
the session they read.
"""

from __future__ import annotations

import json

from cash._console import safe_text

from ...backends._writes import discarded_writes
from ..provenance import BADGE_WORDS, ProvenanceTracker
from ._args import parse_mode, strip_inline_comment
from .session import CashSession, StatsSummary

__all__ = ["show_provenance", "show_stats"]


# ---------------------------------------------------------------------------
# Module-level formatting helpers (used by cash_stats)
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}us"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def _fmt_signed_time(seconds: float) -> str:
    """Format a possibly-negative duration, e.g. ``-2.3s``.

    ``_fmt_time`` assumes a non-negative value (its ``< 0.001`` branch would
    turn -2.3 into ``-2300000us``), so the NET saving — which is deliberately
    allowed to go negative when cash cost more than it saved — is formatted
    here as ``-<magnitude>``.
    """
    if seconds < 0:
        return f"-{_fmt_time(-seconds)}"
    return _fmt_time(seconds)


def _stats_json(stats: dict, summary: StatsSummary, discarded: list) -> dict:
    """``%cash_stats json``: the counters with the derived figures."""
    return {
        **stats,
        "discarded_writes": len(discarded),
        "net_time_saved": summary.net_saved,
        "net_time_saved_upper_bound": summary.net_upper,
        "total_measured_saved": summary.measured_saved,
        # False ⇒ the upper bound rests on baselines nobody re-measured,
        # so its sign is not evidence of anything.
        "net_sign_verified": summary.net_saved >= 0 or summary.net_upper < 0,
        "hit_rate_percent": round(summary.hit_rate, 1),
        # Hits over the statements caching was ever on the table for.
        # ``None`` (not 0.0) when nothing this session cleared the
        # floor: a rate with an empty denominator is undefined, and
        # emitting 0.0 would read as "cash missed everything".
        "hit_rate_cacheable_percent": (
            round(summary.cacheable_rate, 1) if summary.cacheable_rate is not None else None
        ),
        "statements_cacheable_total": summary.cacheable_total,
    }


def _print_counts(stats: dict, summary: StatsSummary) -> None:
    """The statement counts and the hit rate, each rate with its denominator."""
    print("Cash Session Statistics")
    # These reset on a kernel restart and were read as the
    # project's totals. Name the scope up front.
    print("  (since this kernel started; a restart resets them)")
    print("-" * 40)
    print(f"  Cells executed:      {stats['cells_executed']}")
    print(f"  Statements computed: {stats['statements_computed']}")
    print(f"  Statements restored: {stats['statements_restored']}")
    print(f"  Statements skipped:  {stats['statements_skipped']}")
    trivial = summary.total_stmts - summary.cacheable_total
    if summary.cacheable_rate is None:
        # Honest silence. No statement was expensive enough to cache, so
        # there is no hit rate to report -- printing "0%" here would blame
        # cash for correctly declining to cache a notebook of prints.
        print("  Cache hit rate:      n/a  (no statement was expensive enough to cache)")
    elif trivial <= 0:
        print(
            f"  Cache hit rate:      {summary.cacheable_rate:.1f}%  "
            f"({summary.cacheable_hit}/{summary.cacheable_total} statements)"
        )
    else:
        # Both numbers, with the meaningful one first and each labelled by
        # its own denominator so neither can be read as the other.
        print(
            f"  Cache hit rate:      {summary.cacheable_rate:.1f}%  "
            f"({summary.cacheable_hit}/{summary.cacheable_total} statements worth caching)"
        )
        print(
            f"                       {summary.hit_rate:.1f}% counting all {summary.total_stmts} "
            f"statements -- the other {trivial} were too"
        )
        print("                       cheap to cache, so cash never tried: not misses.")


def _print_time(stats: dict, summary: StatsSummary) -> None:
    """Compute, gross saving, overhead and the net, as certain as the evidence."""
    net_saved, net_upper = summary.net_saved, summary.net_upper
    overhead, gross_saved = summary.overhead, summary.gross_saved
    print(f"  Compute time:        {_fmt_time(stats['total_compute_time'])}")
    print(f"  Gross time saved:    {_fmt_time(gross_saved)}  (estimated)")
    print(f"  Cash overhead:       {_fmt_time(overhead)}  (measured)")
    # NET is the honest headline: what cash actually bought you once its own
    # tax is paid, counting only savings this session could verify. Show a
    # negative plainly rather than flooring it.
    if net_saved >= 0:
        # "verified" = this kernel recomputed it; "measured" = an earlier
        # run on this machine did, and the least it ever cost is credited.
        basis = "verified" if summary.measured_saved <= 0 else "measured"
        print(f"  Net time saved:      {_fmt_signed_time(net_saved)}  ({basis})")
    elif net_upper < 0:
        # Even the most generous reading of the cache's own baselines is a
        # loss, so the sign is certain without verifying anything.
        print(
            f"  Net time saved:      {_fmt_signed_time(net_upper)}  (cash cost you {_fmt_time(-net_upper)} this session)"
        )
    else:
        # The unverified case: gross says win, measurement says nothing.
        # Report the floor, and the ceiling as a claim rather than a fact.
        print(f"  Net time saved:      at least {_fmt_signed_time(net_saved)}, at best {_fmt_signed_time(net_upper)}")
        print(
            f"    Cash measured only the {_fmt_time(overhead)} it spent. The "
            f"{_fmt_time(gross_saved)} it avoided is what these values cost"
        )
        print("    when first cached; if they would recompute faster today (warm file cache,")
        print("    warm imports), the real figure is nearer the low end. Time a run with")
        print("    caching off to settle it.")


def _print_discarded(discarded: list) -> None:
    """The writes that failed, when there were any."""
    if not discarded:
        return
    print()
    print(f"  Discarded writes:    {len(discarded)}  -- these results were NOT cached")
    print("    A cache write failed, so that work recomputes every run. Nothing raised")
    print("    at the time, which is why the numbers above can look healthy anyway.")
    print(f"    First: {discarded[0][1]}")
    if len(discarded) > 1:
        print(f"    ... and {len(discarded) - 1} more.")


def _print_footer(tracked: int) -> None:
    print()
    print(f"  Tracked variables:   {tracked}")
    print()
    # Points at the CLI: there is no magic for the backend. `cash inspect` is
    # named too: it is the view that answers "is my cache worth what it
    # costs", and users found it only by hunting through docs/cli.md.
    print("  The cache on disk outlives this kernel. In a terminal, `cash info`")
    print("  gives its size and `cash inspect` lists its entries, the time each")
    print("  saves beside the space it takes (`cash clear` empties it).")


def show_stats(session: CashSession, line: str, tracked_variables: int) -> None:
    """``%cash_stats [json|reset]`` for *session*.

    *tracked_variables* is the count of variables with a lineage, for the
    footer.
    """
    # ``reset`` mutates state, so an unrecognised argument must not fall
    # through to "print the stats" — that reports success (stats appear) for
    # a reset that never happened.
    mode = parse_mode(line, ("", "json", "reset"))
    if mode is None:
        print(f"[Error] %cash_stats: unrecognised argument: {strip_inline_comment(line)!r}")
        print("   Valid forms: %cash_stats | %cash_stats json | %cash_stats reset")
        return

    if mode == "reset":
        session.reset()
        print("[OK] Session statistics reset.")
        return

    stats = session.stats
    summary = session.summary()
    # Deliberately no backend walk here (no ``list_entries()``): on a
    # disk cache with thousands of entries that is an O(N) scan that opens
    # every metadata file. ``cash inspect`` gives the backend-wide view.

    # Writes that failed and were thrown away. A silent loss: the entry is
    # absent, so that work recomputes every run, and none of the counters
    # above can show it -- a discarded write is not a miss, it is a hit that
    # never got the chance to exist. The only other report is a logger
    # warning from ``_report_failed_writes`` at shutdown, which in a
    # notebook means at kernel death, i.e. never. This is the one place a
    # user asking "is caching working?" can actually be told that it isn't.
    discarded = discarded_writes()

    if mode == "json":
        print(json.dumps(_stats_json(stats, summary, discarded), indent=2))
        return

    _print_counts(stats, summary)
    print()
    _print_time(stats, summary)
    _print_discarded(discarded)
    _print_footer(tracked_variables)


def _provenance_unknown_args(parts: list[str]) -> list[str]:
    """The words of a ``%cash_provenance`` line it does not understand."""
    if parts and parts[0] in ("--all", "--clear"):
        return parts[1:]
    if not parts:
        return []
    unknown = [p for p in parts[1:] if p not in ("--graph", "--time", "--timeline", "--json")]
    if parts[0].startswith("-"):
        unknown.insert(0, parts[0])
    return unknown


def _print_tracked(provenance: ProvenanceTracker) -> None:
    """``%cash_provenance --all``: every tracked variable and its latest status."""
    tracked = sorted(provenance.tracked_variables)
    if not tracked:
        print("No provenance data recorded yet.")
        return
    print(f"Tracked variables ({len(tracked)}):")
    for var in tracked:
        latest = provenance.get_latest(var)
        status_word = BADGE_WORDS.get(latest.status if latest else "", "UNKNOWN")
        history_count = len(provenance.get_history(var))
        print(f"  {status_word:<8} {var} ({history_count} records)")


def show_provenance(provenance: ProvenanceTracker, line: str) -> None:
    """``%cash_provenance <var> [--graph] [--time] [--json] | --all | --clear``."""
    parts = strip_inline_comment(line).split()
    # A misspelt flag (``--grpah``) or a stray word must not be dropped
    # silently, as if the output it asked for simply had nothing to show.
    unknown = _provenance_unknown_args(parts)
    if unknown:
        print(f"[Error] %cash_provenance: unrecognised argument: {' '.join(unknown)!r}")
        print("   Valid forms: %cash_provenance <var> [--graph] [--time] [--json] | --all | --clear")
        return

    if not parts or parts[0] == "--all":
        _print_tracked(provenance)
        return

    if parts[0] == "--clear":
        provenance.clear()
        print("Provenance data cleared.")
        return

    var_name = parts[0]
    if "--json" in parts:
        print(provenance.to_json(var_name))
        return
    print(
        safe_text(
            provenance.format_provenance(
                var_name,
                show_graph="--graph" in parts,
                show_timeline="--time" in parts or "--timeline" in parts,
            )
        )
    )
