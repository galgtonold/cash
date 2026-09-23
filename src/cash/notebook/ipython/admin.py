"""The session-inspection magics, ``%cash_stats`` and ``%cash_provenance``.

They live in a mixin inherited by :class:`~cash.notebook.ipython.magics.CashMagics`,
so IPython registers them with the rest.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from IPython.core.magic import line_magic

from cash._console import safe_text

from ...backends._writes import discarded_writes
from ._args import parse_mode, strip_inline_comment

if TYPE_CHECKING:
    from .magics import CashMagics

__all__ = ["CashAdminMagicsMixin"]


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


class CashAdminMagicsMixin:
    """Mixin providing the session-inspection magics.

    All methods expect ``self`` to be a fully-initialised
    :class:`~cash.notebook.ipython.magics.CashMagics` instance (i.e. attributes such
    as ``self._cash_instance``, ``self.tracking_state.variable_lineage``, etc. are available).
    """

    # ------------------------------------------------------------------
    # Session statistics
    # ------------------------------------------------------------------

    @line_magic
    def cash_stats(self: CashMagics, line: str) -> None:
        """Display session-wide caching statistics.

        Usage::

            %cash_stats           # Show human-readable stats
            %cash_stats json      # Return JSON format
            %cash_stats reset     # Reset session stats
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
            # Rebuilt from the same definition a fresh session uses, so a new
            # counter can never be added to the stats and silently survive a
            # reset (it already happened once).
            # Local: import cycle ipython.admin -> ipython.magics -> ipython.admin.
            from .magics import new_session_stats

            self._session.stats.update(new_session_stats())
            # The verified-saving baselines are part of the stats, not of the
            # cache: a reset must drop them too or savings would be credited
            # against measurements the reset claims to have forgotten.
            self._session.measured_compute.clear()
            # Same rule for the decorator baselines: a reset that keeps
            # them would credit a post-reset hit as "verified" against a compute
            # the reset claims to have forgotten.
            self._session.measured_decorator_compute.clear()
            # On disk too: a reset that kept them would credit a later hit
            # against a measurement it claims to have forgotten.
            self._baselines().clear()
            print("[OK] Session statistics reset.")
            return

        stats = self._session.stats
        total_stmts = stats["statements_computed"] + stats["statements_restored"] + stats["statements_skipped"]
        hit_rate = (stats["statements_restored"] + stats["statements_skipped"]) / max(total_stmts, 1) * 100

        # The rate over ALL statements answers a question nobody asked: its
        # denominator is dominated by prints, imports and cheap assignments that
        # cash deliberately never tried to cache. Counting cash's own correct
        # "not worth caching" decisions as misses reported 14.9% for a session
        # in which 100% of the expensive statements hit. The earlier fix addressed
        # an OVERstatement of savings; that is the same failure inverted, so the
        # same rule binds: the number must not imply a conclusion the data does
        # not support, in EITHER direction.
        cacheable_hit = stats.get("statements_cacheable_hit", 0)
        cacheable_miss = stats.get("statements_cacheable_miss", 0)
        cacheable_total = cacheable_hit + cacheable_miss
        cacheable_rate = (cacheable_hit / cacheable_total * 100) if cacheable_total else None

        # Two nets, because two different qualities of evidence.
        #
        # ``gross_saved`` is a counterfactual: each restore is credited with the
        # compute time recorded when the value was FIRST cached. Nothing
        # re-measures that. If the first run was colder — cold page cache, cold
        # imports — the credit is stale-high, and a session that was slower by
        # wall clock still prints a win. That is the lie the verified net fixes, and it
        # is not fixable by estimating harder: the true recompute cost cannot be
        # known without doing the recompute.
        #
        # So the HEADLINE net is credited only from savings this session
        # verified by computing the same statement itself. The gross figure is
        # still shown, explicitly as an unverified upper bound. This
        # deliberately UNDERSTATES a session that really did save time but never
        # re-measured a baseline — an understatement is a defensible error here;
        # an overstatement is the bug.
        gross_saved = stats["total_time_saved"]
        verified_saved = stats.get("total_verified_saved", 0.0)
        # Measured on this machine in an earlier kernel, at the least it ever
        # cost. Evidence of the same kind as ``verified``, one run older --
        # and the only kind a Restart & Run All can have, which is where the
        # net used to print as a range straddling zero (round 30).
        measured_saved = stats.get("total_measured_saved", 0.0)
        overhead = stats.get("total_overhead", 0.0)
        net_saved = verified_saved + measured_saved - overhead
        net_upper = gross_saved - overhead

        # NOTE: We deliberately don't walk the backend here (no
        # ``list_entries()``). On disk-backed caches with thousands of
        # entries that's an O(N) scan that opens every metadata file --
        # the same pathology we removed from ``_diagnose_miss`` in the
        # 2026-05-18 overhead pass. Users who want a backend-wide view
        # can run ``cash inspect`` from the command line.

        # Writes that failed and were thrown away. A silent loss: the entry is
        # absent, so that work recomputes every run, and none of the counters
        # above can show it -- a discarded write is not a miss, it is a hit that
        # never got the chance to exist. The only other report is a logger
        # warning from ``_report_failed_writes`` at shutdown, which in a
        # notebook means at kernel death, i.e. never. This is the one place a
        # user asking "is caching working?" can actually be told that it isn't.
        discarded = discarded_writes()

        if mode == "json":
            result = {
                **stats,
                "discarded_writes": len(discarded),
                "net_time_saved": net_saved,
                "net_time_saved_upper_bound": net_upper,
                "total_measured_saved": measured_saved,
                # False ⇒ the upper bound rests on baselines nobody re-measured,
                # so its sign is not evidence of anything.
                "net_sign_verified": net_saved >= 0 or net_upper < 0,
                "hit_rate_percent": round(hit_rate, 1),
                # Hits over the statements caching was ever on the table for.
                # ``None`` (not 0.0) when nothing this session cleared the
                # floor: a rate with an empty denominator is undefined, and
                # emitting 0.0 would read as "cash missed everything".
                "hit_rate_cacheable_percent": (round(cacheable_rate, 1) if cacheable_rate is not None else None),
                "statements_cacheable_total": cacheable_total,
            }
            print(json.dumps(result, indent=2))
            return

        print("Cash Session Statistics")
        # Round 25: these reset on a kernel restart and were read as the
        # project's totals. Name the scope up front.
        print("  (since this kernel started; a restart resets them)")
        print("-" * 40)
        print(f"  Cells executed:      {stats['cells_executed']}")
        print(f"  Statements computed: {stats['statements_computed']}")
        print(f"  Statements restored: {stats['statements_restored']}")
        print(f"  Statements skipped:  {stats['statements_skipped']}")
        trivial = total_stmts - cacheable_total
        if cacheable_rate is None:
            # Honest silence. No statement was expensive enough to cache, so
            # there is no hit rate to report -- printing "0%" here would blame
            # cash for correctly declining to cache a notebook of prints.
            print("  Cache hit rate:      n/a  (no statement was expensive enough to cache)")
        elif trivial <= 0:
            print(f"  Cache hit rate:      {cacheable_rate:.1f}%  ({cacheable_hit}/{cacheable_total} statements)")
        else:
            # Both numbers, with the meaningful one first and each labelled by
            # its own denominator so neither can be read as the other.
            print(
                f"  Cache hit rate:      {cacheable_rate:.1f}%  "
                f"({cacheable_hit}/{cacheable_total} statements worth caching)"
            )
            print(
                f"                       {hit_rate:.1f}% counting all {total_stmts} "
                f"statements -- the other {trivial} were too"
            )
            print("                       cheap to cache, so cash never tried: not misses.")
        print()
        print(f"  Compute time:        {_fmt_time(stats['total_compute_time'])}")
        print(f"  Gross time saved:    {_fmt_time(gross_saved)}  (estimated)")
        print(f"  Cash overhead:       {_fmt_time(overhead)}  (measured)")
        # NET is the honest headline: what cash actually bought you once its own
        # tax is paid, counting only savings this session could verify. Show a
        # negative plainly rather than flooring it.
        if net_saved >= 0:
            # "verified" = this kernel recomputed it; "measured" = an earlier
            # run on this machine did, and the least it ever cost is credited.
            basis = "verified" if measured_saved <= 0 else "measured"
            print(f"  Net time saved:      {_fmt_signed_time(net_saved)}  ({basis})")
        elif net_upper < 0:
            # Even the most generous reading of the cache's own baselines is a
            # loss, so the sign is certain without verifying anything.
            print(
                f"  Net time saved:      {_fmt_signed_time(net_upper)}"
                f"  (cash cost you {_fmt_time(-net_upper)} this session)"
            )
        else:
            # The unverified case: gross says win, measurement says nothing.
            # Report the floor, and the ceiling as a claim rather than a fact.
            print(
                f"  Net time saved:      at least {_fmt_signed_time(net_saved)}, at best {_fmt_signed_time(net_upper)}"
            )
            print(
                f"    Cash measured only the {_fmt_time(overhead)} it spent. The "
                f"{_fmt_time(gross_saved)} it avoided is what these values cost"
            )
            print("    when first cached; if they would recompute faster today (warm file cache,")
            print("    warm imports), the real figure is nearer the low end. Time a run with")
            print("    caching off to settle it.")
        if discarded:
            print()
            print(f"  Discarded writes:    {len(discarded)}  -- these results were NOT cached")
            print("    A cache write failed, so that work recomputes every run. Nothing raised")
            print("    at the time, which is why the numbers above can look healthy anyway.")
            print(f"    First: {discarded[0][1]}")
            if len(discarded) > 1:
                print(f"    ... and {len(discarded) - 1} more.")

        print()
        tracked = len(self.tracking_state.variable_lineage)
        print(f"  Tracked variables:   {tracked}")
        print()
        # Points at the CLI, not at an admin magic: there has never been
        # one. Sending a user who is looking at a multi-hundred-MB .cash
        # to a UsageError is worse than saying nothing, and inspecting the
        # backend is exactly what they came here to do.
        # `cash inspect` named too: it is the view that answers "is my cache
        # worth what it costs", and two round-27 testers found it only by
        # hunting through docs/cli.md.
        print("  The cache on disk outlives this kernel. In a terminal, `cash info`")
        print("  gives its size and `cash inspect` lists its entries, the time each")
        print("  saves beside the space it takes (`cash clear` empties it).")

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    @line_magic
    def cash_provenance(self: CashMagics, line: str) -> None:
        """Show provenance (computation history) for a variable.

        Usage::

            %cash_provenance x           - Show how 'x' was computed
            %cash_provenance x --graph   - Include dependency graph
            %cash_provenance x --time    - Include a timeline of computations
            %cash_provenance x --json    - Output as JSON
            %cash_provenance --all       - List all tracked variables
            %cash_provenance --clear     - Clear provenance data
        """
        parts = strip_inline_comment(line).split()

        if not parts or parts[0] == "--all":
            tracked = sorted(self._session.provenance.tracked_variables)
            if not tracked:
                print("No provenance data recorded yet.")
            else:
                print(f"Tracked variables ({len(tracked)}):")
                for var in tracked:
                    latest = self._session.provenance.get_latest(var)
                    status_icon = {"computed": "[C]", "restored": "[R]", "skipped": "[S]"}.get(
                        latest.status if latest else "", "[?]"
                    )
                    history_count = len(self._session.provenance.get_history(var))
                    print(f"  {status_icon} {var} ({history_count} records)")
            return

        if parts[0] == "--clear":
            self._session.provenance.clear()
            print("Provenance data cleared.")
            return

        var_name = parts[0]
        show_graph = "--graph" in parts
        show_time = "--time" in parts or "--timeline" in parts
        as_json = "--json" in parts

        if as_json:
            print(self._session.provenance.to_json(var_name))
        else:
            print(
                safe_text(
                    self._session.provenance.format_provenance(
                        var_name,
                        show_graph=show_graph,
                        show_timeline=show_time,
                    )
                )
            )
