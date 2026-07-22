"""Administrative magic commands extracted from CashMagics.

This module provides a mixin class with utility / diagnostic / benchmark
magic commands.  The mixin is inherited by :class:`~cash.notebook.ipython.magics.CashMagics`
so that IPython recognises these commands automatically.
"""

from __future__ import annotations

import logging
import pickle
import time
from typing import TYPE_CHECKING, Any

from IPython.core.magic import line_magic

from cash.utils import safe_text

from ._args import parse_mode, strip_inline_comment

if TYPE_CHECKING:
    from .magics import CashMagics

logger = logging.getLogger(__name__)

__all__ = ["CashAdminMagicsMixin"]


# ---------------------------------------------------------------------------
# Module-level formatting helpers (used by cash_stats)
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}us"
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


def _fmt_signed_time(seconds: float) -> str:
    """Format a possibly-negative duration, e.g. ``-2.3s``.

    ``_fmt_time`` assumes a non-negative value (its ``< 0.001`` branch would
    turn -2.3 into ``-2300000us``), so the NET saving — which is deliberately
    allowed to go negative when cash cost more than it saved — is formatted
    here as ``-<magnitude>`` (CAS-143).
    """
    if seconds < 0:
        return f"-{_fmt_time(-seconds)}"
    return _fmt_time(seconds)


def _fmt_size(bytes_val: int) -> str:
    if bytes_val < 1024:
        return f"{bytes_val}B"
    if bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f}KB"
    if bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f}MB"
    return f"{bytes_val/1024**3:.2f}GB"


# ---------------------------------------------------------------------------
# Module-level I/O helpers (used by cash_diff)
# ---------------------------------------------------------------------------

def _load_cache_file_data(filepath: str) -> dict | None:
    """Try to load a cache file as JSON first, then pickle. Returns None on format failure.

    Raises FileNotFoundError if the file does not exist.
    """
    import json
    import pickle as pkl
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    except OSError:
        raise
    try:
        with open(filepath, 'rb') as f:
            return pkl.load(f)
    except (pickle.UnpicklingError, TypeError, ValueError):
        pass
    return None


class CashAdminMagicsMixin:
    """Mixin providing administrative / diagnostic / benchmark magic commands.

    All methods expect ``self`` to be a fully-initialised
    :class:`~cash.notebook.ipython.magics.CashMagics` instance (i.e. attributes such
    as ``self._cash_instance``, ``self._tracking_state.variable_lineage``, etc. are available).
    """

    # ------------------------------------------------------------------
    # Cache integrity
    # ------------------------------------------------------------------

    @line_magic
    def cash_verify(self: CashMagics, line: str) -> None:
        """Verify cache integrity and report any issues.

        Usage::

            %cash_verify          # Check all cache entries
            %cash_verify --fix    # Check and remove corrupted entries
        """
        fix_mode = '--fix' in line if line else False
        backend = self._cash_instance.backend

        print("[Verify] Checking cache integrity...")

        try:
            total, healthy, corrupted, issues = self._verify_backend_entries(backend, fix_mode)
        except (OSError, AttributeError, TypeError) as e:
            print(f"[Error] Error accessing backend: {e}")
            return

        print("\n[Report] Cache Verification:")
        print(f"   Total entries: {total}")
        print(f"   Healthy: {healthy}")
        print(f"   Corrupted: {corrupted}")

        if issues:
            print("\n[Warning] Issues found:")
            for issue in issues[:20]:
                print(f"   {issue}")
            if len(issues) > 20:
                print(f"   ... and {len(issues) - 20} more")
            if fix_mode:
                print(f"\n[Fixed] {corrupted} corrupted entries deleted.")
            else:
                print("\n[Tip] Run '%cash_verify --fix' to remove corrupted entries.")
        else:
            print("\n[OK] All cache entries are healthy!")

        lineage_issues = self._count_orphaned_lineages()
        if lineage_issues:
            print(f"\n[Lineage] {lineage_issues} tracked variable(s) not in current namespace")
            print("   This is normal if you haven't re-run all cells yet.")

    def _verify_backend_entries(
        self: CashMagics, backend: Any, fix_mode: bool,
    ) -> tuple[int, int, int, list[str]]:
        """Check all backend entries and return (total, healthy, corrupted, issues)."""
        entries = backend.list_entries()
        total = len(entries)
        healthy = 0
        corrupted = 0
        issues: list[str] = []

        for entry in entries:
            key = entry.get('key', '')
            try:
                value = backend.get(key)
                if value is not None:
                    healthy += 1
                else:
                    issues.append(f"[Warning] Key '{key[:40]}...' returns None (missing data)")
                    corrupted += 1
                    if fix_mode:
                        backend.delete(key)
            except (KeyError, TypeError, ValueError, OSError, pickle.UnpicklingError) as e:
                issues.append(f"[Error] Key '{key[:40]}...' corrupted: {str(e)[:60]}")
                corrupted += 1
                if fix_mode:
                    try:
                        backend.delete(key)
                    except (OSError, TypeError, KeyError):
                        logger.debug("Failed to delete corrupted cache entry %s", key[:40])

        return total, healthy, corrupted, issues

    def _count_orphaned_lineages(self: CashMagics) -> int:
        """Count lineage entries whose variables are no longer in the namespace."""
        count = 0
        for var_name in self._tracking_state.variable_lineage:
            if var_name not in self.shell.user_ns:
                count += 1
                if self._debug:
                    print(f"   [Info] Lineage tracked for '{var_name}' but variable not in namespace")
        return count

    # ------------------------------------------------------------------
    # State repair
    # ------------------------------------------------------------------

    @line_magic
    def cash_repair(self: CashMagics, line: str) -> None:
        """Repair cash state by clearing corrupted data and rebuilding metadata.

        Usage::

            %cash_repair              # Clear corrupted entries
            %cash_repair --full       # Full reset (clear all cache + state)
            %cash_repair --state      # Reset only in-memory state (keep cache)
        """
        # An unrecognised argument must NEVER fall through to the default
        # repair: this magic is the documented recovery from a poisoned cache,
        # so quietly running a lesser repair and reporting success turns a
        # recoverable state into a confidently wrong one (CAS-181).
        mode = parse_mode(line, ('', '--full', '--state'))
        if mode is None:
            print(f"[Error] %cash_repair: unrecognised argument: {strip_inline_comment(line)!r}")
            print("   Nothing was repaired. Valid forms:")
            print("     %cash_repair            Remove corrupted entries (keeps cache)")
            print("     %cash_repair --state    Reset in-memory state (keeps cache)")
            print("     %cash_repair --full     Clear ALL cache and state")
            return

        if mode == '--full':
            print("[Repair] Full repair: clearing all cache and state...")
            try:
                self._cash_instance.backend.clear()
                print("   [OK] Cache cleared")
            except (OSError, AttributeError, TypeError) as e:
                print(f"   [Error] Error clearing cache: {e}")

            self._tracking_state.variable_lineage.clear()
            self._tracking_state.executed_cell_codes.clear()
            self._tracking_state.executed_cell_hashes.clear()
            self._tracking_state.executed_file_deps.clear()
            self._tracking_state.vars_with_mutation_lineage.clear()
            self._tracking_state.variable_hashes.clear()
            self._tracking_state.variable_sources.clear()
            self._tracking_state.current_session_hashes.clear()
            self._tracking_state.executed_cell_source_hashes.clear()
            self._tracking_state.rng_post_states.clear()
            self._tracking_state.rng_pre_states.clear()
            self._tracking_state.observed_rng_cells.clear()
            self._executed_cell_raw_codes.clear()
            print("   [OK] In-memory state cleared")
            print("\n[OK] Full repair complete. Re-run your cells to rebuild cache.")

        elif mode == '--state':
            print("[Repair] State repair: clearing in-memory tracking state...")
            self._tracking_state.variable_lineage.clear()
            self._tracking_state.executed_cell_codes.clear()
            self._tracking_state.executed_cell_hashes.clear()
            self._tracking_state.executed_file_deps.clear()
            self._tracking_state.vars_with_mutation_lineage.clear()
            self._tracking_state.variable_hashes.clear()
            self._tracking_state.variable_sources.clear()
            self._tracking_state.current_session_hashes.clear()
            self._tracking_state.executed_cell_source_hashes.clear()
            self._tracking_state.rng_post_states.clear()
            self._tracking_state.rng_pre_states.clear()
            self._tracking_state.observed_rng_cells.clear()
            self._executed_cell_raw_codes.clear()
            print("   [OK] In-memory state cleared (cache preserved)")
            print("\n[OK] State repair complete. Re-run your cells to rebuild lineage tracking.")

        else:
            print("[Repair] Repairing cache: removing corrupted entries...")
            self.cash_verify('--fix')

            stale_vars = [v for v in self._tracking_state.variable_lineage if v not in self.shell.user_ns]
            for v in stale_vars:
                del self._tracking_state.variable_lineage[v]
                self._tracking_state.executed_cell_codes.pop(v, None)
                self._tracking_state.executed_cell_hashes.pop(v, None)
                self._tracking_state.executed_file_deps.pop(v, None)

            if stale_vars:
                print(f"\n[Cleaned] {len(stale_vars)} stale lineage entries: {', '.join(stale_vars[:5])}")

            # Name the repair that ran. A bare "Repair complete." reads as "your
            # cache is clean now" — which is what the user reaching for this
            # command after a bad value actually wants to hear, and is not what
            # this branch did (CAS-181).
            print("\n[OK] Repair complete (default mode): corrupted entries removed.")
            print("   Cached values were kept. To clear the cache: %cash_repair --full")

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
        # a reset that never happened (CAS-181).
        mode = parse_mode(line, ('', 'json', 'reset'))
        if mode is None:
            print(f"[Error] %cash_stats: unrecognised argument: {strip_inline_comment(line)!r}")
            print("   Valid forms: %cash_stats | %cash_stats json | %cash_stats reset")
            return

        if mode == 'reset':
            # Rebuilt from the same definition a fresh session uses, so a new
            # counter can never be added to the stats and silently survive a
            # reset (it already happened once).
            from .magics import new_session_stats
            self._session.stats.update(new_session_stats())
            # The verified-saving baselines are part of the stats, not of the
            # cache: a reset must drop them too or savings would be credited
            # against measurements the reset claims to have forgotten.
            self._session.measured_compute.clear()
            # Same rule for the decorator baselines (CAS-222): a reset that keeps
            # them would credit a post-reset hit as "verified" against a compute
            # the reset claims to have forgotten.
            self._session.measured_decorator_compute.clear()
            print("[OK] Session statistics reset.")
            return

        stats = self._session.stats
        total_stmts = stats['statements_computed'] + stats['statements_restored'] + stats['statements_skipped']
        hit_rate = (stats['statements_restored'] + stats['statements_skipped']) / max(total_stmts, 1) * 100

        # The rate over ALL statements answers a question nobody asked: its
        # denominator is dominated by prints, imports and cheap assignments that
        # cash deliberately never tried to cache. Counting cash's own correct
        # "not worth caching" decisions as misses reported 14.9% for a session
        # in which 100% of the expensive statements hit (CAS-177). CAS-157 fixed
        # an OVERstatement of savings; that is the same failure inverted, so the
        # same rule binds: the number must not imply a conclusion the data does
        # not support, in EITHER direction.
        cacheable_hit = stats.get('statements_cacheable_hit', 0)
        cacheable_miss = stats.get('statements_cacheable_miss', 0)
        cacheable_total = cacheable_hit + cacheable_miss
        cacheable_rate = (cacheable_hit / cacheable_total * 100) if cacheable_total else None

        # Two nets, because two different qualities of evidence (CAS-157).
        #
        # ``gross_saved`` is a counterfactual: each restore is credited with the
        # compute time recorded when the value was FIRST cached. Nothing
        # re-measures that. If the first run was colder — cold page cache, cold
        # imports — the credit is stale-high, and a session that was slower by
        # wall clock still prints a win. That is the lie CAS-157 fixes, and it
        # is not fixable by estimating harder: the true recompute cost cannot be
        # known without doing the recompute.
        #
        # So the HEADLINE net is credited only from savings this session
        # verified by computing the same statement itself. The gross figure is
        # still shown, explicitly as an unverified upper bound. This
        # deliberately UNDERSTATES a session that really did save time but never
        # re-measured a baseline — an understatement is a defensible error here;
        # an overstatement is the bug.
        gross_saved = stats['total_time_saved']
        verified_saved = stats.get('total_verified_saved', 0.0)
        overhead = stats.get('total_overhead', 0.0)
        net_saved = verified_saved - overhead
        net_upper = gross_saved - overhead

        # NOTE: We deliberately don't walk the backend here (no
        # ``list_entries()``). On disk-backed caches with thousands of
        # entries that's an O(N) scan that opens every metadata file --
        # the same pathology we removed from ``_diagnose_miss`` in the
        # 2026-05-18 overhead pass. Users who want a backend-wide view
        # can run ``%cash_admin`` (or the lower-level inspection tools).

        if mode == 'json':
            import json
            result = {
                **stats,
                'net_time_saved': net_saved,
                'net_time_saved_upper_bound': net_upper,
                # False ⇒ the upper bound rests on baselines nobody re-measured,
                # so its sign is not evidence of anything.
                'net_sign_verified': net_saved >= 0 or net_upper < 0,
                'hit_rate_percent': round(hit_rate, 1),
                # Hits over the statements caching was ever on the table for.
                # ``None`` (not 0.0) when nothing this session cleared the
                # floor: a rate with an empty denominator is undefined, and
                # emitting 0.0 would read as "cash missed everything" (CAS-177).
                'hit_rate_cacheable_percent': (
                    round(cacheable_rate, 1) if cacheable_rate is not None else None
                ),
                'statements_cacheable_total': cacheable_total,
            }
            print(json.dumps(result, indent=2))
            return

        print("Cash Session Statistics")
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
            print("  Cache hit rate:      n/a  (no statement was expensive "
                  "enough to cache)")
        elif trivial <= 0:
            print(f"  Cache hit rate:      {cacheable_rate:.1f}%  "
                  f"({cacheable_hit}/{cacheable_total} statements)")
        else:
            # Both numbers, with the meaningful one first and each labelled by
            # its own denominator so neither can be read as the other.
            print(f"  Cache hit rate:      {cacheable_rate:.1f}%  "
                  f"({cacheable_hit}/{cacheable_total} statements worth caching)")
            print(f"                       {hit_rate:.1f}% counting all {total_stmts} "
                  f"statements -- the other {trivial} were too")
            print("                       cheap to cache, so cash never tried: "
                  "not misses.")
        print()
        print(f"  Compute time:        {_fmt_time(stats['total_compute_time'])}")
        print(f"  Gross time saved:    {_fmt_time(gross_saved)}  (estimated)")
        print(f"  Cash overhead:       {_fmt_time(overhead)}  (measured)")
        # NET is the honest headline: what cash actually bought you once its own
        # tax is paid, counting only savings this session could verify. Show a
        # negative plainly rather than flooring it.
        if net_saved >= 0:
            print(f"  Net time saved:      {_fmt_signed_time(net_saved)}  (verified)")
        elif net_upper < 0:
            # Even the most generous reading of the cache's own baselines is a
            # loss, so the sign is certain without verifying anything.
            print(f"  Net time saved:      {_fmt_signed_time(net_upper)}"
                  f"  (cash cost you {_fmt_time(-net_upper)} this session)")
        else:
            # The unverified case: gross says win, measurement says nothing.
            # Report the floor, and the ceiling as a claim rather than a fact.
            print(f"  Net time saved:      at least {_fmt_signed_time(net_saved)}, "
                  f"at best {_fmt_signed_time(net_upper)}")
            print(f"    Cash measured only the {_fmt_time(overhead)} it spent. The "
                  f"{_fmt_time(gross_saved)} it avoided is what these values cost")
            print("    when first cached; if they would recompute faster today "
                  "(warm file cache,")
            print("    warm imports), the real figure is nearer the low end. Time "
                  "a run with")
            print("    caching off to settle it.")
        print()
        tracked = len(self._tracking_state.variable_lineage)
        print(f"  Tracked variables:   {tracked}")
        print()
        # Points at the CLI, not at "%cash_admin": that magic has never
        # existed. Sending a user who is looking at a multi-hundred-MB .cash
        # to a UsageError is worse than saying nothing, and inspecting the
        # backend is exactly what they came here to do.
        print("  Cache size / entries are not shown here to keep this command")
        print("  cheap; run `cash info` (or `cash clear`) in a terminal.")

    # ------------------------------------------------------------------
    # Export / import / diff
    # ------------------------------------------------------------------

    @line_magic
    def cash_export(self: CashMagics, line: str) -> None:
        """Export cache entries to a portable file.

        Usage::

            %cash_export results.cache              # Export all cache (pickle)
            %cash_export results.cache --vars x,y,z # Export specific variables
            %cash_export results.json --json         # Export lineage as JSON
        """
        parts = strip_inline_comment(line).split()
        if not parts:
            print("Usage: %cash_export <filename> [--vars x,y,z] [--json]")
            return

        filepath = parts[0]
        use_json = '--json' in parts

        export_vars = None
        if '--vars' in parts:
            idx = parts.index('--vars')
            if idx + 1 < len(parts):
                export_vars = set(parts[idx + 1].split(','))

        try:
            if use_json:
                self._export_json(filepath, export_vars)
            else:
                self._export_pickle(filepath, export_vars)
        except (OSError, TypeError, pickle.PicklingError, ValueError) as e:
            print(f"[Error] Export failed: {e}")

    def _export_json(self: CashMagics, filepath: str, export_vars: set | None) -> None:
        """Export lineage metadata as JSON."""
        import json

        export_data: dict[str, Any] = {'version': 1, 'lineage': {}, 'cell_codes': {}}

        if export_vars:
            for var in export_vars:
                if var in self._tracking_state.variable_lineage:
                    export_data['lineage'][var] = self._tracking_state.variable_lineage[var]
                if var in self._tracking_state.executed_cell_codes:
                    export_data['cell_codes'][var] = self._tracking_state.executed_cell_codes[var]
        else:
            export_data['lineage'] = dict(self._tracking_state.variable_lineage)
            export_data['cell_codes'] = dict(self._tracking_state.executed_cell_codes)

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"[OK] Exported lineage for {len(export_data['lineage'])} variables to '{filepath}' (JSON)")

    def _export_pickle(self: CashMagics, filepath: str, export_vars: set | None) -> None:
        """Export full cache entries as pickle."""
        import pickle as pkl

        backend = self._cash_instance.backend
        entries = backend.list_entries()
        export_data: dict[str, Any] = {
            'version': 1, 'entries': [], 'lineage': {}, 'cell_codes': {},
        }

        exported_count = 0
        for entry in entries:
            key = entry.get('key', '')
            if export_vars:
                outputs = entry.get('outputs', [])
                if not any(v in export_vars for v in outputs) and not any(v in key for v in export_vars):
                    continue
            meta, value = backend.get(key)
            if value is not None:
                export_data['entries'].append({'key': key, 'metadata': meta, 'value': value})
                exported_count += 1

        if export_vars:
            for var in export_vars:
                if var in self._tracking_state.variable_lineage:
                    export_data['lineage'][var] = self._tracking_state.variable_lineage[var]
                if var in self._tracking_state.executed_cell_codes:
                    export_data['cell_codes'][var] = self._tracking_state.executed_cell_codes[var]
        else:
            export_data['lineage'] = dict(self._tracking_state.variable_lineage)
            export_data['cell_codes'] = dict(self._tracking_state.executed_cell_codes)

        with open(filepath, 'wb') as f:
            pkl.dump(export_data, f)

        print(f"[OK] Exported {exported_count} cache entries to '{filepath}'")

    @line_magic
    def cash_import(self: CashMagics, line: str) -> None:
        """Import cache entries from a portable file.

        Usage::

            %cash_import results.cache           # Import all entries
            %cash_import results.cache --merge    # Merge with existing cache
        """
        import pickle as pkl

        parts = strip_inline_comment(line).split()
        if not parts:
            print("Usage: %cash_import <filename> [--merge]")
            return

        filepath = parts[0]
        merge_mode = '--merge' in parts
        backend = self._cash_instance.backend

        try:
            with open(filepath, 'rb') as f:
                import_data = pkl.load(f)

            if import_data.get('version', 0) != 1:
                print(f"[Warning] Unknown export format version: {import_data.get('version', 0)}")

            imported_count, skipped_count = _import_entries(backend, import_data.get('entries', []), merge_mode)
            self._import_metadata(import_data, merge_mode)

            msg = f"[OK] Imported {imported_count} cache entries from '{filepath}'"
            if skipped_count:
                msg += f" (skipped {skipped_count} existing)"
            print(msg)

        except FileNotFoundError:
            print(f"[Error] File not found: '{filepath}'")
        except (OSError, TypeError, ValueError, pickle.UnpicklingError, KeyError) as e:
            print(f"[Error] Import failed: {e}")

    def _import_metadata(self: CashMagics, import_data: dict, merge_mode: bool) -> None:
        """Import lineage and cell-code metadata from an export dict."""
        for var, lin_hash in import_data.get('lineage', {}).items():
            if not merge_mode or var not in self._tracking_state.variable_lineage:
                self._tracking_state.variable_lineage[var] = lin_hash
        for var, code in import_data.get('cell_codes', {}).items():
            if not merge_mode or var not in self._tracking_state.executed_cell_codes:
                self._tracking_state.executed_cell_codes[var] = code

    @line_magic
    def cash_diff(self: CashMagics, line: str) -> None:
        """Compare current cache state with another exported cache file.

        Usage::

            %cash_diff other_session.cache       - Compare with exported cache
            %cash_diff other_session.cache --vars - Show variable-level diff
        """
        args = strip_inline_comment(line).split()
        if not args:
            print("Usage: %cash_diff <cache_file> [--vars]")
            return

        filepath = args[0]
        show_vars = '--vars' in args

        try:
            other_data = _load_cache_file_data(filepath)
            if other_data is None:
                print(f"[Error] Invalid cache file format: '{filepath}'")
                return

            other_lineage = other_data.get('lineage', {})
            current_lineage = dict(self._tracking_state.variable_lineage)
            only_current, only_other, changed, identical = _compute_lineage_diff(current_lineage, other_lineage)

            print(f"Cache Diff: current session vs '{filepath}'")
            print(f"{'='*50}")
            print(f"  Only in current session: {len(only_current)}")
            print(f"  Only in '{filepath}':    {len(only_other)}")
            print(f"  Changed (diff lineage):  {len(changed)}")
            print(f"  Identical:               {len(identical)}")

            if show_vars:
                _print_diff_details(only_current, only_other, changed, identical)

            print(f"{'='*50}")

        except FileNotFoundError:
            print(f"[Error] File not found: '{filepath}'")
        except (OSError, TypeError, ValueError, KeyError) as e:
            print(f"[Error] Diff failed: {e}")

    # ------------------------------------------------------------------
    # Module tracking
    # ------------------------------------------------------------------

    @line_magic
    def cash_track(self: CashMagics, line: str) -> None:
        """Track a local Python module for changes.

        Usage::

            %cash_track my_helpers         - Track my_helpers module
            %cash_track my_helpers --reload - Force reload now
            %cash_track --list             - List tracked modules
            %cash_track --check            - Check for changes
        """
        parts = strip_inline_comment(line).split()
        ft = self._statement_processor.function_tracker

        if not parts or parts[0] == '--list':
            tracked = ft._tracked_modules
            if not tracked:
                print("No modules tracked. Use: %cash_track module_name")
            else:
                print("Tracked modules:")
                for mod in sorted(tracked):
                    mtime = ft._module_mtimes.get(mod, 'unknown')
                    print(f"  {mod} (mtime: {mtime})")
            return

        if parts[0] == '--check':
            _check_tracked_modules(ft)
            return

        _track_or_import_module(ft, parts[0], '--reload' in parts)

    # ------------------------------------------------------------------
    # Structured logging
    # ------------------------------------------------------------------

    @line_magic
    def cash_log(self: CashMagics, line: str) -> None:
        """View recent structured log events.

        Usage::

            %cash_log              - Show last 20 events
            %cash_log 50           - Show last 50 events
            %cash_log clear        - Clear log buffer
            %cash_log json         - Output as JSON array
        """
        parts = strip_inline_comment(line).split()

        handler = self._find_cash_log_handler()
        if handler is None:
            print("No log handler active. Use '%cash_debug on' first.")
            return

        if parts and parts[0] == 'clear':
            handler.clear()
            print("Log buffer cleared.")
            return

        limit, as_json = _parse_log_args(parts)
        events = handler.get_events(limit=limit)
        if not events:
            print("No log events recorded.")
            return

        if as_json:
            import json as _json
            print(_json.dumps(events, indent=2, default=str))
        else:
            for evt in events:
                print(_format_log_event(evt))

    def _find_cash_log_handler(self: CashMagics) -> Any | None:
        """Return the active CashLogHandler, or None if not found."""
        handler = getattr(self, '_log_handler', None)
        if handler is None:
            from ...logging import CashLogHandler
            cash_logger = logging.getLogger("cash")
            for h in cash_logger.handlers:
                if isinstance(h, CashLogHandler):
                    return h
        return handler

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    @line_magic
    def cash_provenance(self: CashMagics, line: str) -> None:
        """Show provenance (computation history) for a variable.

        Usage::

            %cash_provenance x           - Show how 'x' was computed
            %cash_provenance x --graph   - Include dependency graph
            %cash_provenance x --json    - Output as JSON
            %cash_provenance --all       - List all tracked variables
            %cash_provenance --clear     - Clear provenance data
        """
        parts = strip_inline_comment(line).split()

        if not parts or parts[0] == '--all':
            tracked = sorted(self._session.provenance.tracked_variables)
            if not tracked:
                print("No provenance data recorded yet.")
            else:
                print(f"Tracked variables ({len(tracked)}):")
                for var in tracked:
                    latest = self._session.provenance.get_latest(var)
                    status_icon = {"computed": "[C]", "restored": "[R]", "skipped": "[S]"}.get(
                        latest.status if latest else "", "[?]")
                    history_count = len(self._session.provenance.get_history(var))
                    print(f"  {status_icon} {var} ({history_count} records)")
            return

        if parts[0] == '--clear':
            self._session.provenance.clear()
            print("Provenance data cleared.")
            return

        var_name = parts[0]
        show_graph = '--graph' in parts
        show_time = '--time' in parts or '--timeline' in parts
        as_json = '--json' in parts

        if as_json:
            print(self._session.provenance.to_json(var_name))
        else:
            print(safe_text(self._session.provenance.format_provenance(
                var_name,
                show_graph=show_graph,
                show_timeline=show_time,
            )))

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    @line_magic
    def cash_audit(self: CashMagics, line: str) -> None:
        """Manage audit logging for cache operations.

        Usage::

            %cash_audit on                   - Enable audit logging (in-memory)
            %cash_audit on --file audit.log  - Enable with file output
            %cash_audit off                  - Disable audit logging
            %cash_audit show                 - Show recent audit entries
            %cash_audit summary              - Show summary statistics
            %cash_audit clear                - Clear audit entries
        """
        parts = strip_inline_comment(line).split()
        if not parts:
            status = "enabled" if self._session.audit.enabled else "disabled"
            print(f"Audit logging: {status}")
            return

        cmd = parts[0].lower()

        if cmd == 'on':
            self._audit_cmd_on(parts)
        elif cmd == 'off':
            self._session.audit.disable()
            print("Audit logging disabled.")
        elif cmd == 'show':
            self._audit_cmd_show(parts)
        elif cmd == 'summary':
            self._audit_cmd_summary()
        elif cmd == 'clear':
            self._session.audit.clear()
            print("Audit entries cleared.")
        else:
            print(f"Unknown audit command: {cmd}")
            print("Usage: %cash_audit on|off|show|summary|clear")

    def _audit_cmd_on(self: CashMagics, parts: list[str]) -> None:
        """Handle '%cash_audit on [--file path]'."""
        file_path = None
        if '--file' in parts:
            idx = parts.index('--file')
            if idx + 1 < len(parts):
                file_path = parts[idx + 1]
        self._session.audit.enable(file_path)
        msg = "Audit logging enabled"
        if file_path:
            msg += f" (logging to {file_path})"
        print(msg)

    def _audit_cmd_show(self: CashMagics, parts: list[str]) -> None:
        """Handle '%cash_audit show [operation] [--json]'."""
        operation = None
        as_json = '--json' in parts
        for p in parts[1:]:
            if not p.startswith('--'):
                operation = p
                break
        entries = self._session.audit.get_entries(operation=operation, limit=50)
        print(self._session.audit.format_entries(entries, as_json=as_json))

    def _audit_cmd_summary(self: CashMagics) -> None:
        """Handle '%cash_audit summary'."""
        summary = self._session.audit.get_summary()
        if summary['total'] == 0:
            print("No audit entries recorded.")
            return
        print("Audit Summary:")
        print(f"  Total entries: {summary['total']}")
        print(f"  Unique variables: {summary['unique_variables']}")
        print(f"  Time range: {summary['time_range']}")
        print("  Operations:")
        for op, count in sorted(summary.get('operations', {}).items()):
            print(f"    {op}: {count}")

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    @line_magic
    def cash_benchmark(self: CashMagics, line: str) -> None:
        """Benchmark cache performance for a cell or statement.

        Usage::

            %cash_benchmark             - Benchmark the next cell (runs 3 times)
            %cash_benchmark 5           - Run 5 iterations
            %cash_benchmark --cold      - Clear cache before each run (cold start)
            %cash_benchmark --compare   - Compare cached vs uncached execution
        """
        parts = strip_inline_comment(line).split()

        iterations = 3
        cold_start = '--cold' in parts
        compare_mode = '--compare' in parts

        for p in parts:
            if p.isdigit():
                iterations = max(1, min(int(p), 100))
                break

        self._benchmark_config = {
            'iterations': iterations,
            'cold_start': cold_start,
            'compare_mode': compare_mode,
            'active': True,
        }
        print(f"[Benchmark] Mode enabled for next cell ({iterations} iterations"
              f"{', cold start' if cold_start else ''}"
              f"{', compare mode' if compare_mode else ''})")

    def _run_benchmark(self: CashMagics, cell_code: str, iterations: int, cold_start: bool, compare_mode: bool) -> None:
        """Execute a benchmark run and report results."""
        import statistics

        # Use perf_counter, not time.time — the latter has ~16ms resolution
        # on Windows and produces zero-duration measurements for fast cells.
        uncached_times: list[float] = []
        if compare_mode:
            # The "without caching" arm must genuinely recompute on EVERY
            # iteration. ``self.shell.run_cell`` is the cash-PATCHED entry point
            # (installed at init, it dispatches to ``_execute_cell`` — cash's
            # full caching pipeline), so running the cell through it stores the
            # result on the first iteration and then RESTORES it from cache on
            # every later one. That measured cache-hit-vs-cache-hit and reported
            # a meaningless ~1x "speedup" on a workload that was really ~170x
            # faster to restore than recompute (CAS-168).
            #
            # ``self._original_run_cell`` is the real, unpatched IPython
            # ``run_cell`` captured *before* cash installed its hook, so it
            # executes the user code as if cash were not present: a true
            # recompute each iteration, and the honest "without caching"
            # baseline.
            for _i in range(iterations):
                start = time.perf_counter()
                self._original_run_cell(cell_code, silent=True)
                uncached_times.append(time.perf_counter() - start)

            # The uncached arm bypassed cash entirely, so nothing was stored in
            # cash's cache. Warm it once (untimed) so the "with caching" arm
            # below measures a genuine cache HIT rather than a first-run store.
            # Skipped under --cold, whose contract is to clear the cache before
            # each timed iteration (i.e. deliberately measure the miss path).
            if not cold_start:
                self._execute_cell(cell_code)

        cached_times: list[float] = []
        for _i in range(iterations):
            if cold_start:
                self._clear_cache_for_cold_start()
            start = time.perf_counter()
            self._execute_cell(cell_code)
            cached_times.append(time.perf_counter() - start)

        print(f"\n{'='*50}")
        print(f"Benchmark Results ({iterations} iterations)")
        print(f"{'='*50}")
        _print_timing_section("With caching", cached_times, statistics)
        _print_timing_section("Without caching", uncached_times, statistics)

        if uncached_times and cached_times:
            mean_cached = statistics.mean(cached_times)
            mean_uncached = statistics.mean(uncached_times)
            if mean_uncached > 0 and mean_cached > 0:
                speedup = mean_uncached / mean_cached
                savings_pct = (1 - mean_cached / mean_uncached) * 100
                print(f"\n  Speedup: {speedup:.1f}x ({savings_pct:.0f}% faster with caching)")
            else:
                # Both runs measured below timer resolution.  Tell the user the
                # result is unreliable rather than silently omit the line.
                print("\n  Speedup: n/a (timings below timer resolution)")

        print(f"{'='*50}")

    def _clear_cache_for_cold_start(self: CashMagics) -> None:
        """Clear cache and lineage state for a cold-start benchmark iteration."""
        if hasattr(self, '_backend') and self._backend:
            try:
                self._backend.clear()
            except (OSError, AttributeError, TypeError):
                logger.debug("Failed to clear cache for cold start benchmark")
        self._tracking_state.variable_lineage.clear()
        self._tracking_state.executed_cell_codes.clear()


# ---------------------------------------------------------------------------
# Module-level helpers (used by magic methods above, but stateless)
# ---------------------------------------------------------------------------

def _import_entries(backend: Any, entries: list, merge_mode: bool) -> tuple[int, int]:
    """Store entries into backend, respecting merge_mode. Returns (imported, skipped)."""
    imported = 0
    skipped = 0
    for entry in entries:
        key = entry['key']
        meta = entry['metadata']
        value = entry['value']
        if not merge_mode:
            backend.set(key, value, meta)
            imported += 1
        else:
            _, existing_val = backend.get(key)
            if existing_val is None:
                backend.set(key, value, meta)
                imported += 1
            else:
                skipped += 1
    return imported, skipped


def _compute_lineage_diff(
    current: dict[str, str], other: dict[str, str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return (only_current, only_other, changed, identical) sets."""
    current_vars = set(current)
    other_vars = set(other)
    only_current = current_vars - other_vars
    only_other = other_vars - current_vars
    common = current_vars & other_vars
    changed = {v for v in common if current.get(v) != other.get(v)}
    identical = common - changed
    return only_current, only_other, changed, identical


def _print_diff_details(
    only_current: set[str], only_other: set[str], changed: set[str], identical: set[str],
) -> None:
    """Print variable-level diff details."""
    if only_current:
        print(f"\n  [+] Only in current: {', '.join(sorted(only_current))}")
    if only_other:
        print(f"  [-] Only in other:   {', '.join(sorted(only_other))}")
    if changed:
        print(f"  [~] Changed:         {', '.join(sorted(changed))}")
    if identical:
        print(f"  [=] Identical:       {', '.join(sorted(identical))}")


def _check_tracked_modules(ft: Any) -> None:
    """Check tracked modules for changes and reload any that changed."""
    changed = ft.check_tracked_modules()
    if changed:
        print(f"Changed modules: {', '.join(sorted(changed))}")
        for mod in changed:
            ft.reload_module(mod)
            print(f"  Reloaded: {mod}")
    else:
        print("No tracked modules have changed.")


def _track_or_import_module(ft: Any, module_name: str, force_reload: bool) -> None:
    """Track a module by name, importing it first if necessary."""
    file_path = ft.track_module(module_name)
    if file_path:
        print(f"Tracking module '{module_name}' ({file_path})")
        if force_reload:
            if ft.reload_module(module_name):
                print(f"  Reloaded: {module_name}")
            else:
                print(f"  Reload failed for: {module_name}")
        return

    import importlib
    try:
        importlib.import_module(module_name)
        file_path = ft.track_module(module_name)
        if file_path:
            print(f"Tracking module '{module_name}' ({file_path})")
        else:
            print(f"[Warning] Module '{module_name}' imported but has no file (built-in?)")
    except ImportError:
        print(f"[Error] Module '{module_name}' not found")


def _parse_log_args(parts: list[str]) -> tuple[int, bool]:
    """Parse %cash_log arguments, returning (limit, as_json)."""
    limit = 20
    as_json = False
    for p in parts:
        if p == 'json':
            as_json = True
        elif p.isdigit():
            limit = int(p)
    return limit, as_json


def _format_log_event(evt: dict) -> str:
    """Format a single log event dict as a printable string."""
    ts = time.strftime('%H:%M:%S', time.localtime(evt['time']))
    level = evt['level'][:4]
    msg = evt['msg']
    extra = ''
    if 'event' in evt:
        extra += f" [{evt['event']}]"
    if 'duration_ms' in evt:
        extra += f" ({evt['duration_ms']:.1f}ms)"
    return f"  {ts} {level}{extra} {msg}"


def _print_timing_section(label: str, times: list[float], statistics: Any) -> None:
    """Print benchmark timing statistics for one run series."""
    if not times:
        return
    mean = statistics.mean(times)
    if len(times) > 1:
        stdev = statistics.stdev(times)
        print(f"  {label}:")
        print(f"    Mean:   {mean*1000:.1f}ms")
        print(f"    Stdev:  {stdev*1000:.1f}ms")
        print(f"    Min:    {min(times)*1000:.1f}ms")
        print(f"    Max:    {max(times)*1000:.1f}ms")
    else:
        print(f"  {label}:  {mean*1000:.1f}ms")
