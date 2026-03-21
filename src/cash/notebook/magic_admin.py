"""Administrative magic commands extracted from CashMagics.

This module provides a mixin class with utility / diagnostic / benchmark
magic commands.  The mixin is inherited by :class:`~cash.notebook.magics.CashMagics`
so that IPython recognises these commands automatically.
"""

from __future__ import annotations

import logging
import pickle
import time
from typing import TYPE_CHECKING, Any

from IPython.core.magic import line_magic

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
    :class:`~cash.notebook.magics.CashMagics` instance (i.e. attributes such
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
            if not hasattr(backend, 'list_entries'):
                print("[Info] Backend does not support listing entries.")
                return
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
        mode = line.strip().lower() if line else ''

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

            print("\n[OK] Repair complete.")

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
        mode = line.strip().lower() if line else ''

        if mode == 'reset':
            self._session.stats.update({
                'cells_executed': 0,
                'statements_computed': 0,
                'statements_restored': 0,
                'statements_skipped': 0,
                'total_compute_time': 0.0,
                'total_restored_time': 0.0,
                'total_time_saved': 0.0,
            })
            print("[OK] Session statistics reset.")
            return

        stats = self._session.stats
        total_stmts = stats['statements_computed'] + stats['statements_restored'] + stats['statements_skipped']
        hit_rate = (stats['statements_restored'] + stats['statements_skipped']) / max(total_stmts, 1) * 100

        backend = self._cash_instance.backend
        try:
            entries = backend.list_entries()
            total_entries = len(entries)
            total_storage = sum(e.get('size', 0) for e in entries)
        except (AttributeError, TypeError, OSError):
            total_entries = 0
            total_storage = 0

        if mode == 'json':
            import json
            result = {
                **stats,
                'hit_rate_percent': round(hit_rate, 1),
                'cache_entries': total_entries,
                'storage_bytes': total_storage,
            }
            print(json.dumps(result, indent=2))
            return

        print("Cash Session Statistics")
        print("-" * 40)
        print(f"  Cells executed:      {stats['cells_executed']}")
        print(f"  Statements computed: {stats['statements_computed']}")
        print(f"  Statements restored: {stats['statements_restored']}")
        print(f"  Statements skipped:  {stats['statements_skipped']}")
        print(f"  Cache hit rate:      {hit_rate:.1f}%")
        print()
        print(f"  Compute time:        {_fmt_time(stats['total_compute_time'])}")
        print(f"  Time saved:          {_fmt_time(stats['total_time_saved'])}")
        print()
        print(f"  Cache entries:       {total_entries}")
        print(f"  Storage used:        {_fmt_size(total_storage)}")

        tracked = len(self._tracking_state.variable_lineage)
        print(f"  Tracked variables:   {tracked}")

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
        parts = line.strip().split()
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

        parts = line.strip().split()
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
        args = line.strip().split()
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
        parts = line.strip().split()
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
        parts = line.strip().split()

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
            from ..logging import CashLogHandler
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
        parts = line.strip().split()

        if not parts or parts[0] == '--all':
            tracked = sorted(self._provenance.tracked_variables)
            if not tracked:
                print("No provenance data recorded yet.")
            else:
                print(f"Tracked variables ({len(tracked)}):")
                for var in tracked:
                    latest = self._provenance.get_latest(var)
                    status_icon = {"computed": "[C]", "restored": "[R]", "skipped": "[S]"}.get(
                        latest.status if latest else "", "[?]")
                    history_count = len(self._provenance.get_history(var))
                    print(f"  {status_icon} {var} ({history_count} records)")
            return

        if parts[0] == '--clear':
            self._provenance.clear()
            print("Provenance data cleared.")
            return

        var_name = parts[0]
        show_graph = '--graph' in parts
        show_time = '--time' in parts or '--timeline' in parts
        as_json = '--json' in parts

        if as_json:
            print(self._provenance.to_json(var_name))
        else:
            print(self._provenance.format_provenance(
                var_name,
                show_graph=show_graph,
                show_timeline=show_time,
            ))

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
        parts = line.strip().split()
        if not parts:
            status = "enabled" if self._audit.enabled else "disabled"
            print(f"Audit logging: {status}")
            return

        cmd = parts[0].lower()

        if cmd == 'on':
            self._audit_cmd_on(parts)
        elif cmd == 'off':
            self._audit.disable()
            print("Audit logging disabled.")
        elif cmd == 'show':
            self._audit_cmd_show(parts)
        elif cmd == 'summary':
            self._audit_cmd_summary()
        elif cmd == 'clear':
            self._audit.clear()
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
        self._audit.enable(file_path)
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
        entries = self._audit.get_entries(operation=operation, limit=50)
        print(self._audit.format_entries(entries, as_json=as_json))

    def _audit_cmd_summary(self: CashMagics) -> None:
        """Handle '%cash_audit summary'."""
        summary = self._audit.get_summary()
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
        parts = line.strip().split()

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

        uncached_times: list[float] = []
        if compare_mode:
            for _i in range(iterations):
                start = time.time()
                self.shell.run_cell(cell_code, silent=True)
                uncached_times.append(time.time() - start)

        cached_times: list[float] = []
        for _i in range(iterations):
            if cold_start:
                self._clear_cache_for_cold_start()
            start = time.time()
            self._execute_cell(cell_code)
            cached_times.append(time.time() - start)

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
