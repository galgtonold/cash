# Magic Commands

Cash registers a suite of IPython magic commands that control caching, inspect
session state, and move cache data between sessions. This page is the canonical
reference for all **20** magics — each entry lists the exact signature, every
parsed flag, and a working example. Behaviour is derived directly from
`src/cash/notebook/ipython/magics.py` and `src/cash/notebook/ipython/admin.py`.

<!-- The count above, and the completeness of the table and the per-magic
     sections below, are checked against the registered decorators by
     tests/docs/test_doc_claims.py::test_every_registered_magic_is_documented
     and ::test_magics_page_states_the_right_count. Each section additionally
     carries a claim anchor pinned to the method that implements it, so a
     changed signature or flag surfaces in the re-verification queue. -->

## At a glance

| Magic | Purpose |
|-------|---------|
| [`%cash_on`](#cash_on) | Enable automatic caching for subsequent cells. |
| [`%cash_off`](#cash_off) | Disable automatic caching. |
| [`%cash_persist`](#cash_persist) | Cache *every* statement, bypassing the cost-aware floors. |
| [`%cash_help`](#cash_help) | Print a quick-reference card (optionally per topic). |
| [`%cash_feedback`](#cash_feedback) | Show bug-report and feedback URLs. |
| [`%cash_status`](#cash_status) | Inspect the last cell + session as dict / JSON. |
| [`%cash_badge`](#cash_badge) | Set per-cell badge mode (html/print/off). |
| [`%cash_stats`](#cash_stats) | Session-wide cache statistics. |
| [`%cash_debug`](#cash_debug) | Toggle / configure debug logging. |
| [`%%cash`](#cash-cell) | Cache a single cell explicitly. |
| [`%cash_verify`](#cash_verify) | Audit cache integrity, optionally delete corrupted entries. |
| [`%cash_repair`](#cash_repair) | Recover from corruption or reset tracking state. |
| [`%cash_provenance`](#cash_provenance) | Variable computation history. |
| [`%cash_track`](#cash_track) | Watch a local module for source changes. |
| [`%cash_diff`](#cash_diff) | Diff current session against an exported cache file. |
| [`%cash_log`](#cash_log) | View / clear the in-memory structured log buffer. |
| [`%cash_audit`](#cash_audit) | Manage the cache-operation audit log. |
| [`%cash_export`](#cash_export) | Serialize cache (and/or lineage) to a file. |
| [`%cash_import`](#cash_import) | Load cache from a file written by `%cash_export`. |
| [`%cash_benchmark`](#cash_benchmark) | Arm the next cell to run N timed iterations. |

---

## Enabling and configuring (user-facing)

### `%cash_on`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_on @656d4606 -->

Enable automatic caching for every subsequent cell.

**Signature:** `%cash_on [ttl=N]`

**Arguments:**

- `ttl=N` — *Optional.* Default TTL in seconds applied to every cached
  statement. Must be an integer; invalid values log a warning and the command
  returns without enabling caching.

**Side effects:**

- Invalidates the notebook-path discovery cache so the current notebook is
  re-discovered on the next cell (fixes notebook-switch within a kernel).
- Resets the upstream checker's AST and simulation caches.
- Sets the global TTL (`None` if not supplied) and flips
  `_auto_cache_enabled = True`.

**Examples:**

```python
%cash_on
%cash_on ttl=3600
```

### `%cash_off`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_off @b6796687 -->

Disable automatic caching. Subsequent cells run uncached until you call
`%cash_on` again.

**Signature:** `%cash_off`

**Arguments:** None.

**Side effects:**

- Clears `_auto_cache_enabled` and the global TTL.

**Example:**

```python
%cash_off
```

### `%cash_persist`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_persist @0197157d -->

Cache *every* statement regardless of how cheap it was to compute — equivalent
to putting `# @cash:persist` on every statement. It bypasses the cost-aware
floors (the 10 ms "too cheap to cache" floor and the size-aware skip). Useful
for reproducibility, benchmarks, and debugging cache behaviour; wasteful for
trivial statements in normal use.

**Signature:** `%cash_persist [on|off]`

**Arguments:**

- `on` — cache every statement.
- `off` — restore the default cost-aware policy.
- *(no argument)* — toggle the current state.

**Example:**

```python
%cash_persist on     # every statement is now cached
%cash_persist off    # back to the cost-aware default
```

### `%cash_help`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_help @7290a5b1 -->

Print a quick-reference card. With no argument the main card is shown; pass a
topic name to drill into a sub-card.

**Signature:** `%cash_help [topic]`

**Arguments:**

- *(no argument)* — Main quick-reference (essential commands, cache management,
  module tracking, annotation syntax). See [Annotations](annotations.md) for the
  full `@cash:` directive reference.
- `badge` / `badges` — Badge display modes and status icons.
- `debug` / `debugging` — Debug logging and verification commands.
- `collab` / `collaboration` / `sharing` — Export / import / diff commands.
- `inspect` / `provenance` / `audit` — Status, stats, provenance, audit, log.

**Example:**

```python
%cash_help
%cash_help badge
%cash_help inspect
```

### `%cash_feedback`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_feedback @ae5c76bf -->

Print bug-report and feedback URLs. Useful pointer for beta users.

**Signature:** `%cash_feedback`

**Arguments:** None.

**Example:**

```python
%cash_feedback
```

### `%cash_status`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_status @29029be3 -->

Report status of the last cell plus a snapshot of session state (lineage,
executed-code map, auto-cache flag, backend stats).

**Signature:** `%cash_status [mode]`

**Arguments:**

- *(no argument)* — Pretty-print the status as JSON **and** return the dict.
- `dict` — Return the status as a Python `dict` (no print).
- `json` — Return the status as a JSON-serialized string.

**Returns:** dict (default and `dict` mode) or JSON string (`json` mode).

**Example:**

```python
%cash_status               # prints and returns a dict
status = %cash_status dict # capture as dict
blob   = %cash_status json # capture as JSON string
```

### `%cash_badge`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_badge @6bc4c33b -->

Set the badge display mode for subsequent cached cells. See
[Reading the Cash Badge](badges.md) for the full anatomy of each mode.

**Signature:** `%cash_badge [mode]`

**Arguments:**

- *(no argument)* — Print the current badge mode and usage hint; no change.
- `html` — Interactive HTML badges with live progress updates (default).
- `print` — Text summary printed once after the cell completes.
- `off` — No badge output at all.

**Example:**

```python
%cash_badge print
%cash_badge off
%cash_badge          # show current mode
```

### `%cash_stats`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @851d8352 -->

Show session-wide cache statistics: counts, hit rate, compute time, and the
savings broken out as **gross saved**, **cash overhead**, and **net saved**,
plus tracked variables. The headline net is `verified saved − overhead` — the
subset of savings *this* session re-measured itself, not the full gross, so a
stale first-run timing can never inflate it. (`gross − overhead` is reported
separately as an upper bound.) Reporting net keeps the headline honest:
cash's own per-cell overhead is subtracted from the recompute it avoided, and a session whose overhead outweighs its hits reads as
a plain "cash cost you Xs this session" rather than a phantom win. The command
deliberately avoids walking the backend so it stays cheap on large on-disk
caches.

**Signature:** `%cash_stats [mode]`

**Arguments:**

- *(no argument)* — Human-readable summary printed to stdout.
- `json` — Pretty-print as JSON (includes `total_overhead`, `net_time_saved`,
  and `hit_rate_percent`).
- `reset` — Zero out the in-memory counters (`cells_executed`,
  `statements_computed`, `statements_restored`, `statements_skipped`,
  `total_compute_time`, `total_restored_time`, `total_time_saved`,
  `total_overhead`).

**Example:**

```python
%cash_stats
%cash_stats json
%cash_stats reset
```

### `%cash_debug`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @ce13e22b -->

Toggle or configure debug logging. Levels propagate to the global `cash` logger
and to the statement processor, upstream checker, and core Cash instance.

**Signature:** `%cash_debug [on|off|json|file <path>]`

**Arguments:**

- *(no argument)* — Toggle debug on/off.
- `on` (aliases: `true`, `1`, `enable`) — Enable debug logging at DEBUG level.
- `off` (aliases: `false`, `0`, `disable`) — Disable debug logging (INFO level).
- `json` — Enable DEBUG with a JSON-formatted log handler (powers
  [`%cash_log`](#cash_log)).
- `file <path>` — Enable DEBUG and also log to `<path>` in JSON format.

Argument matching is case-insensitive.

**Example:**

```python
%cash_debug on
%cash_debug off
%cash_debug json
%cash_debug file /tmp/cash.log
```

### `%%cash` { #cash-cell }
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash @42d8d6c6 -->

Cell magic. Explicitly cache a single cell with the same statement-level
processing as `%cash_on` (upstream simulation, file tracking, badge rendering).

**Signature:**

```python
%%cash [ttl=N]
<cell body>
```

**Arguments:**

- `ttl=N` — *Optional.* TTL in seconds for statements computed in this cell.

**Behaviour:**

- Parses the cell with `ast.parse` and processes each top-level statement
  individually. A `SyntaxError` aborts the cell early.
- Runs upstream simulation against the on-disk notebook (so unsaved upstream
  edits won't be seen — save before running).
- Renders the configured badge (`html` / `print` / off — see
  [`%cash_badge`](#cash_badge)).
- Caches cells that use top-level `await` on the same terms as any other cell
  (see below).

**Example:**

<!-- test:skip reason="illustrative — references missing big.csv; stub pandas returns dict without .groupby" -->
```python
%%cash ttl=600
import pandas as pd
df = pd.read_csv("big.csv")
agg = df.groupby("region").sum()
```

**Top-level `await`:** ipykernel dispatches a cell containing top-level `await`
through `shell.run_cell_async`, not the `pre_run_cell` hook that `%cash_on`
patches. Cash intercepts that entry point as well, so awaited cells get lineage
tracking, upstream reset, and result caching — the async pipeline is the
line-for-line twin of the sync one. A cache hit returns before the coroutine is
built, so an unchanged re-run skips the `await` rather than re-issuing the call.

---

## Admin and integrity

### `%cash_verify`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_verify @ced721ff -->

Audit cache integrity by reading every entry in the backend. Reports total,
healthy, corrupted, and the first 20 issues. Also notes lineage entries whose
variables are no longer in the namespace.

**Signature:** `%cash_verify [--fix]`

**Arguments:**

- *(no argument)* — Read-only audit; suggests running `--fix` if issues are
  found.
- `--fix` — Call `backend.delete(key)` on every corrupted entry.

**Note:** every built-in backend implements `list_entries`, so this works
everywhere out of the box. A custom third-party backend that cannot list its
entries makes the command print `[Error] Error accessing backend: ...` and
return.

**Example:**

```python
%cash_verify
%cash_verify --fix
```

### `%cash_repair`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_repair @2cfd33e0 -->

Recover from corruption or reset tracking state. Three modes with different
levels of aggression.

**Signature:** `%cash_repair [--full|--state]`

**Arguments:**

- *(no argument)* — Default: run `%cash_verify --fix`, then prune lineage
  entries whose variables are no longer in `shell.user_ns`.
- `--state` — Clear all in-memory tracking state (variable lineage, executed
  cell codes/hashes, file deps, mutation lineage, variable hashes/sources,
  current-session hashes, raw cell codes). **Cache backend is preserved.**
- `--full` — Clear the backend (`backend.clear()`) **and** all in-memory
  state.

After `--state` or `--full` you must re-run upstream cells to rebuild lineage.

**Example:**

```python
%cash_repair
%cash_repair --state
%cash_repair --full
```

---

## Inspecting your session

### `%cash_provenance`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_provenance @c880acac -->

Show how a variable was computed: its lineage hash, the cell code that produced
it, and (optionally) a dependency graph or timeline.

**Signature:** `%cash_provenance [<var>|--all|--clear] [--graph] [--time|--timeline] [--json]`

**Arguments:**

- *(no argument)* / `--all` — List every tracked variable with a status icon
  (`[C]` computed, `[R]` restored, `[S]` skipped, `[?]` unknown) and the number
  of records in its history.
- `--clear` — Clear all provenance records.
- `<var>` — Detail view for one variable. Combinable flags:
  - `--graph` — Include the dependency graph.
  - `--time` / `--timeline` — Include a timeline of computations.
  - `--json` — Output as JSON (mutually exclusive with the human-readable
    formatter; if `--json` is set neither `--graph` nor `--time` are honoured).

**Example:**

```python
%cash_provenance              # list all tracked vars
%cash_provenance --all
%cash_provenance df           # detail for `df`
%cash_provenance df --graph --time
%cash_provenance df --json
%cash_provenance --clear
```

### `%cash_track`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_track @0665b4b1 -->

Watch a local Python module for source changes; reload it (and invalidate any
cache entries that depend on it) when it changes.

**Signature:** `%cash_track [<module>|--list|--check] [--reload]`

**Arguments:**

- *(no argument)* / `--list` — List tracked modules with their last-seen
  `mtime`. Prints a usage hint if no modules are tracked.
- `--check` — Re-check every tracked module and reload any that changed.
- `<module>` — Start tracking `<module>`. If the module isn't already
  imported, the magic calls `importlib.import_module(<module>)` first.
- `--reload` — Combined with `<module>`: force-reload it now in addition to
  starting tracking.

**Example:**

```python
%cash_track my_helpers
%cash_track my_helpers --reload
%cash_track --list
%cash_track --check
```

### `%cash_diff`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_diff @8d07650d -->

Compare the current session's lineage with a cache file produced by
[`%cash_export`](#cash_export). Reports per-bucket counts (only-current,
only-other, changed, identical) and optionally per-variable detail.

**Signature:** `%cash_diff <cache_file> [--vars]`

**Arguments:**

- `<cache_file>` — *Required, positional.* Path to a `%cash_export` file.
  Auto-detects JSON first, then pickle.
- `--vars` — Print per-variable detail for each diff bucket.

**Example:**

```python
%cash_diff teammate_session.cache
%cash_diff teammate_session.cache --vars
```

### `%cash_log`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_log @ce466107 -->

View the in-memory structured log buffer (CashLogHandler). Requires a JSON
handler installed via [`%cash_debug json`](#cash_debug) (or
`%cash_debug file <path>`); otherwise the command prints a hint and returns.

**Signature:** `%cash_log [<N>|json|clear]`

**Arguments:**

- *(no argument)* — Show the last 20 events.
- `<N>` (integer) — Show the last `N` events.
- `json` — Output the events as a JSON array.
- `clear` — Clear the log buffer.

**Example:**

```python
%cash_debug json     # arm the handler first
%cash_log
%cash_log 50
%cash_log json
%cash_log clear
```

### `%cash_audit`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_audit @42b61658 -->

Manage audit logging of cache operations (hits, misses, computes, restores).
Audit entries are kept in memory by default and optionally mirrored to a file.

**Signature:** `%cash_audit [on [--file <path>]|off|show [<operation>] [--json]|summary|clear]`

**Arguments:**

- *(no argument)* — Print `Audit logging: enabled` or `disabled`.
- `on` — Enable audit logging. Optional `--file <path>` mirrors entries to
  a file as well.
- `off` — Disable audit logging.
- `show` — Show the last 50 entries. Optional positional `<operation>`
  filters by operation type (e.g. `cache_hit`). Optional `--json` prints as
  JSON.
- `summary` — Print totals: entry count, unique variables, time range, and
  per-operation counts.
- `clear` — Clear all audit entries.

**Example:**

```python
%cash_audit on
%cash_audit on --file audit.log
%cash_audit show
%cash_audit show cache_hit
%cash_audit show --json
%cash_audit summary
%cash_audit clear
%cash_audit off
```

---

## Moving data between sessions

### `%cash_export`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_export @1ce0818b -->

Serialize the cache to a file. Default is a pickle that contains backend
entries plus lineage and cell-code metadata; `--json` writes a lineage-only
JSON file suitable for [`%cash_diff`](#cash_diff).

**Signature:** `%cash_export <filename> [--vars x,y,z] [--json]`

**Arguments:**

- `<filename>` — *Required, positional.* Output path.
- `--vars x,y,z` — Comma-separated list of variable names to restrict the
  export to. Without it, all entries / all lineage are exported.
- `--json` — Write JSON (lineage + cell codes only, no backend values). The
  default format is pickle (full entries).

**Example:**

```python
%cash_export results.cache
%cash_export results.cache --vars df,agg,model
%cash_export lineage.json --json
```

### `%cash_import`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_import @1a4d7a1b -->

Load cache entries from a file written by `%cash_export`. **Pickle only** —
JSON exports are lineage-only and cannot be re-imported as cache entries.

**Signature:** `%cash_import <filename> [--merge]`

**Arguments:**

- `<filename>` — *Required, positional.* Path to a pickle export.
- `--merge` — Skip entries whose key already exists in the backend and lineage
  entries for variables already tracked. Without `--merge`, existing keys are
  overwritten.

Prints a warning if the export version isn't `1`.

**Example:**

```python
%cash_import results.cache
%cash_import results.cache --merge
```

---

## Benchmarking

### `%cash_benchmark`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_benchmark @ee658c4b -->

Arm the **next** cell to run N timed iterations. Does nothing on its own — the
next cell triggers the benchmark when it executes.

**Signature:** `%cash_benchmark [<N>] [--cold] [--compare]`

**Arguments:**

- `<N>` (integer) — *Optional.* Number of iterations. Defaults to `3` and is
  clamped to the range `[1, 100]`.
- `--cold` — Clear the cache before each cached iteration (measures cold-start
  time).
- `--compare` — Also run the cell `N` times **without** caching so the output
  includes a cached-vs-uncached speedup line.

Timing uses `time.perf_counter` (not `time.time`) to avoid Windows' ~16ms
clock resolution skewing fast cells. If both runs measure below timer
resolution, the speedup line reports `n/a`.

**Example:**

```python
%cash_benchmark               # 3 iterations, warm cache
%cash_benchmark 10
%cash_benchmark 5 --cold
%cash_benchmark 5 --compare
%cash_benchmark 10 --cold --compare
```

The arming pattern in practice:

<!-- test:skip reason="illustrative — references undefined pd and missing big.csv" -->
```python
%cash_benchmark 5 --compare
# next cell:
df = pd.read_csv("big.csv").groupby("region").sum()
```
