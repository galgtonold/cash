# Magic Commands

Cash registers a small set of IPython magic commands that control caching and
inspect session state. This page is the canonical
reference for all **9** magics — each entry lists the exact signature, every
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
| [`%cash_help`](#cash_help) | List every magic, or print one magic's full usage. |
| [`%cash_status`](#cash_status) | Inspect the last cell + session as dict / JSON. |
| [`%cash_badge`](#cash_badge) | Set per-cell badge mode (html/print/off). |
| [`%cash_stats`](#cash_stats) | Session-wide cache statistics. |
| [`%cash_debug`](#cash_debug) | Toggle / configure debug logging. |
| [`%cash_provenance`](#cash_provenance) | Variable computation history. |

---

## Enabling and configuring (user-facing)

### `%cash_on`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_on @8cbb17b4 -->

Enable automatic caching for every subsequent cell.

**Signature:** `%cash_on [ttl=N]`

**Arguments:**

- `ttl=N` — *Optional.* Default TTL in seconds applied to every cached
  statement. Must be an integer; an invalid value — or any argument other than
  `ttl` — prints an error and the command returns **without** enabling caching.

With caching disabled (`CASH_DISABLE=1` or `disable=True`), `%cash_on` prints
that it did nothing and leaves every cell running uncached — see
[testing your code](tutorials/feature-guides/testing-your-code.md).

**Side effects:**

- Invalidates the notebook-path discovery cache so the current notebook is
  re-discovered on the next cell (fixes notebook-switch within a kernel).
- Resets the upstream checker's AST and simulation caches.
- Drops any cell snapshot cash's JupyterLab extension pushed, for the same
  reason: a snapshot of the *previous* notebook must not answer the new one's
  first upstream check.
- Sets the global TTL (`None` if not supplied) and flips
  `_auto_cache_enabled = True`.
- Prints a one-time `[Tip]` about saving before running, **unless** this is
  Colab or cash's JupyterLab extension is installed in this environment. Those
  two are the whole gate. A VS Code hot-exit backup is a live reader too and
  does **not** suppress the tip: whether a usable backup exists is a property of
  each individual read, not something `%cash_on` can settle up front, so VS Code
  users still see it. See
  [editing without saving](known-limitations.md#editing-without-saving).

**Examples:**

```python
%cash_on
%cash_on ttl=3600
```

#### Top-level `await` { #top-level-await }

ipykernel dispatches a cell containing top-level `await`
through `shell.run_cell_async`, not the `pre_run_cell` hook that `%cash_on`
patches. Cash intercepts that entry point as well, so awaited cells get lineage
tracking, upstream reset, and result caching — the async pipeline is the
line-for-line twin of the sync one. A cache hit returns before the coroutine is
built, so an unchanged re-run skips the `await` rather than re-issuing the call.

### `%cash_off`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_off @700a45e0 -->

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
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_persist @729a986b -->

Cache *every* statement regardless of how cheap it was to compute — equivalent
to putting `# @cash:persist` on every statement. It bypasses the cost-aware
floors (the 10 ms "too cheap to cache" floor and the size-aware skip). Useful
for reproducibility, benchmarks, and debugging cache behaviour; wasteful for
trivial statements in normal use. It sets the `persist_all` setting, the same
switch as `cash.configure(persist_all=True)`, and takes effect from the next
statement.

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
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_help @6f3ad8e6 -->
<!-- claim: cash/notebook/ipython/_help.py:help_text @800a29c8 -->

Print every registered magic with the first line of its docstring, a working
example of each `# @cash:` annotation, and the links for docs, bug reports and
questions. Pass a magic's name to print its full usage instead.

The card is built from the magics IPython actually registered and from their
docstrings, and each annotation example is checked against the real parser, so
it lists exactly the magics that exist.

**Signature:** `%cash_help [name]`

**Arguments:**

- *(no argument)* — The full card. See [Annotations](annotations.md) for the
  full `@cash:` directive reference.
- `name` — One magic's docstring: its usage and flags. The `%` and `cash_`
  prefixes are optional, so `badge`, `cash_badge` and `%cash_badge` all work.
  An unknown name says so and prints the full card.

**Example:**

```python
%cash_help
%cash_help badge
%cash_help stats
```

### `%cash_status`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_status @8a8b6dd0 -->

Report status of the last cell plus a snapshot of session state (lineage,
executed-code map, auto-cache flag, and the number of entries in the backend).

**Signature:** `%cash_status [mode]`

**Arguments:**

- *(no argument)* — Pretty-print the status as JSON **and** return the dict.
- `dict` — Return the status as a Python `dict` (no print).
- `json` — Return the status as a JSON-serialized string.

Any other argument prints an error and returns `None`.

**Returns:** dict (default and `dict` mode) or JSON string (`json` mode).

**Example:**

```python
%cash_status               # prints and returns a dict
status = %cash_status dict # capture as dict
blob   = %cash_status json # capture as JSON string
```

### `%cash_badge`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_badge @016db083 -->

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
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @711be826 -->

Show cache statistics for this kernel session (a restart resets them; what the
cache on disk holds is `cash info`'s): counts, hit rate, compute time, and the
savings broken out as **gross saved**, **cash overhead**, and **net saved**,
plus tracked variables. The headline net counts only savings backed by a
measurement, never the full gross, so a stale first-run timing can never
inflate it. Two kinds of measurement count, and the line says which:

* **verified** — this session recomputed the same statement, so it knows
  today's cost;
* **measured** — an earlier kernel on this machine did. The *least* it was
  ever measured to cost is what gets credited, so a baseline taken on a cold
  first run cannot be paid out forever.

The second kind is what makes a Restart & Run All readable: a fresh kernel
recomputes nothing, so before it the net after a restart printed as a range
whose floor was exactly minus cash's own overhead.

(`gross − overhead` is still reported separately as an upper bound, and a
cache built on another machine vouches for nothing: there the range remains.)
Reporting net keeps the headline honest: cash's own overhead is subtracted
from the recompute it avoided — including the time it spends *inside* a
statement, keying and hashing the calls it routes, which is charged to cash
and not to your code — and a session whose overhead outweighs its hits reads
as a plain "cash cost you Xs this session" rather than a phantom win. The command
deliberately avoids walking the backend so it stays cheap on large on-disk
caches.

If any cache write failed, a **discarded writes** line names the count and the
first cause. Those results were never stored, so they recompute every run —
and none of the counters above can show it, because a discarded write is not a
miss but a hit that never got the chance to exist. Nothing raises at the time,
which is why the rest of the summary can look healthy while the cache is
quietly doing less than it appears to. `%cash_stats reset` does **not** clear
them: a counter is something you may choose to forget, an unresolved fault is
not, and the entries are still missing from disk afterwards.

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
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @29ba1a1b -->

Toggle or configure debug logging. The level is set on the global `cash`
logger, which the upstream checker and the statement pipeline log through, and
passed to the core Cash instance.

**Signature:** `%cash_debug [on|off|json|file <path>]`

**Arguments:**

- *(no argument)* — Toggle between `on` and `off`.
- `on` (aliases: `true`, `1`, `enable`) — Enable debug logging at DEBUG level,
  printed into the cell that produced each record.
- `off` (aliases: `false`, `0`, `disable`) — Disable debug logging: remove the
  handlers cash added and the level it set on the `cash` logger. Handlers your
  own logging setup added stay.
- `json` — Enable DEBUG and print each record as a JSON object.
- `file <path>` — Enable DEBUG and also append the records to `<path>`, one
  JSON object per line. Quote a path that contains a space or a `#`.

Each mode replaces the handlers the previous one added, so switching modes never
prints a record twice. The mode is case-insensitive; a path keeps its case. Any
other argument prints an error and changes nothing.

**Example:**

```python
%cash_debug on
%cash_debug off
%cash_debug json
%cash_debug file /tmp/cash.log
```

---

## Inspecting your session

### `%cash_provenance`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_provenance @a44eb4b5 -->

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

Any other flag, or an extra word, prints an error and shows nothing else.

**Example:**

```python
%cash_provenance              # list all tracked vars
%cash_provenance --all
%cash_provenance df           # detail for `df`
%cash_provenance df --graph --time
%cash_provenance df --json
%cash_provenance --clear
```
