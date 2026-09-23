# Magic commands

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. The canonical reference for all **9** magics.

`import cash` registers the magics; `%load_ext cash` also works. `%cash_help`
lists them in the notebook, and `%cash_help NAME` prints one magic's usage.

<!-- The count above and the table and sections below are checked against the
     registered magics by tests/docs/test_doc_claims.py. Each section carries a
     claim anchor pinned to the method that implements it. -->

| Magic | Purpose |
|-------|---------|
| [`%cash_on`](#cash_on) | Cache every cell from here on. |
| [`%cash_off`](#cash_off) | Stop caching. |
| [`%cash_persist`](#cash_persist) | Store every statement on disk, however cheap. |
| [`%cash_badge`](#cash_badge) | Choose the badge: HTML, text or none. |
| [`%cash_debug`](#cash_debug) | Log cash's decisions. |
| [`%cash_stats`](#cash_stats) | Session statistics and time saved. |
| [`%cash_status`](#cash_status) | The last cell's result as a dict or JSON. |
| [`%cash_provenance`](#cash_provenance) | How a variable was computed. |
| [`%cash_help`](#cash_help) | List the magics and annotations. |

## Turning caching on and off

### `%cash_on`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_on @8cbb17b4 -->

**Usage:** `%cash_on [ttl=N]`

Caches every cell run after this one. Put it with `import cash`, alone, in the
first cell: nothing in that cell is cached.

- `ttl=N`: every stored result expires after `N` seconds. A `# @cash:ttl=N` on a
  statement overrides it. Any other argument, or a value that is not a whole
  number, prints an error and leaves caching off.

It prints `Cash enabled.`, and `Found existing cache with N entries.` when the
cache already holds results. Without a live reader for unsaved edits (see
[Editing without saving](known-limitations.md#editing-without-saving)), it also
prints a one-time tip to save before running. With `CASH_DISABLE=1` set, it says
so and does nothing.

```python
%cash_on
%cash_on ttl=3600
```

### `%cash_off`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_off @700a45e0 -->

**Usage:** `%cash_off`

Cells after this one run uncached until the next `%cash_on`. It also clears the
default `ttl`.

### `%cash_persist`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_persist @6b423b4b -->

**Usage:** `%cash_persist [on|off]`

Stores every statement on disk, as if each carried `# @cash:persist`, skipping
the [cost model](cost-model.md)'s thresholds. With no argument it toggles.
Useful for benchmarks and reproducible runs; wasteful for everyday work.

```python
%cash_persist on
%cash_persist off
```

## Seeing what happened

### `%cash_badge`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_badge @7312ea34 -->

**Usage:** `%cash_badge [html|print|off]`

- `html`: the interactive badge above each cell's output (default).
- `print`: a plain-text summary after each cell, for headless runs, CI logs and
  agents.
- `off`: no badge.

With no argument it prints the current mode. See [Reading the badge](badges.md).

```python
%cash_badge print
```

### `%cash_debug`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @fb6167b9 -->

**Usage:** `%cash_debug [on|off|json|file PATH]`

- `on` / `off`: log cash's decisions as cells run, or stop. With no argument it
  toggles. `true`/`1`/`enable` and `false`/`0`/`disable` also work.
- `json`: log each record as a JSON object.
- `file PATH`: also append each record to `PATH`, one JSON object per line.

What the log lines mean is in [Debugging](tutorials/feature-guides/debugging-and-monitoring.md#2-turn-on-cash_debug).

```python
%cash_debug on
%cash_debug file /tmp/cash.log
```

### `%cash_stats`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @711be826 -->

**Usage:** `%cash_stats [json|reset]`

Prints this kernel session's statistics: cells run, statements computed,
restored and skipped, hit rate, compute time, and time saved as gross, overhead
and **net**. Net counts only savings a measurement backs, so it can be negative.
A **discarded writes** line appears when a result could not be written.

- `json`: the same numbers as a dict.
- `reset`: zeroes the counters and forgets the stored measurements behind the
  net figure. The discarded-writes line stays.

[Debugging](tutorials/feature-guides/debugging-and-monitoring.md#3-check-the-session-with-cash_stats)
explains each line.

```python
%cash_stats
%cash_stats reset
```

### `%cash_status`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_status @0d042233 -->

**Usage:** `%cash_status [dict|json]`

Returns the last cell's statements (status, code, outputs, times), the session's
variable lineage and whether caching is on. With no argument it prints JSON and
returns the dict; `dict` returns it without printing; `json` returns a JSON
string. Statuses use the enum names: `COMPUTED` for the badge's EXECUTED,
`RESTORED` for CACHED.

```python
status = %cash_status dict
```

### `%cash_provenance`
<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_provenance @57772837 -->

**Usage:** `%cash_provenance [NAME|--all|--clear] [--graph] [--time] [--json]`

- No argument or `--all`: every tracked variable with its last status
  (`EXECUTED`, `CACHED` or `SKIPPED`) and number of records.
- `NAME`: how that variable was computed: code, inputs, duration and history.
  Add `--graph` for its dependency graph, `--time` for a timeline, or `--json`
  for JSON instead of text.
- `--clear`: forget all provenance records.

```python
%cash_provenance
%cash_provenance df --graph
```

### `%cash_help`
<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_help @6f3ad8e6 -->
<!-- claim: cash/notebook/ipython/_help.py:help_text @800a29c8 -->

**Usage:** `%cash_help [NAME]`

With no argument, lists every magic with a one-line summary, one example of each
`# @cash:` [annotation](annotations.md), and links for docs and bug reports. With
a name (`badge`, `cash_badge` or `%cash_badge`), prints that magic's full usage.

```python
%cash_help
%cash_help badge
```
