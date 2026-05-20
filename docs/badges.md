# Reading the Cash badge

Every cell Cash touches gets a badge above its output — a one-line summary plus an expandable detail panel. The badge is the answer to "what did cash do, and why?". This page walks through what the badge shows, then runs through the most common questions it gets used to answer.

## 1. Anatomy

This is a badge from a cell where the upstream `df` was restored, the intermediate `features` was restored, and the final `preds` was recomputed because a new `features` lineage invalidated its cache:

--8<-- "docs/_badges/anatomy_hero.html"

What you're looking at:

1. **Header line** — the collapsed view. Shows the cell-level status (`MIXED` here, because some rows restored and others computed), the count chips (`⚡ 2`, `⚙️ 1`), and `saved Xs / ran Ys` totals. Click anywhere on the header to expand.
2. **Upstream section** — statements from earlier cells that Cash re-checked (or re-ran) before this cell could run. Each row is prefixed with `⬆️` so you can tell at a glance which work is "for this cell" vs "to set up this cell".
3. **Current section** — statements in *this* cell. The `⚡` / `⚙️` icon is the per-row status. Restored rows show `saved Xs`; computed rows show the wall time they took.
4. **Decorator section** — calls Cash recorded for `@cash.cache`-decorated functions invoked during this cell. The `2/3 cached` summary collapses many calls to one line.
5. **Per-row anatomy** — left rail color is the status, then the code (syntax-highlighted), then storage-tier dots (`RAM` and/or `DISK`), then a timing bar scaled to the cell max, then a time chip on the right.
6. **Overhead breakdown** — what Cash itself spent on bookkeeping. The badge shows up to four buckets: `upstream` (re-checking upstream cells), `init` (badge initialization), `progress` (updating the progress display), and `other` (everything else not accounted for by the per-row timings). Shows up when the breakdown is meaningful relative to the cell time.
7. **Report incorrect caching** (footer link) — if a row looks wrong (restored when it shouldn't have been, or recomputed when it should've been a hit), this is the bug-report shortcut. Pre-fills cell context.

## 2. Status reference

Every row and the overall badge use one of these statuses:

| Status | Icon | When you see it |
|---|---|---|
| **RESTORED** | ⚡ | Row's value came from the cache. |
| **COMPUTED** | ⚙️ | Row ran. May or may not have been stored — check the row detail for "NOT CACHED". |
| **SKIPPED** | ⏩ | Row was unreachable on this run (downstream of a branch not taken) or its value isn't needed. |
| **MIXED** | — | Cell-level only: some rows restored, some computed. |
| **FUNCTION_CHANGED** | 🔄 | A helper function this row calls had its source change since the last run. |
| **MODULE_RELOADED** | 🔄 | A tracked local import was edited; everything downstream re-runs. |
| **WARNING** | ⚠️ | Something to look at (e.g. unseeded random); the row still ran. |
| **ERROR** | ❌ | The statement raised. |

Individual examples:

--8<-- "docs/_badges/status_restored.html"

--8<-- "docs/_badges/status_computed.html"

--8<-- "docs/_badges/status_skipped.html"

--8<-- "docs/_badges/status_mixed.html"

--8<-- "docs/_badges/status_function_changed.html"

--8<-- "docs/_badges/status_module_reloaded.html"

--8<-- "docs/_badges/status_warning.html"

--8<-- "docs/_badges/status_error.html"
