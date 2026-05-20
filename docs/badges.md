# Reading the Cash badge

Every cell Cash touches gets a badge above its output — a one-line summary plus an expandable detail panel. The badge is the answer to "what did cash do, and why?". This page walks through what the badge shows, then runs through the most common questions it gets used to answer.

## 1. Anatomy

This is a badge from a cell where the upstream `df` was restored, the intermediate `features` was restored, and the final `preds` was recomputed because a new `features` lineage invalidated its cache:

--8<-- "docs/_badges/anatomy_hero.html"

What you're looking at:

1. **Header line** — the collapsed view. Shows the cell-level status (`EXECUTED` here), the total time and savings (`0.42s · saved 3.46s`), a tiny sparkline, and counter chips (`exec 1`, `cached 2`). Click the header to expand and see the panel below.
2. **Upstream context** — a collapsed pill at the top of the panel labeled "upstream context · 1 step · ↑2.85s". Click to expand and see the upstream rows (statements from earlier cells that Cash had to re-check or re-restore for this cell to be valid).
3. **Current cell** — the section labeled `CURRENT CELL`. Each row is a statement in *this* cell with its per-row status — `restored` (green rail) for `features = encode(df)`, `computed` (ochre rail) for `preds = decorated_predict(features)`. Click any row for a detail tooltip showing the cache key, storage tiers, miss reason, and `@cash.cache` hit ratio (`2/3 cache hits` here).
4. **Decorator cache** — the section labeled `DECORATOR CACHE (@cash.cache)`. One row per `@cash.cache` call during this cell. The example shows three calls to `predict_one()`: two HITs and one MISS.
5. **Per-row anatomy** — left vertical rail color is the status (green = cached, ochre = computed), then the syntax-highlighted code, then storage-tier dots (RAM/DISK — filled dot = present in that tier, ring = restored from that tier), then a timing bar scaled to the cell max, then a time chip on the right.
6. **Overhead** — the bottom row labeled `overhead`, split into up to four sub-parts: `upstream` (re-checking upstream cells), `init` (badge initialization), `progress` (updating the progress display), and `other` (everything else). Shows when the breakdown is meaningful relative to cell time.
7. **Report incorrect caching** (footer link) — if a row looks wrong, this opens a pre-filled GitHub issue with the badge output included.

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
