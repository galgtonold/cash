# Cash — Domain Vocabulary

Single source of truth for the terms used in this codebase. When you reach for an architectural noun, find it here first; if it's missing and load-bearing, add it.

This is **not** a tutorial. It's a glossary that disambiguates words we already use.

---

## Core concepts

**Statement** — A single top-level Python statement inside a notebook cell. Cash caches at statement granularity, not cell granularity. A cell containing three statements produces three cache entries.

**Cache key** — The deterministic identifier under which a statement's outputs are stored. Format: `stmt:{sha256(code + ':'.join(sorted(input_lineages)) + file_hash_component)}`. **All cache-key computation goes through `compute_cache_key()` in [`cache_key.py`](src/cash/notebook/cache_key.py).** See [`.github/copilot-instructions.md`](.github/copilot-instructions.md) §Critical Conventions for why.

**Lineage** — A hash chain that captures *how* a variable was produced: `hash(code + sorted(input_lineages) + file_deps)`. Two variables with the same value but different lineage are not cache-equivalent. Lineage flows through the dependency graph; cache keys are computed from it.

**LineageStore** — The single seam through which a variable's persistent lineage is read and written. Lives in [`lineage_store.py`](src/cash/notebook/lineage_store.py); held by `TrackingState.lineage`. Owns the `variable_lineage` dict and coordinates the paired `_cash_lineage_hash` attribute write so the two can never drift. Owns the priority ladder for lineage resolution (virtual → store → `_cash_lineage_hash` → compute_hash → str fallback). **Does not** own transient simulation state (`virtual_lineage` is passed in per-call) or skip-check state (`executed_input_lineages` etc. — different invariants).

**Upstream simulation** — Before running a cell, cash simulates all upstream cells to detect stale variables. Lives in [`upstream.py`](src/cash/notebook/upstream.py).

---

## Architectural terms

**Cacheability** — The property *"can this statement be cached?"*. Distinct from *"should the user have asked for it?"* (annotations) and *"has it been cached before?"* (cache hit). Cacheability is a pure-AST decision: given the code, are there in-place mutations, side effects, or stateful calls that would make caching unsound? Lives in [`cacheability.py`](src/cash/notebook/cacheability.py).

**StatementAnalysis** — Value object returned by `analyze_statement(code, tree, outputs)`. Carries everything the AST can tell us about a statement: `top_level_mutated_vars`, `all_mutated_vars`, `side_effects`, `called_names`. Pure data + a `skip_reasons(outputs)` helper that turns findings into human-readable strings for `metrics['uncacheable_reasons']`. **Does not** know about user_ns, decorators, or annotations — those are caller concerns.

**Annotation** — A `@cash:` comment directive (`no_cache`, `ttl`, `persist`, `allow-random`) that the user writes in the source. Parsed in [`annotations.py`](src/cash/notebook/annotations.py) into a `CacheAnnotation`. Distinct from cacheability: annotations are *user input*, cacheability is *code analysis*. The caller (`statement_processor`) merges them.

**Purity / Stateful** — A function-level marker. `@pure` declares "calling me produces no observable state change"; `@stateful` declares the opposite. Lives in [`purity.py`](src/cash/notebook/purity.py) and is part of the **public API** (re-exported from `cash`). Cacheability *consumes* the purity registry via name-resolution against `user_ns`; it does not own it.
