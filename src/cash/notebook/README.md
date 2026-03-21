# Notebook Subsystem

Statement-level caching for Jupyter notebooks via IPython magics.

## Module Map

| Module | Purpose |
|--------|---------|
| `magics.py` | IPython magic commands (`%cash_on`, `%%cash`), cell-level orchestration |
| `magic_admin.py` | Admin/debug magic commands (`%cash_status`, `%cash_debug`) |
| `statement_processor.py` | Per-statement cache lookup → execute/restore → lineage tracking |
| `upstream.py` | Detect changed upstream cells, simulate lineage forward propagation |
| `cache_key.py` | **Single source of truth** for statement cache key computation |
| `cache_status.py` | CacheStatus enum and ExecutionResult |
| `analysis.py` | AST-based input/output variable detection |
| `annotations.py` | `@cash:` comment directive parsing (no-cache, ttl, persist) |
| `control_structures.py` | Per-iteration caching for loops and conditionals |
| `file_tracker.py` | Intercept file reads (pandas, numpy, polars, open) for dependency tracking |
| `function_tracker.py` | Track function source code changes, module hot reload |
| `mutation_detector.py` | AST-based detection of in-place mutations |
| `module_invalidator.py` | Module reimport invalidation |
| `side_effects.py` | Detection of file writes, network calls |
| `randomness.py` | Unseeded random call detection |
| `purity.py` | `@pure` and `@stateful` decorator system |
| `provenance.py` | Variable computation history |
| `audit.py` | Compliance audit logging |
| `badge_renderer.py` | HTML/text badge rendering for cache status display |
| `error_display.py` | Clean error formatting for notebook output |
| `_protocols.py` | Protocol interfaces (ShellProtocol, CashInstanceProtocol) |

## Dependency Flow

```
magics.py ──→ statement_processor.py ──→ cache_key.py
    │                  │
    ├──→ upstream.py ──┘ (uses cache_key.py for simulation)
    │        │
    │        └──→ analysis.py
    │
    ├──→ control_structures.py (delegates back to statement_processor)
    ├──→ badge_renderer.py
    └──→ error_display.py

Shared state flows through _protocols.py TrackingState (single owner).
```

**Circular dependency note**: `magics → statement_processor → core → notebook/__init__ → magics`.
Broken via `__getattr__` lazy imports in `__init__.py` and `TYPE_CHECKING` guards in `upstream.py`.
