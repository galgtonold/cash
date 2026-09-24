# Seeing what Cash did

!!! info "Applies to: both paths"
    Anyone checking whether Cash reused a result, and why it did or did not.

## Asking a decorated function

<!-- claim: cash/decorator/explain.py:ExplainMixin._explain_call @bd141dbf -->
`func.explain(*args, **kwargs)` answers "would a call with these arguments hit,
and why?" without calling the function, changing its counters or writing
anything:

```python
import cash

@cash.cache
def load(n):
    return list(range(n))

load(1000)
print(load.explain(1000))
print(load.explain(5))
```

For a function in `demo.py`, that prints:

```
[HIT] demo.load - hit
  cache_dir: /home/me/project/.cash
  cache_key: demo.load:c57a56db...::99622373...
  entry_id: 1cea148d7152
  cached_at: 1790204693.5030863
  cache_age_seconds: 0.0016086101531982422
  execution_time_saved: 4.1041999793378636e-05
[MISS] demo.load - no_entry
  cache_dir: /home/me/project/.cash
  cache_key: demo.load:c57a56db...::f6f5c047...
  entry_id: f72bb562e114
  why: new arguments: called with arguments not seen on the last call
```

<!-- claim: cash/decorator/explain.py:EXPLAIN_NO_ENTRY == "no_entry", cash/decorator/explain.py:EXPLAIN_TTL_EXPIRED == "ttl_expired", cash/decorator/explain.py:EXPLAIN_FILE_CHANGED == "file_changed", cash/decorator/explain.py:EXPLAIN_KEY_UNCOMPUTABLE == "key_uncomputable", cash/decorator/explain.py:EXPLAIN_DISABLED == "disabled" -->
The reason is `hit` or one of `no_entry`, `ttl_expired`, `file_changed`,
`key_uncomputable` and `disabled`, each with details such as which file
changed or which argument type could not be hashed. The result is a
[`CacheExplanation`](../api/cash.md).

<!-- claim: cash/core.py:Cash._wrap_with_stats.cache_info @905b7b2b -->
`func.cache_info()` counts the hits and misses of this wrapper since it was
created, with the reasons for the misses. `explain()` itself is not a call, so
above it reports one miss and no hits. Its `total_time_saved` adds up the
compute time recorded when each entry was first written; it is an estimate of
what the hits avoided, not a measurement, so treat it as an upper bound.
`cash.show_stats()` prints the same per-function table for every cached
function in a script.

=== "Decorator"

    Set `debug = true` (or `CASH_DEBUG=true`) to log one line per call: a hit,
    or a miss and why. See
    [configuration](../getting-started/configuration.md).

=== "Notebook"

    Re-running the cell that defines a function creates a new wrapper, so
    `cache_info()` starts again from zero. Use `explain()` or `%cash_stats`,
    which read the cache itself.

## In a notebook

### The badge

<!-- claim: cash/notebook/badge_renderer/view_builder.py:_statement_row @38bc6e18 -->
Every cell run under `%cash_on` prints a badge: which statements came from the
cache, which ran, which ran without being stored and why, and the time each
took. The code shown on each row is your own source. Comments are not part of
the key, so editing one changes what the row shows but still hits.
[Reading the badge](../badges.md) explains every field.

### Where a value came from

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_provenance @efdb75ad, cash/notebook/provenance.py:ProvenanceTracker._format_graph_section @8089ee0d -->
`%cash_provenance NAME` shows the code that produced a variable, its inputs,
the files it read and how long it took. `--graph` adds the chain of inputs
(five levels deep), `--time` the last ten records with whether each step was
computed, restored or skipped, and `--json` the same records as data.
`%cash_provenance --all` lists every tracked variable.

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_provenance df --graph
```

### Debug output

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @fb6167b9 -->
`%cash_debug on` prints Cash's reasoning as it happens: keys, lineages,
upstream checks and restore decisions. `%cash_debug json` prints the same
records as JSON, `%cash_debug file log.txt` also appends them to a file, and
`%cash_debug off` stops. See [magic commands](../magics.md).

### Session totals

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @711be826 -->
`%cash_stats` reports the session: cells run, statements computed, restored
and skipped, the hit rate, and the time saved. `%cash_stats json` gives the
same numbers as JSON and `%cash_stats reset` starts over.

- The hit rate counts only statements worth caching, so cells full of
  `print()` calls do not drag it down.
- **Net time saved** counts only savings backed by a measurement (this
  session's, or the lowest an earlier session on this machine recorded), minus
  Cash's own measured overhead. **Gross time saved** is labelled as an
  estimate. When there is no measurement, as with a cache built on another
  machine, Cash prints a range rather than the flattering figure.
- `%cash_stats` does not scan the cache folder; for its size and entry
  count, run `cash info` in a terminal.

## From a terminal

<!-- claim: cash/__main__.py:main @fc218b1b -->
The `cash` command works on the cache folder itself: `cash info` shows the
settings in force and what the cache holds, `cash inspect` lists entries by
function and size, and `cash clear` deletes them. See
[where the cache folder is](storage.md#where-the-cache-folder-is) and
[the CLI page](../cli.md).
