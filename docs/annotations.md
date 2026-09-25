# Annotations

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. The reference for every `# @cash:` comment.

An annotation is a comment on or just above a statement that changes how cash
treats that statement. [Controlling caching](tutorials/feature-guides/controlling-cache-behavior.md)
shows when to reach for each one; this page is the reference.

## At a glance

| Directive | Effect |
|---|---|
| `# @cash:persist` | Store the result on disk, even if it is too cheap to be stored otherwise. |
| `# @cash:no-cache` | Run the statement every time. Wins over every other directive. |
| `# @cash:ttl=N` | Treat the stored result as expired after `N` seconds. |
| `# @cash:allow-random` | Silence the unseeded-randomness warning. Does not change caching. |
| `# @cash:cache-fit` | Cache a bare estimator fit such as `clf.fit(X, y)`. |
| `# @cash:no-cache-calls` | Stop caching the calls inside the statement. |
| `# @cash:assume-safe` | Cache the statement despite a side effect that is harmless to skip. |

<!-- test:skip reason="illustrative: train_model, fetch_quotes and data are the reader's own" -->
```python { .nb-cell }
# @cash:persist
features = build_features(data)   # stored on disk, so it survives a restart

# @cash:ttl=300
quotes = fetch_quotes("AAPL")     # fetched again after five minutes
```

## Where to put them

<!-- claim: cash/analysis/annotations.py:parse_annotations_in_range @27219e63 -->
- **Above the statement.** Cash reads the comment lines directly above a
  statement, walking up until a blank line or a line of code. Plain comments in
  between are fine; a blank line ends the search.
- **At the end of the line.** `x = expensive()  # @cash:persist` works too.
- **On a loop or `if` header.** The directive applies to every statement in the
  block. Inside the block, it applies to the statement below it only. A `while`
  or `with` block, or a `for` with `break`, `continue` or an `else:` block, runs
  as one unit, so a directive anywhere inside it applies to the whole block.
- **At the top of a cell.** `no-cache` and `no-cache-calls` in the cell's first
  comment lines apply to every statement in the cell. They do not reach a `for`
  or `if` block that comes after another statement; put the directive directly
  above that block. The other directives apply to the first statement only.

<!-- test:skip reason="illustrative: contrasts directive placement" -->
```python { .nb-cell }
# @cash:persist
# builds the lookup table used below
# @cash:ttl=3600
table = build_table(raw)          # persist and ttl both apply

# @cash:persist

model = train(table)              # not applied: the blank line breaks it
```

<!-- claim: cash/analysis/annotations.py:CacheAnnotation.merge @b10e0cdc -->
When several directives reach one statement, they combine: each flag applies if
any of them sets it, and the last `ttl=` wins. `no-cache` beats `persist` and
also switches off the caching of calls inside the statement.

<!-- claim: cash/analysis/annotations.py:ANNOTATION_PATTERN @412c3ce1, cash/analysis/annotations.py:parse_annotation_line @5d8ea461 -->
Spelling: `@cash:` must be lower case, and a space after the colon or around `=`
is fine (`# @cash: persist`, `# @cash:ttl = 60`). An unknown name warns once with
[`ANNOT-UNKNOWN-DIRECTIVE`](warnings.md#annot-unknown-directive) and suggests the
right spelling (`nocache` → `no-cache`).

## Directives

### `# @cash:persist`

Stores the result on disk even when it took less than the 0.1 s that disk storage
normally needs, so it survives a kernel restart. It skips every floor and gate of
the [cost model](cost-model.md); only a storage tier's size cap still applies.
Results that took longer than 0.1 s are usually on disk already.
`%cash_persist on` does the same for every statement.

### `# @cash:no-cache`

<!-- claim: cash/analysis/cacheability_decision.py:decide_cacheability @e0e77376 -->
The statement runs every time and nothing is stored, including the calls inside
it. The badge shows a plain `EXECUTED` row. A statement below that reads its
result is cached as usual and computed again when the value changes.

Use it for what cash cannot see for itself: a read whose answer must be fresh
every run (a GET for a live price), a helper with an effect hidden from cash, or
a helper that reads the clock. You do not need it for file writes, POSTs,
`datetime.now()` or plots written in the statement: cash already runs those every
time. See [What gets cached](notebook_caching_api.md#what-gets-cached).

`no-cache` also gives a random draw a fresh value each run; see
[Randomness](known-limitations.md#randomness).

### `# @cash:ttl=N`

The stored result expires `N` seconds after it was written; the next run
computes it again. `N` is a whole number of seconds with no unit; `ttl=0` means
it is always expired. It overrides the default from `%cash_on ttl=N`.

A value cash cannot read (`ttl=5m`, `ttl=-30`, `ttl=`) is ignored with an
[`ANNOT-TTL-INVALID`](warnings.md#annot-ttl-invalid) warning, and the statement
keeps its normal caching. Write `# @cash:ttl=300` for five minutes.

### `# @cash:allow-random`

Silences the `RANDOM-UNSEEDED` and `RANDOM-REPLAYED` warnings for the statement.
The value is still cached and replayed. To get a new draw every run, use
`no-cache`. See [Randomness](known-limitations.md#randomness).

### `# @cash:cache-fit`

A bare fit such as `clf.fit(X, y)` changes `clf` in place, so by default it runs
every time (the badge says `NOT CACHED` and names this directive). With
`cache-fit`, cash caches the fit and a hit restores the fitted estimator. It also
covers `X2 = vec.fit_transform(X)` and `labels = km.fit_predict(X)`, which fit in
place while returning something else.

<!-- test:skip reason="illustrative: needs a real estimator and training data" -->
```python { .nb-cell }
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# @cash:cache-fit
clf.fit(X, y)
```

!!! warning "A hit can replace the object"
    A cache hit may bind `clf` to the restored estimator instead of updating the
    object in place, so another name for it (`backup = clf`) can be left pointing
    at an unfitted estimator. Use `cache-fit` only when nothing else refers to the
    estimator, and check that the badge reads `CACHED` on a re-run.

You rarely need it: an assignment that binds the fitted estimator,
`clf = RandomForestClassifier(random_state=42).fit(X, y)`, caches with no
directive and has no such caveat.

### `# @cash:assume-safe`

<!-- claim: cash/analysis/cacheability_decision.py:decide_cacheability @e0e77376, cash/analysis/annotations.py:leading_cell_annotation @0d279828 -->
Cash runs a statement with a side effect every time, and it judges effects by
name: it cannot tell a POST that creates an order from a POST that runs a search.
`assume-safe` tells it the effect is harmless to skip, so the statement is cached
and a hit returns the stored answer without sending the request:

<!-- test:skip reason="illustrative: needs a live endpoint" -->
```python { .nb-cell }
hits = session.post(SEARCH_URL, json=query).json()   # @cash:assume-safe
```

It waives side effects only. A statement that changes an object in place, reads
the clock or calls `input()` still runs every time. It applies to its own
statement, even at the top of a cell. Add `# @cash:ttl=N` when the answer can go
stale.

### `# @cash:no-cache-calls` { #call-level-caching-default-and-cashno-cache-calls }

<!-- claim: cash/notebook/call_unit.py:CallUnit._entry_for @56cf9df0 -->
Cash also caches the expensive **calls inside** a statement, by default and with
no directive. That is what keeps work cached where the statement itself cannot
be: in `results.append(compute(x))` the append runs every time, but `compute(x)`
is served from the cache. `no-cache-calls` turns this off for a statement, a
block (on its header) or a cell (at its top).

A call is cached when it does not read the statement's own target:

| Statement | Cached call |
|---|---|
| `s += compute(x)` | `compute(x)` |
| `out.append(compute(x))` | `compute(x)` |
| `rows.append(dict(t=t, v=compute(t)))` | `compute(t)` |
| `s = merge(s, x)` | none: the call reads `s` |
| `df.sort_values(inplace=True)` | none: the change is the work |

A call's key is the function and the values it receives, so a reordered loop or
the same call in another cell is served too. Inside a comprehension each element
is part of the key. Builtins, classes and method calls (`model.predict(x)`) are
not cached this way. Calls show on the badge tagged `[intercepted]`:

```text
  @cash.cache:
    compute() [intercepted]: 2/3 cached (0.402s)
```

The statement around a cached call is then stored only for its own work; see
[a statement around a cached call](cost-model.md#a-statement-around-a-cached-call).

<!-- claim: cash/notebook/statement/call_routing.py:CallRouting.current_call_ttl @2d43c3de, cash/notebook/statement/call_routing.py:CallRouting.current_call_persist @acb22cc8 -->
The statement's `ttl=`, `persist` and `assume-safe` apply to the calls inside it,
and `no-cache` switches them off.

!!! warning "A cached call skips effects cash cannot see"
    Cash does not serve a call from the cache when it sees the function change
    an argument, draw random numbers or write a file, and it restores a change
    the function makes to a global. Any other effect is skipped on a hit. Use `no-cache-calls` on the statement, or
    mark the function [`@stateful`](tutorials/feature-guides/controlling-cache-behavior.md#stateful-helpers),
    when you are not sure.

## Common mistakes

<!-- test:skip reason="illustrative: each directive here is misplaced or misspelled" -->
```python { .nb-cell }
# @Cash:persist             # ignored: @cash: must be lower case
model = train()

# @cash:persist

model = train()             # ignored: a blank line breaks the link

model = train()
# @cash:persist             # applies to the next statement, not this one

# @cash:perist              # typo: warns ANNOT-UNKNOWN-DIRECTIVE, then ignored
model = train()
```
