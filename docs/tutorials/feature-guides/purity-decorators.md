# `@pure` and `@stateful` — telling Cash what to trust

Cash reads your code to decide what is safe to cache. `@pure` and `@stateful` are
how you overrule that judgement — one says "trust this function", the other says
"never cache anything that calls this".

This tutorial walks through both markers, the auto-detection machinery behind
them, and the footguns that bite people in practice.

!!! note "Which marker do you need?"
    They act on different mechanisms, and that decides which one is useful to you:

    | You want to… | Use | Applies to |
    |---|---|---|
    | Stop a notebook statement from caching | **`@stateful`** | statements (`%cash_on`) **and** the decorator's analyzer |
    | Vouch for a function so cash stops warning about it | **`@pure`** | the `@cash.cache` analyzer |
    | Silence or harden a decorated function wholesale | `assume_safe=` / `strict=` | `@cash.cache` — see the [decorator guide](../../decorator.md#strict-and-assume_safe-purity-gates) |

    `@pure` does **not** turn caching on for a notebook statement — statements
    that call ordinary helpers already cache. Use `@stateful` when you need to
    stop one.

## Why this exists

Cash inspects your code with an AST visitor before deciding what to cache. It
flags mutations to top-level variables (`data.append(...)`), attribute writes,
file I/O, network calls, and a long list of "looks side-effectful" patterns —
because replaying a cached return value when the real call would have written to
disk or posted to an API would be a serious bug.

But cash can only see what's in front of it. A call into a function it can't
introspect is a judgement call, and it can be wrong in either direction:

- **It can't tell that your side-effecting helper matters.** A statement calling
  `post_to_slack(df)` looks like any other call, and its return value caches
  happily — so the second run silently skips the notification. `@stateful` is
  how you say "never cache a statement that calls this."
- **It can flag a function you know is fine.** The `@cash.cache` analyzer warns
  about callees that look impure. `@pure` is how you say "I've audited this,
  stop warning" — the warning is advisory, so this is about noise, not
  correctness.

Cash also runs a fallback heuristic (`analyze_function_purity`) on undeclared
functions, so you don't have to mark everything. The markers matter when you
know something the analyzer cannot.

## Quick start

The two markers do different jobs, and picking the right one starts with knowing
which mechanism you're steering:

- **`@stateful`** stops a **notebook statement** from being cached at all. This is
  the marker that changes what `%cash_on` does.
- **`@pure`** tells the purity **analyzer** to trust a function, silencing the
  warning that [`@cash.cache`](#purity-on-the-decorator-cashcache) raises about
  it. It does *not* decide whether a notebook statement caches.

Start from the default: an ordinary helper needs no marker at all. Cash already
caches statements that call it, and a re-run restores:

```python { .nb-cell }
import cash
%cash_on

def featurize(df):
    return df.assign(score=df["a"] * df["b"])

result = featurize(my_df)      # re-run the cell: the badge reads RESTORED
```

<iframe class="cash-badge" src="/_badges/purity_restored.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

Now mark a helper `@stateful` — the side effect is the point of calling it, so a
replayed return value would be wrong. Cash stops caching every statement that
calls it, and the badge names the reason:

<!-- test:skip reason="illustrative — posts to a fake endpoint; the @stateful verdict is the point, not the call" -->
```python { .nb-cell }
from cash import stateful

@stateful
def log_run(df):
    requests.post("https://hooks.example.com/run", json={"rows": len(df)})

log_run(my_df)                 # NOT CACHED — "Calls @stateful function"
```

<iframe class="cash-badge" src="/_badges/not_cached_purity.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

That is the whole notebook-statement story: **`@stateful` is the lever; nothing
else is required.** `@pure` earns its keep on the decorator — see
[Purity on the decorator](#purity-on-the-decorator-cashcache).

## `@pure` — "trust me, this is safe"

!!! note "`@pure` does not make a notebook statement cacheable"
    A statement calling an unmarked helper already caches — there is no
    "refused until you vouch for it" state to rescue. `@pure` is a promise to
    the **purity analyzer**, and the analyzer is what
    [`@cash.cache`](#purity-on-the-decorator-cashcache) consults: marking a
    callee `@pure` silences the `CashImpurityWarning` that would otherwise fire
    for it. That applies to decorated functions written in a notebook cell too —
    it's the decorator path that matters, not the file it lives in. In the
    statement path (`%cash_on`) the only marker that changes a verdict is
    [`@stateful`](#stateful-this-should-never-cache).

### When to use it

Use `@pure` when *all* of the following hold for the function:

1. The return value is a deterministic function of the arguments. Same inputs in, same output out.
2. There are no side effects you care about — no file writes, no network calls, no DB writes, no logging that downstream code depends on, no in-place mutation of arguments.
3. The function doesn't depend on global state (current time, environment variables, RNG without a fixed seed, module-level mutable objects).

Concrete examples:

- Pure math: `def euclidean(p, q): return math.sqrt(sum((a-b)**2 for a, b in zip(p, q)))`
- Deterministic transformations on immutable inputs: pandas/polars dataframe transformations that return new frames.
- Feature engineering helpers that take inputs and return derived columns.
- Parsers, formatters, validators that don't touch the outside world.

### Example

```python
import polars as pl
from cash import pure

@pure
def featurize(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        score=pl.col("clicks") * pl.col("dwell_ms") / 1000,
        bucket=pl.col("score").qcut(10),
    )

@pure
def euclidean(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5
```

### What it actually does

`@pure` is a one-line marker. It sets `_cash_pure = True` on both the original function and the wrapper (`src/cash/notebook/purity.py`).

When the statement processor evaluates a cell, it looks at every bare-name call (`foo(x)`, not `obj.foo(x)`). For each name, it consults `_check_callable_stateful` (`src/cash/notebook/statement/processor.py`), which:

1. Returns `False` (not stateful) if the name is a known-pure builtin like `len` or `sum`.
2. Returns `True` if the resolved object has `_cash_stateful = True`.
3. Returns `False` if the resolved object has `_cash_pure = True` — the path `@pure` activates.
4. Otherwise falls back to `analyze_function_purity`, and returns `False` regardless.

Read steps 3 and 4 together and the consequence is clear: **only step 2 changes
the outcome.** `@pure` short-circuits to the same "not stateful" answer the
fallthrough already gives, so in the statement path it is a performance and
predictability nicety — Cash never peeks inside and never runs the AST heuristic
— not a switch that turns caching on. Its load-bearing use is on the decorator,
below.

### When NOT to use it

Do *not* apply `@pure` to a function that:

- Reads or writes files. `open(...)`, `pd.read_csv(...)`, `df.to_parquet(...)`. Even reads can be problematic if the file on disk changes between runs.
- Calls the network. `requests.get`, gRPC, message queues — all forbidden.
- Reads the wall clock or RNG state without a seed. `datetime.utcnow()`, `random.random()`, `np.random.randn()` without `seed`.
- Reads or writes module globals, environment variables, or any shared mutable state.
- Mutates its arguments in place. `lst.append(...)`, `df.sort_values(..., inplace=True)`. The first call will cache the side effect, future calls will skip it, and your data won't get sorted.
- Calls another function you don't trust to be pure.

If you're unsure, leave it undecorated and let Cash's heuristic decide.

## `@stateful` — "this should never cache"

### When to use it

`@stateful` is the opposite assertion: even if Cash *could* cache the cell, you'd rather it didn't. Use it for any function whose *side effect* is the whole point of calling it, not just the return value.

Concrete examples:

- Functions that send notifications, hit dashboards, post to Slack.
- Functions that write to a production database or a shared store.
- Functions that train a model and update an external artifact registry.
- Functions whose return value depends on the current wall clock, network state, or a shared queue.

### Example

```python
import requests
from cash import stateful

DASHBOARD_URL = "https://dash.example.com/metrics"

@stateful
def log_to_dashboard(metrics: dict) -> dict:
    """Side-effecting: posts metrics to the team dashboard."""
    response = requests.post(DASHBOARD_URL, json=metrics)
    return response.json()

@stateful
def send_alert(channel: str, message: str) -> None:
    slack.chat_postMessage(channel=channel, text=message)
```

Now any cell that calls `log_to_dashboard(...)` or `send_alert(...)` runs fresh on every notebook re-execution — Cash will not replay a cached value.

### What it actually does

`@stateful` sets `_cash_stateful = True` on the wrapped function (`src/cash/notebook/purity.py`). When the statement processor walks the bare-name calls in a cell and finds one whose resolved callable has that attribute, `_check_callable_stateful` returns `True`. The caller (in `decide_cacheability`) then refuses to cache the cell and records the reason "Calls @stateful function".

`@stateful` is checked *before* `@pure` in `_check_callable_stateful`, so if you ever (accidentally) stack both decorators on the same function, stateful wins. Don't rely on that — see the [caveats](#mixing-markers).

## Auto-detection (`analyze_function_purity`)

You won't decorate everything. For undecorated functions, Cash falls back to an AST-based heuristic — `analyze_function_purity` (`src/cash/notebook/purity.py`). It:

1. Grabs the function's source via `inspect.getsource`.
2. Parses it with `ast`.
3. Walks the body with `_ImpurityVisitor`.

The visitor flags the function as impure if it sees any of:

- A `global` or `nonlocal` declaration.
- A `yield` or `yield from` (generators are not safe to memoize as plain values).
- A call to a known-impure builtin: `print`, `input`, `open`, `exec`, `eval`, `compile`, `exit`, `quit`, `breakpoint`.
- A dotted call to a known-impure module function: `os.system`, `os.remove`, `subprocess.run`, `shutil.move`, `requests.get`, `requests.post`, `json.dump`, `pickle.dump`, `logging.info`, and friends.
- A method call to a name in the "write-ish" set: `write`, `writelines`, `append`, `extend`, `insert`, `pop`, `remove`, `sort`, `reverse`, `clear`, `update`, `add`, `discard`, `to_csv`, `to_excel`, `to_parquet`, `to_json`, `to_pickle`, `savefig`, `save`, `send`, `sendall`, `sendto`.
- Any assignment whose target is an attribute (`self.x = ...`) or subscript (`d[k] = ...`).
- Any `+=`/`del` on an attribute or subscript.

The full set lives in `_IMPURE_FUNCTION_CALLS`, `_IMPURE_MODULE_CALLS`, and `_WRITE_METHODS` at `src/cash/notebook/purity.py`.

Results are cached by source-SHA-256 to keep repeated analyses cheap — the cache holds up to 200 entries and evicts oldest-first.

You can call this directly if you want to inspect a function programmatically:

```python
from cash import analyze_function_purity

def helper(x):
    return x * 2

def writer(x):
    with open("/tmp/log", "w") as f:
        f.write(str(x))
    return x

analyze_function_purity(helper)   # True
analyze_function_purity(writer)   # False — `open` and `write` flagged
```

The heuristic is intentionally conservative. False positives (declaring something impure when it isn't) are recoverable: slap on `@pure`. False negatives (declaring something pure when it isn't) would be catastrophic, so the bias is "if in doubt, impure".

## Known-pure builtins

Cash short-circuits the analysis for stdlib names it already knows are safe. The full list lives in `KNOWN_PURE_BUILTINS` (`src/cash/notebook/purity.py`):

```
# Type constructors / conversions
int, float, str, bool, bytes, complex,
list, tuple, set, frozenset, dict

# Numeric / math
abs, round, pow, divmod, min, max, sum

# Sequence / iteration
len, sorted, reversed, enumerate, zip, range,
map, filter, all, any

# Object introspection
type, isinstance, issubclass, id, hash,
callable, hasattr, getattr,
repr, ascii, format, chr, ord,
hex, oct, bin

# Containers
iter, next, slice
```

You never need to decorate these. A cell that does `n = len(data); s = sum(data); top = max(data)` is cached the same as one that does `n = my_pure_len(data)`.

You can check membership programmatically — but mind the [string-not-callable footgun](#is_known_pure-takes-a-string):

```python
from cash.notebook.purity import is_known_pure

is_known_pure("len")     # True
is_known_pure("requests.get")  # False
```

## Caveats — read these before deploying

Six footguns account for most "but Cash said it would cache" surprises.

### Method calls bypass the gate

The bare-name check at `src/cash/notebook/cacheability.py` only collects calls whose `node.func` is an `ast.Name`. Method calls (`obj.train()`) have `node.func` as `ast.Attribute` and are *not* fed into `_check_callable_stateful`.

That means this does **not** work:

<!-- test:expect-raises -->
```python
from cash import stateful

class Trainer:
    @stateful
    def train(self, data):
        self.model.fit(data)
        return self.model.score(data)

trainer = Trainer()
score = trainer.train(data)   # cell may STILL be cached!
```

`Trainer.train` has `_cash_stateful = True`, but Cash never sees that — it only sees a method call on `trainer`. This is the single biggest footgun. Workarounds:

- Expose a top-level function that wraps the method: `def train(t, d): return t.train(d)`, decorated. Call *that* from your notebook.
- Add a `# @cash:no-cache` annotation on the offending cell.
- Mutate something visible at top level inside the cell so the mutation visitor catches it (ugly, do not recommend).

### `@pure` is trusted blindly

There is no runtime safety net inside `@pure`. Cash does not validate that your function is actually pure — it just sets the marker and moves on. If you lie, Cash will silently cache the wrong thing:

```python
from cash import pure

@pure                              # we lied
def featurize_and_log(df):
    with open("/tmp/audit.log", "a") as f:
        f.write(f"called at {datetime.utcnow()}\n")
    return df.assign(score=df["a"] * df["b"])
```

The first call writes one line to `/tmp/audit.log` and returns the new frame. The second call (with the same `df`) returns the *cached* frame and writes nothing. The audit log silently goes stale.

The mutation visitor at the cell level (`_MutationVisitor` in `cacheability.py`) catches obvious top-level side effects (`data.append(1)` written *in the cell*), but it does not enter function bodies. Side effects buried inside a `@pure` function are invisible.

Rule of thumb: if you're not 100% sure, don't write `@pure`. The default is already pretty smart.

### `@stateful` doesn't propagate

`@stateful` is a flag on a single function. It does not propagate to callers:

```python
from cash import stateful

@stateful
def write_to_db(row): ...

def update_everything(rows):
    for row in rows:
        write_to_db(row)
    return len(rows)

# Cell:
n = update_everything(rows)   # CACHED — Cash doesn't see write_to_db
```

Cash only inspects bare-name calls *in the current cell*. The call to `write_to_db` is inside `update_everything`'s body, which Cash doesn't recursively descend into. The cell sees one bare-name call: `update_everything`, which is undecorated.

If you want the cell to opt out, mark `update_everything` as `@stateful` too. (Or, better, add a `# @cash:no-cache` annotation on the cell so the decision is visible at the call site.)

### Mixing markers

Both `_cash_pure` and `_cash_stateful` can technically coexist on the same function. The check order in `_check_callable_stateful` happens to look at stateful first, so stateful wins:

```python
from cash import pure, stateful

@pure
@stateful           # please don't
def confused(x):
    return x * 2
```

This works (the cell will refuse to cache), but it's a check-order artifact, not a language-level guarantee. Treat it as undefined behavior and never stack the two decorators on the same function.

### `is_known_pure` takes a string

Common slip-up: the helper takes a *name string*, not a callable:

```python
from cash.notebook.purity import is_known_pure

is_known_pure(len)       # False — `len` the function object is not in the frozenset
is_known_pure("len")     # True
```

It's checking membership in a `frozenset[str]` (`src/cash/notebook/purity.py`). Always pass the name, not the callable. If you have a callable and want the name, use `func.__name__`.

### Source-hash cached analysis

`analyze_function_purity` keys its result cache on the SHA-256 of the function's source. Two byte-identical functions share a verdict — fine in 99% of cases, but it bites in one specific pattern: redefining a function with the *same body* but different surrounding globals or different intent.

```python
from cash import analyze_function_purity
from cash.notebook.purity import clear_purity_cache

DASHBOARD = None

def push(metrics):
    return {k: v * 2 for k, v in metrics.items()}

analyze_function_purity(push)   # True — looks pure

# Later, you monkey-patch the body via globals to add a side effect.
# Source bytes unchanged, verdict still in cache.
DASHBOARD = some_real_client
def push(metrics):                                # exact same source bytes
    return {k: v * 2 for k, v in metrics.items()}

analyze_function_purity(push)   # Still True — cached verdict sticks
```

In practice you'll mostly hit this when stubbing functions in tests. The fix is one line:

```python
# test:inject: from cash.notebook.purity import clear_purity_cache
clear_purity_cache()
```

`clear_purity_cache` is a public helper (`src/cash/notebook/purity.py`) — call it in `setUp` / a pytest fixture / wherever you redefine functions and want fresh analysis.

## Purity on the decorator (`@cash.cache`)

Everything above is the *notebook* story (`%cash_on` decides per-statement
what to cache based on purity). The same machinery now runs on
`@cash.cache`-decorated functions too — with three opt-in modes that map
cleanly to "I want a warning", "I want it silent", and "I want it to
fail CI".

### Default: warn at first call

<!-- test:expect-warning reason="this section exists to demonstrate the first-call impurity warning" -->
```python
import cash

@cash.cache
def fetch_user(uid):
    return requests.get(f"https://api/{uid}").json()

fetch_user(42)
# CashImpurityWarning: @cash.cache on __main__.fetch_user: the analyzer
# found likely side effects or scope mutations. Cached results may not
# reflect side-effect intent. Suppress with @cash.cache(assume_safe=True)
# after auditing, or refactor.
#   in __main__.fetch_user:
#     line 2: [impure_call] requests.get() — known I/O / side-effecting
```

The function is still cached. The warning is one-shot per
`(function, reason)`, so noisy hot loops don't flood the log. The
emitted warning is also stored on the wrapper:

```python
fetch_user.cache_info()["warnings"]
# [{'category': 'CashImpurityWarning', 'message': '...', 'timestamp': ...}]
```

!!! warning "Untrackable dependencies *raise*, they don't warn"
    Warn-and-cache is for ordinary side effects (I/O, mutations, discarded
    returns). A different class — a dependency resolved from a **runtime value**
    that cash can't track: `eval`/`exec`/`compile`, dynamic dispatch via
    `getattr(obj, name)()`, or `importlib.import_module` — **raises
    `CashImpureFunctionError` on the first call even in default mode**, because a
    cached result could go silently stale. Pass `assume_safe=True` to cache it
    anyway, or refactor to a statically-named call.

### `assume_safe=True` — silence after auditing

Use when caching the function is fine (e.g. the side effect is
idempotent — same URL returns the same JSON; logging is acceptable
to lose on a hit):

```python
@cash.cache(assume_safe=True)
def fetch_user(uid):
    return requests.get(f"https://api/{uid}").json()
```

The analyzer still runs (its helper-source-hashes are needed for cache
invalidation), but no warning fires.

### `strict=True` — fail loud

Useful in CI to fail the build when someone introduces caching of
side-effecting code:

<!-- test:expect-raises -->
```python
@cash.cache(strict=True)
def fetch_user(uid):
    return requests.get(f"https://api/{uid}").json()

fetch_user(42)
# Traceback (most recent call last):
#   ...
# cash.CashImpureFunctionError: @cash.cache(strict=True) on
# __main__.fetch_user: purity issues detected. Either fix the function,
# mark callees with @pure / @stateful, or relax to assume_safe=True.
#   in __main__.fetch_user:
#     line 2: [impure_call] requests.get() — known I/O / side-effecting
```

In strict mode, opaque callees (functions whose source we can't read)
also count as issues — the paranoid setting.

`strict=True` and `assume_safe=True` are mutually exclusive; passing
both raises `ValueError` at decoration time.

### `cash.mark_pure(func)` and `cash.mark_stateful(func)`

The `@pure` and `@stateful` decorators wrap the function — convenient
for code you own, awkward for third-party callables (C extensions,
classes you can't subclass). Use `mark_pure` / `mark_stateful` to
annotate in-place without wrapping:

```python
import cash, pandas as pd

# We've audited and know this is fine — silence the analyzer for it.
cash.mark_pure(pd.DataFrame.merge)

# This one really does write to disk — tell the analyzer.
cash.mark_stateful(pd.DataFrame.to_sql)
```

Now any `@cash.cache`d function whose body calls `pd.DataFrame.merge`
won't flag on it, and any function whose body calls
`pd.DataFrame.to_sql` will.

### What the analyzer looks at

The decorator-side analyzer walks the function body AND
**module-bounded helpers** (functions defined in the same top-level
package, or any non-installed-library code) and any **closure-bound
helpers** reachable through `__globals__` / `__closure__`. For each,
it flags:

- **Impure calls** — `requests.post`, `os.system`, file writes,
  `logging.*`, pandas `inplace=True`, …
- **Dynamic patterns** — `eval`/`exec`/`compile`,
  `getattr(obj, name)()` with non-constant name, calling a parameter
  as a function
- **Discarded calls** — `f(x)` as a statement (return thrown away)
  when `f` isn't known-pure
- **Scope mutations** — `global`/`nonlocal`, attribute/subscript
  assignment, augmented-assign, and write-methods (`.append`, `.update`,
  …) on a name that could reach caller-visible state
- **Reads of a *mutated* module global** — if the function reads a
  module-level variable that is reassigned or mutated somewhere in its
  module, the cached result won't reflect changes to it. Only globals that
  are actually written are flagged — a constant (a lookup/dispatch table you
  never modify) is fine. Fix by passing the value as an argument or declaring
  it via `depends_on=`/`dynamic_depends_on=`. The detection is scope-aware: a
  local that merely shares a name with a global doesn't trip it.

In-place mutation of a **fresh local** is *not* flagged. A name bound only
to a freshly-allocated mutable object — a list/dict/set literal or
comprehension, or a known constructor like `[]`, `dict()`, `np.zeros(...)`,
`pd.DataFrame(...)`, `.copy()` — cannot alias the caller's state, so
`pos = np.zeros(n); pos[i] = ...` and `lines = []; lines.append(...)` are
pure. Mutating a parameter, an alias of one (`x = data; x.append(...)`), or
module/enclosing state still flags.

Stops at library boundaries (anything under `site-packages` /
stdlib) — those are trusted unless you `mark_stateful` them
explicitly. A call to another `@cash.cache`-decorated function is treated
as a dependency-graph edge, not walked into.

A nice side-effect of the same walk: helper source hashes flow into
the cache key. Edit a helper (in another file, or even live in a
notebook cell), and the parent's cache invalidates automatically.

## API reference (compact)

| Symbol | Import path | Type | Effect |
|---|---|---|---|
| `pure` | `from cash import pure` | decorator | Sets `_cash_pure = True`. Cash skips the stateful check for bare-name calls to this function. |
| `stateful` | `from cash import stateful` | decorator | Sets `_cash_stateful = True`. Cash refuses to cache any cell that calls this by bare name. |
| `mark_pure(func)` | `from cash import mark_pure` | in-place marker | Sets `_cash_pure = True` on *func* without wrapping. For third-party callables. |
| `mark_stateful(func)` | `from cash import mark_stateful` | in-place marker | Sets `_cash_stateful = True` on *func* without wrapping. |
| `is_pure(func)` | `from cash import is_pure` | bool | Marker-only check. Does not analyze source. |
| `is_stateful(func)` | `from cash import is_stateful` | bool | Marker-only check. |
| `analyze_function_purity(func, user_ns=None)` | `from cash import analyze_function_purity` | bool | AST-based heuristic. Result is SHA-256-cached. |
| `is_known_pure(name)` | `from cash.notebook.purity import is_known_pure` | bool | Membership check against the builtin allow-list. **Takes a string.** |
| `KNOWN_PURE_BUILTINS` | `from cash.notebook.purity import KNOWN_PURE_BUILTINS` | `frozenset[str]` | The stdlib allow-list. |
| `clear_purity_cache()` | `from cash.notebook.purity import clear_purity_cache` | `None` | Clears the SHA-256 result cache. Testing/debug use. |
| `CashImpurityWarning` | `from cash import CashImpurityWarning` | warning class | Emitted by `@cash.cache` (default mode) when the analyzer finds issues. Subclasses `CashCacheIneffectiveWarning`. |
| `CashImpureFunctionError` | `from cash import CashImpureFunctionError` | exception class | Raised by `@cash.cache(strict=True)` on any purity issue, **and by a plain `@cash.cache` on untrackable-dependency patterns** (`eval`/`exec`, dynamic `getattr(...)()`, `importlib`). `assume_safe=True` suppresses it. |

## Related

- [Decorator (`@cash.cache`)](../../decorator.md) — full decorator walkthrough with purity-mode parameters.
- [Annotations](../../annotations.md) — statement-level `@cash:persist` / `@cash:no-cache` / `@cash:ttl=N`.
- [Reading the Cash Badge](../../badges.md) — badge shows "Calls @stateful function" as a miss reason.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — how annotations and TTL interact with purity.
- [Notebook Caching API](../../notebook_caching_api.md) — big-picture decision flow.
