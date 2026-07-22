# Known limitations (writing cache-safe cells)

Cash restores a cell's inputs by re-deriving the state its upstream cells produced. That works because it can see what each statement reads and writes. Where it *cannot* see a change — because the change happened through a channel it does not track — it will hand you a value that looks plausible and is wrong.

In practice this page doubles as the checklist for **writing cache-safe cells**: every item below is a pattern that can slip past cash, paired with the one-line habit that avoids it (rebind instead of mutate, seed the RNG, keep state in variables). If you read one page before writing cached notebooks, read this one.

This page lists every such case we know about, what you actually see, and what to do instead. If you hit something here, it is a known limitation rather than a broken install. If you hit something **not** here, please [open an issue](https://github.com/galgtonold/cash/issues) — that is exactly the report we want.

!!! info "One trigger dominates this page"
    Most correctness limitations below need an **isolated re-run**: running a single cell on its own, rather than running the whole notebook top-to-bottom. A full `Run All` re-executes each cell in order, so cash never has to reconstruct anything and these cases do not arise. When a limitation is isolated-re-run-only, it says so.

---

## Cached randomness is replayed, not redrawn

The single most-often-misread behaviour, and it is working as designed.

An unseeded random draw that is expensive enough to cache **is** cached. Re-running the cell returns the *same* numbers, because you are seeing a restored value rather than a fresh draw:

<!-- test:skip reason="illustrative: demonstrates replayed randomness across re-runs" -->
```python
import numpy as np

x = np.random.rand(200_000)   # cached; every re-run returns the identical array
```

Cash warns the first time this happens:

> Unseeded randomness restored from cache … The value you are seeing is a replay of an earlier run, not a fresh draw — re-running will not change it.

**Why it looks inconsistent.** Cheap draws behave differently. A statement that runs in well under 10 ms falls below the cost floor and is never cached at all, so `random.random()` *does* give you a new number each time while `np.random.rand(200_000)` does not. The rule is the cost floor, not the kind of randomness.

**What to do**

| Goal | Do this |
|---|---|
| A fresh draw every run | `# @cash:no-cache` on the statement |
| Reproducible results | Seed the RNG (`np.random.seed(0)`) — then the replay *is* the correct value |
| Keep the cache, silence the warning | `# @cash:allow-random` |

!!! warning "`allow-random` does not change caching"
    `# @cash:allow-random` suppresses the *warning* only. The statement is still cached and still replayed. If you want a fresh draw, you need `no-cache`.

### Editing a bare `seed()` cell without re-running it

Cash tracks a global re-seed when the seeding statement itself **executes** — in the same cell as the draw, or when you re-run the seed cell. Editing a bare `np.random.seed(...)` (or `random.seed(...)`) cell and running **only** a downstream draw is the one case it cannot reconstruct:

<!-- test:skip reason="illustrative: cross-cell bare-seed edit, needs a real kernel" -->
```python
np.random.seed(0)         # cell 1 — edit to seed(1) but do NOT re-run this cell
x = np.random.rand(10**6) # cell 2 — run this alone
```

`x` does not reflect `seed(1)`: a bare `seed()` produces no variable, so cash has no dependency edge from the draw back to the seed cell, and its upstream reconstruction — which rebuilds ordinary *variables* from an edited-but-not-rerun cell correctly — has nothing to rebuild here. The draw is served stale, or recomputed against whatever global state the kernel is in. This matches what you'd get with caching off, but it does **not** match a clean top-to-bottom run.

**What to do:** re-run the seed cell after editing it (then the draw refreshes correctly), or seed in the same cell as the draw, or use a named generator (below).

### Per-object generators (`np.random.default_rng`) are only partially tracked

The **module-global** RNG channels — `np.random.*`, `random.*`, `torch.*` — are fully tracked: a draw is flagged on the badge (a `random` / `unseeded` pill), an unseeded draw's cached value is announced as a frozen replay, editing a `seed()` invalidates everything cached downstream, and a re-run reflects the position a clean top-to-bottom run would produce.

A **per-object generator** created with `np.random.default_rng()` (or `Generator(...)` / `RandomState(...)`) is a different, narrower story. Its **seed is tracked** — `rng = np.random.default_rng(SEED)` binds a variable, so editing `SEED` and re-running refreshes through the ordinary variable-lineage path, and an *unseeded* named generator (`rng = np.random.default_rng()`) drawn from by name is flagged. But three things about a per-object generator are **not** tracked:

**1. Stream position across cells.** cash does not follow a generator's internal position as several cells draw from it:

<!-- test:skip reason="illustrative: cross-cell Generator stream position, needs a real kernel" -->
```python
rng = np.random.default_rng(0)   # cell 1
a = rng.normal(size=3)           # cell 2 — draws positions 0–2
b = rng.normal(size=3)           # cell 3 — draws positions 3–5
```

Re-running cell 2 (or cell 3) *alone* draws from wherever the live generator happens to be, not the position a top-to-bottom run would produce — the value matches a plain kernel re-run, not cash's usual top-to-bottom contract. (A full `Run All` is correct; the gap is isolated re-runs.) This is the one place cash's per-object handling differs from the global channel, which *is* position-aware.

**2. Anonymous inline draws** get no signal at all. Constructing and drawing in one expression binds no name:

<!-- test:skip reason="illustrative: anonymous inline generator draw" -->
```python
z = np.random.default_rng().standard_normal(3)   # no pill, no warning
```

Because the generator has no name to track and draws from OS entropy (it never touches the module-global state the runtime observer watches), cash sees nothing: no `unseeded` pill, no warning, and the value is cached and frozen like any other unseeded draw — silently.

**3. Draws reached indirectly.** A generator drawn from inside a called function (`arr = make_data()` where `make_data` does `np.random.default_rng().normal(...)`), or via an attribute (`self.rng.normal()`), or handed back from a helper (`g = make_rng(); g.normal()`), is invisible for the same reasons — static analysis can't see the receiver and the module-global observer sees no change.

**What to do:** for reproducibility, **seed the generator** (`np.random.default_rng(42)`) — a seeded generator's cached value is the correct frozen value, and its seed *is* tracked. For a value that must be fresh, mark the cell `# @cash:no-cache`. If you need cash's full position-aware tracking and flagging, use the **module-global** functions (`np.random.seed(42)` + `np.random.rand(...)`) rather than a per-object generator. Constructing and drawing in the same cell also keeps a `Run All` correct.

---

## Mutation that cash cannot see

These all share one shape: a cell changes an object through a path cash does not attribute to that object, so on an isolated re-run the change is applied a second time or not undone.

**They are isolated-re-run only.** `Run All` is unaffected.

### Mutating through an alias

Cash tracks mutation through the name an object was bound to. Reach the same object through a different name and the mutation is invisible — re-running the cell applies it twice:

<!-- test:skip reason="illustrative: alias-mutation shapes, need isolated cell re-runs" -->
```python
b.ref = x           # attribute store
b.ref.append(99)    # ← cash does not know this touched x

t = (lst,)          # container element
t[0].append(3)      # ← nor this

(y := x).append(3)  # walrus as method receiver

y = x if flag else z   # ternary: two possible sources
y.append(3)
```

Nested unpacking has the same hole one level down — `(p, (q,)) = (x, (y,))` then `q.append(9)` double-applies, though the flat `(y,) = (x,)` form is handled.

**What to do:** mutate through the original name (`x.append(99)`), or rebind rather than mutate (`x = x + [99]`).

### Mutating global state inside a function

Cash analyses what a *statement* reads and writes. It cannot see inside a function you call:

<!-- test:skip reason="illustrative: hidden global mutation, needs an isolated cell re-run" -->
```python
c = {'n': 0}                    # cell 1
def tick():
    c['n'] += 1                 # mutates a global cash cannot attribute
    return c['n']

r = tick()                      # cell 2 — re-run alone: prints 2, not 1
```

This is a fundamental limit of analysing impure functions statically. Note that `@stateful` does **not** rescue it: it forces the call to re-execute, but the call still reads an already-advanced global.

**What to do:** pass the state in and return it out, rather than mutating a global.

### Mutating an object created in an earlier cell

Modifying an object in place, when that object was created *upstream*, re-runs the modifying statement rather than restoring it:

<!-- test:skip reason="illustrative: contrasts in-place mutation with rebinding across cells" -->
```python
df = pd.read_csv('data.csv')       # cell 1

df['score'] = expensive(df)        # cell 2 — recomputes on isolated re-run
df = df.assign(score=expensive(df))  # ← rebinding restores from cache instead
```

The trigger is **mutating an upstream object**, not subscript assignment: the same `df['score'] = ...` on a frame built in the *same* cell caches normally.

**What to do:** rebind (`df = df.assign(...)`) when the frame came from another cell.

### Mutating and reassigning the same name in one cell

<!-- test:skip reason="illustrative: mutate-then-reassign raises on isolated re-run" -->
```python
df['c'] = df['a'] * 2                    # mutate
df = df.rename(columns={'a': 'x'})       # then reassign the same name
```

An isolated re-run raises **`KeyError`**, because the re-run reads the already-renamed frame. **What to do:** split into two cells.

### Class variables shared across cells

A class attribute incremented by both an upstream cell and the cell you re-run keeps accumulating, because cash suppresses re-running the class definition to avoid clobbering the upstream cell's contribution:

<!-- test:skip reason="illustrative: cross-cell class-variable accumulator" -->
```python
class Widget:                    # cell 1
    count = 0
    def __init__(self):
        Widget.count += 1
w0 = Widget()

w = Widget()                     # cell 2 — re-run: count climbs past 2
```

Note this needs the counter to be advanced from *both* cells. A class variable only appended to from one cell is handled correctly.

### A function that calls one defined in a later cell

cash follows dependencies *upward*. A function whose body calls a name bound in a
**later** cell has a dependency pointing down the notebook, and editing that
later function does not refresh the call site:

```python
def a(n): return b(n) * 2      # cell 1 — b doesn't exist yet
def b(n): return n + 1         # cell 2
r = a(3)                       # cell 3 -> 8

# edit cell 2 to `return n + 10`, re-run cell 3 only -> still 8, not 26
```

Order is the whole story — write `b` **above** `a` and the same edit propagates
correctly, because `def a` then names `b` as an ordinary input. This is not about
recursion, though mutual recursion always trips it, since each function
references the other before it exists.

**What to do:** define a function above the ones that call it — the order Python
readers expect anyway. If you must keep the order, re-run the defining cell (or
`Run All`) after editing it rather than the call site alone.

### Background threads

A thread that mutates data after the cell that created it has finished is outside cash's view entirely. Re-running an earlier cell can observe the mutated state instead of the state at that point in the notebook.

---

## Errors you may see

### `AmbiguousCellError`

> Ambiguous cell execution! The current cell content appears 2 times in the notebook and no cell ID could be resolved.

Raised when two cells have **byte-identical content** *and* cash cannot resolve a cell ID. Cash fails loudly here rather than guessing, because guessing wrong would silently serve one cell's result for the other.

In JupyterLab and VS Code with IPython ≥ 8.3, cell IDs normally resolve and this does not occur. It shows up in environments that do not supply them.

**What to do:** make the cells distinguishable (a comment is enough), or save the notebook so IDs resolve.

---

## When caching is less effective than you expect

These cost you time. None of them gives a wrong answer.

### Chained file-writing cells re-execute each other

The one worth planning around. If you have N cells that each read a file, modify it, and write it back, a single `Run All` costs **N(N+1)/2** executions rather than N — each writer re-executes every preceding writer's write statement. With 4 such cells, executions per run are `{a: 4, b: 3, c: 2, d: 1}`, and this persists on an unedited notebook.

**What to do:** where possible, write to distinct outputs rather than round-tripping one file through a chain of cells.

### A long `for`-append loop can stop caching

Cash normally caches a loop **per iteration**, so a warm re-run restores every one. A *long* loop can instead be cached as a **single unit** — per-statement bookkeeping stops paying for itself. That switch is reasonable on its own, but a loop that appends into a list is an in-place mutation, which cash will not cache as a whole unit, so the two combine and you get no caching at all.

Three conditions must hold together before the switch happens, which is why many append loops never hit it:

- **more than ~50 iterations**, and
- **estimated bookkeeping above ~1 second** — roughly `iterations × statements-in-body × 8ms`, so a multi-statement body can qualify just past the 50 mark while a **one-line body does not until ~125 iterations** — and
- **no file I/O written directly in the loop body** (a call to a function that does the I/O internally does not count — only I/O written in the body itself).

So the trigger is driven by the *number of statements* cash would have to track, not by how slow the loop is. A one-line append loop over a slow function is one of the shapes that generally keeps caching:

<!-- test:skip reason="illustrative: contrasts two loop shapes; entities/fetch are the reader's own" -->
```python
out = []
for e in entities:          # one-line body: keeps per-iteration caching
    out.append(fetch(e))    # well past 100 iterations

out = []
for e in entities:          # several statements per iteration, >50 of them,
    rec = fetch(e)          # no I/O written here -> may switch to single-unit
    rec["seen"] = True      # and then cache nothing, because `out` is
    out.append(rec)         # mutated in place
```

When it does happen the badge says so (`Storage uncacheable · Reason: In-place mutation on: out`) and gives the fix, so you are not misled about *whether* it cached — but nothing tells you which threshold you crossed.

**What to do:** assign the result instead of appending to it. A comprehension is cached as a single value at any length, and sidesteps the question entirely:

<!-- test:skip reason="illustrative: the comprehension rewrite of the loop above" -->
```python
out = [fetch(e) for e in entities]      # cached regardless of length
```

The same applies to `while` loops that accumulate, and to `for` loops that build a dict or frame by mutation.

If you are unsure which side of the line a particular loop is on, do not infer it from the iteration count — check `%cash_stats` or the badge, which report what actually happened.

### Editing without saving

Editing a cell in VS Code without saving the notebook file can make the cell-ID match fail, which skips the upstream check and misses the cache for statements that did not change. Values stay correct. **What to do:** save the notebook.

### Others

- **`from math import pi`-style imports** can block restore-after-restart where a plain `import math` restores.
- **A legitimately empty cached value** (an empty list or frame) may be recomputed rather than restored.
- **`%%cash` cell magic** does not reset cell-entry lineage on re-run, costing a recompute.

---

## Large objects are hashed by sampling

To keep hashing cheap, cash samples large values rather than reading them whole:

| Type | What is hashed |
|---|---|
| DataFrame | shape, dtypes, **first 5 rows** |
| ndarray | first 100 elements |
| list / tuple (> 200) | length, first 5, last 5 |
| dict (> 200) | length, first 10 keys — **values are not hashed** |

Two large objects that differ only outside the sampled region therefore hash identically. In normal use cash tracks provenance and does not rely on the hash alone, so this is latent; it becomes reachable only after provenance is lost (for example following `cash.reset_session()`), where a change to row 700 of a 1000-row frame can go unnoticed.

**What to do:** if you reset cash state mid-session and then mutate deep inside a large object, restart the kernel rather than relying on invalidation.

!!! note "This sampling is the notebook lineage path, not `@cash.cache` arguments"
    The table above describes how the **notebook** path fingerprints a *tracked
    variable* for lineage. The `@cash.cache` **decorator** hashes its arguments
    by pickling their full content, so a deep mutation to a large DataFrame
    argument (e.g. row 700 of 1000) *is* detected and produces a fresh cache
    entry. The sampling blind spot does not reach the decorator's argument hash.

---

## Reporting something not on this page

Please open an issue with the cell sequence that triggers it and what you expected. Two things make a report immediately actionable:

- **Whether it reproduces on `Run All`** or only on an isolated single-cell re-run — that distinction separates most limitation classes above.
- **A count, not a stopwatch.** Wall-clock cannot distinguish "recomputed" from "restored but slow". A counter that a cached function appends to tells us in one run what timings cannot settle in ten.
