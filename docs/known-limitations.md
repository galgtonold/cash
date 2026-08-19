# Known limitations (writing cache-safe cells)

Cash restores a cell's inputs by re-deriving the state its upstream cells produced. That works because it can see what each statement reads and writes. Where it *cannot* see a change — because the change happened through a channel it does not track — it will hand you a value that looks plausible and is wrong.

In practice this page doubles as the checklist for **writing cache-safe cells**: every item below is a pattern that can slip past cash, paired with the one-line habit that avoids it (rebind instead of mutate, seed the RNG, keep state in variables). If you read one page before writing cached notebooks, read this one.

This page lists every such case we know about, what you actually see, and what to do instead. If you hit something here, it is a known limitation rather than a broken install. If you hit something **not** here, please [open an issue](https://github.com/galgtonold/cash/issues) — that is exactly the report we want.

!!! info "One trigger dominates this page"
    Most correctness limitations below need an **isolated re-run**: running a single cell on its own, rather than running the whole notebook top-to-bottom. A full `Run All` re-executes each cell in order, so cash never has to reconstruct anything and these cases do not arise. When a limitation is isolated-re-run-only, it says so.

---

## Cached randomness is replayed, not redrawn

The single most-often-misread behaviour, and it is working as designed.

<!-- claim: cash/notebook/statement/restore.py:StatementRestorer.restore_from_cache @0388af0f, cash/notebook/randomness.py:restore_rng_state @ccba2493 -->
An unseeded random draw that is expensive enough to cache **is** cached. Re-running the cell returns the *same* numbers, because you are seeing a restored value rather than a fresh draw:

<!-- test:skip reason="illustrative: demonstrates replayed randomness across re-runs" -->
```python
import numpy as np

x = np.random.rand(200_000)   # cached; every re-run returns the identical array
```

Cash warns the first time this happens:

> Unseeded randomness restored from cache … The value you are seeing is a replay of an earlier run, not a fresh draw — re-running will not change it.

<!-- claim: cash/notebook/cost_model.py:_COEFFS @19a78800 broad="the claim is that a cost floor, not the kind of randomness, decides caching" -->
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

<!-- claim: cash/notebook/randomness.py:RNG_CARRIER_CONSTRUCTORS @3248b870, cash/notebook/randomness.py:capture_object_rng_states @b4e01b79 -->
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

<!-- claim: cash/notebook/cacheability.py:bare_alias_targets @b6d520b0, cash/notebook/cacheability.py:reference_alias_targets @0e0de16b -->
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

<!-- claim: cash/notebook/cacheability.py:_literal_unpack_aliases @b9e32ac9 -->
Literal unpacking — flat (`(y,) = (x,)`) *and* nested (`(p, (q,)) = (x, (y,))`) — is **not** in this list: cash recognises every leaf of a 1:1 literal unpack as a pointer copy, at any nesting depth, and refuses to cache the statement. A later `q.append(9)` is therefore not double-applied. One `*rest` or one computed element (`b, c = a, f()`) opts the whole statement back out, since that element may be real work worth caching.

**What to do:** mutate through the original name (`x.append(99)`), or rebind rather than mutate (`x = x + [99]`).

### Mutating global state inside a function

<!-- claim: cash/notebook/cacheability.py:called_function_global_mutations @e5298503 -->
Cash analyses what a *statement* reads and writes, and it tracks the **arguments**
a called function mutates — including imported helpers and bare calls (`proc(df)`
that mutates `df`). It also tracks a function mutating a **global** it wasn't
handed, at cell level, in every spelling of the call:

<!-- test:skip reason="illustrative: hidden global mutation, needs an isolated cell re-run" -->
```python
c = {'n': 0}                    # cell 1
def tick():
    c['n'] += 1                 # a global, mutated from inside the callee
    return c['n']

r = tick()                      # cell 2 — re-runs and restarts agree with a
                                # clean top-to-bottom run
```

Cash does not *replay* the `+= 1`. It treats the statement exactly as it treats
the same mutation written inline: the statement is **not cached**, so it
re-executes and the write really happens. The expensive part is still cached —
the *call* inside the statement is served from cache and its effect on `c` is
restored with it — so what re-runs is the glue, not the work. Re-running cell 2
alone, or restarting the kernel and running everything, both land where a clean
top-to-bottom run lands.

**The same write from inside a loop body is tracked too**, including across
cells: re-running an *earlier* loop cell discards a *later* one's contributions
and lands where a clean top-to-bottom run up to that cell lands.

<!-- test:skip reason="illustrative: cross-cell loop re-run, needs a real kernel" -->
```python
LOG = []
def step(v):
    LOG.append(v)          # captured per call, restored along with the result
    return v * 10

for x in [1, 2, 3]:        # cell A
    out.append(step(x))

for x in [111, 10]:        # cell B
    total += step(x)

# re-run cell A -> LOG holds [1, 2, 3], not cell B's entries
```

**What it costs: nothing, for a reorder.** The global's post-state is captured
with the call and replayed in the *new* order, so reordering the loop's items
keeps the order-independent reuse described under
[Reordering a loop's items](#reordering-a-loops-items-re-runs-the-tail).
Measured on a real kernel — `for v in vols: out.append(price(v))`, where
`price` appends to a global `SEEN`:

| | `SEEN` | `out` |
|---|---|---|
| cold, `vols = [1, 2, 3]` | `[1, 2, 3]` | `[2, 4, 6]` |
| reordered to `[2, 1, 3]` | `[2, 1, 3]` | `[4, 2, 6]` |
| a clean run with no cache | `[2, 1, 3]` | `[4, 2, 6]` |

The callee does not run again; the recorded write is restored in the order the
new sequence implies. So a global write costs you the *statement's* caching —
the statement re-executes, as described above — but not the call's.

**What to do:** nothing. If you would rather not depend on the capture at all,
pass the state in and return it out instead of mutating a global, which also
makes the function a pure function of its arguments.

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

It is not only staleness. Because the call site never learns it depends on `b`,
cash can serve a cached value for code that **no longer runs at all**: delete
cell 2, restart, and `Run All` reprints `r=8` where a plain kernel raises
`NameError`. With `b` defined *above* `a`, the same deletion correctly raises.

**What to do:** define a function above the ones that call it — the order Python
readers expect anyway. If you must keep the order, re-run the defining cell (or
`Run All`) after editing it rather than the call site alone.

### Background threads

A thread that mutates data after the cell that created it has finished is outside cash's view entirely. Re-running an earlier cell can observe the mutated state instead of the state at that point in the notebook.

### Reads through a loader cash cannot see

<!-- claim: cash/notebook/file_tracker.py:_install_module_patches @4cabaa21 -->
Cash records a file dependency by intercepting the *read*: `pd.read_*`, `np.load`, `joblib.load`, `polars`, plain `open()`, and friends. A read that goes through none of those — a C extension that opens the file itself, a third-party client, a `subprocess` — is invisible.

The consequence is easy to mis-guess, so it is worth stating plainly: cash **does not** refuse to cache such a statement. It caches it exactly like any other, with *no file recorded*. Change the file on disk afterwards and nothing invalidates; you get the old value back with a `CACHED` badge and no warning.

<!-- test:skip reason="illustrative: my_reader stands in for the reader's own untracked loader" -->
```python
data = my_reader.load('sensor.bin')    # cash sees a value, not a file read
# ...edit sensor.bin on disk...
data = my_reader.load('sensor.bin')    # CACHED — the old contents
```

**What to do:** name the file explicitly so cash tracks it anyway. In a notebook, add the read of a tracked API alongside it, or annotate the statement `# @cash:no-cache` if the read is cheap. In the decorator path, declare it: `@c.cache(file_depends_on="sensor.bin")`.

(`cash.register_hasher()` does *not* help here — it teaches the **decorator** how to hash an argument of a custom *type*, which is a different problem from fingerprinting a file on disk.)

---

## A loop variable mutated before it's read collides with an earlier iteration

Unlike the mutation cases above, this one is **not** isolated-re-run only — it can give a wrong answer on a fresh `Run All`, the first time the loop ever executes.

<!-- claim: cash/notebook/control_structures/for_handler.py:ForLoopHandler._process_one_iteration @3f828077 -->
Cash decomposes a `for` loop per iteration and uses the loop variable's value — captured at the moment it is *bound*, before any body statement runs — as the per-iteration cache discriminator. That applies both to an ordinary cached statement in the body and to an intercepted (on by default) sub-call whose own arguments give the key nothing else to vary on. If the body **mutates the loop variable before it is used**, the discriminator was already captured before that mutation and cannot see it:

<!-- test:skip reason="illustrative: pull() stands in for a slow call whose only per-iteration signal is the loop variable; call-level caching is on by default and needs no directive to make pull(handle) itself the cached, keyed unit" -->
```python
handle = 'conn-object'          # calling pull() carries no per-iteration signal of its own
for q in [[1], [1]]:            # two iterations, EQUAL at binding time
    q.append(len(accm))         # mutated here, before the next line runs
    accm.append(pull(handle))   # keyed on q's value as BOUND, not as mutated
```

Both iterations bind `q` to an equal value (`[1]`), so both get the same discriminator even though the body has since made them different — the second iteration is served the first's cached result instead of a fresh call.

<!-- claim: cash/notebook/call_unit.py:_loop_var_digest @c26d2458 -->
This is true of a plain cached statement in the loop exactly as it is of an intercepted call: `v = pull(handle)` on its own line, with no directive at all, collapses the same way, because both channels read the same value, frozen at the same moment. Neither spelling is a special case of the other.

**What to do:** the fix is not "mutate vs. rebind" — a body-local rebind is exactly as invisible as an in-place mutation:

<!-- test:skip reason="illustrative: rebinding q instead of mutating it looks like a fix but is not -- q is still a body-local variable, not a loop target" -->
```python
for q in [[1], [1]]:
    q = q + [len(accm)]          # rebound, not mutated -- STILL wrong, identically
    accm.append(pull(handle))
```

<!-- claim: cash/notebook/control_structures/processor.py:extract_target_names @f3f92993 -->
That's because the per-iteration cache key is built from `for`-loop **target names only** (`extract_target_names`, run once against the loop header), captured at the moment those names are bound. An ordinary body-local variable never reaches that key, however it's assigned — mutated, rebound, or read from an outer container makes no difference, because the mechanism never looks at the body at all.

The fix is to introduce the discriminating value as a **genuine additional loop target** instead. `enumerate` is the simplest way when the natural per-iteration signal is the iteration's own position — which it is here, since `len(accm)` grows by exactly one appended item per iteration:

<!-- test:skip reason="illustrative: pairs with the loop above; enumerate's `i` is a real for-loop target, unlike q" -->
```python
for i, base in enumerate([[1], [1]]):
    q = base + [i]
    accm.append(pull(handle))
```

`i` is bound directly by the `for` statement, so — unlike `q` — it *is* captured into the digest at binding time, and correctly differs across iterations.

**To diagnose your own case:** only names bound directly by a `for` statement's target reach the per-iteration cache key. If the value that has to discriminate two iterations lives anywhere else — a body-local variable, however it's produced — it will not reach that key, and no amount of rebinding it earlier in the body changes that. If there's no natural value to promote into the loop's target (`enumerate`, `zip` with a counter, or restructuring the iterable itself), mark the affected statement or call `# @cash:no-cache` instead.

---

## Errors you may see

### `AmbiguousCellError`

> Ambiguous cell execution! The current cell content appears 2 times in the notebook and no cell ID could be resolved.

<!-- claim: cash/exceptions.py:AmbiguousCellError @267a93a2, cash/notebook/upstream/checker.py:UpstreamChecker.check_and_reexecute @20807ce5 broad="the claim is about when this exception type exists to be raised at all" -->
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

Cash normally caches a loop **per iteration**, so a warm re-run restores every one. A *long* loop can instead be cached as a **single unit** — per-statement bookkeeping stops paying for itself. That switch is reasonable on its own, but a loop that appends into a list is an in-place mutation, which cash will not cache as a whole unit, so the two combine and you get no *statement-level* caching at all.

That is a narrower claim than it used to be. By default, cash also caches the expensive **call inside** the statement (`fetch(e)` below, not the `append` around it) — see [Call-level caching](annotations.md#call-level-caching-default-and-cashno-cache-calls-alias-nocachecalls) — so a single-unit append loop still isn't a total loss: the call itself keeps hitting even though the loop's own bookkeeping does not. `# @cash:no-cache-calls` turns that off and gets you back to "no caching at all" if you need to reproduce it, or the call site simply isn't eligible (it reads the loop's own accumulator, say).

<!-- claim: cash/notebook/control_structures/for_handler.py:ForLoopHandler._should_execute_loop_as_single_unit @ad980fd0, cash/notebook/control_structures/for_handler.py:ForLoopHandler._MIN_ITERATIONS_FOR_SINGLE_UNIT == 50, cash/notebook/control_structures/for_handler.py:ForLoopHandler._PER_STMT_OVERHEAD_SEC == 0.008, cash/notebook/control_structures/for_handler.py:ForLoopHandler._MIN_OVERHEAD_SEC == 1.0 -->
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

When it does happen the badge says so — the HTML badge's row detail shows a **Storage** field reading `uncacheable` and a separate **Reason** field naming `In-place mutation on: out`, and `%cash_badge print` appends the same reason to the row — so you are not misled about *whether* it cached. But nothing tells you which threshold you crossed.

**What to do:** by default there is nothing to do — cash already caches `fetch(e)` and lets the append re-run, with no directive needed, see [Call-level caching](annotations.md#call-level-caching-default-and-cashno-cache-calls-alias-nocachecalls). If you've disabled that (`# @cash:no-cache-calls`) or `fetch` isn't eligible, assign the result instead of appending to it — a comprehension is cached as a single value at any length, and sidesteps the question entirely:

<!-- test:skip reason="illustrative: the comprehension rewrite of the loop above" -->
```python
out = [fetch(e) for e in entities]      # cached regardless of length
```

The same applies to `while` loops that accumulate, and to `for` loops that build a dict or frame by mutation.

If you are unsure which side of the line a particular loop is on, do not infer it from the iteration count — check `%cash_stats` or the badge, which report what actually happened.

### Reordering a loop's items re-runs the tail

**This is now the exception, not the default** — read the callout after the
first table before assuming it applies to your loop. A loop body that folds into a **running accumulator** (`total += f(x)`, `acc = acc + f(x)`) reads the accumulator as an input, so iteration *k*'s **statement-level** cache key encodes every iteration before it. That much is still true, unconditionally — the fold's own key is a prefix property and always will be, no matter what you do with the call inside it.

What changed: by default, cash *also* caches the expensive call inside the fold, and a call cache keys on arguments, not on execution history — so it is order-independent by construction. The statement still misses and the loop still re-executes on a reorder, but the **work** doesn't repeat, because `compute(x)` hits its own entry regardless of position. No directive needed:

<!-- test:skip reason="illustrative: `compute` stands in for the reader's own slow function" -->
```python { .nb-cell }
s = 0
for x in [1, 10, 5]:        # compute() sleeps 1s
    s += compute(x)
```

Measured on exactly that cell, with the current default:

| Change to the list | `compute()` calls | Cost |
|---|---|---|
| `[1, 10]` → `[1, 10, 5]` (append) | just the new one | 1 s |
| `[1, 10, 5]` → `[1, 5, 10]` (swap the last two) | 0 | 0 s |
| `[1, 10, 5]` → `[5, 10, 1]` (new first element) | 0 | 0 s |
| back to `[1, 10, 5]` | 0 | 0 s |

Only a genuinely new value costs a call; any reordering of values cash has already seen costs nothing, regardless of position. See [Call-level caching](annotations.md#call-level-caching-default-and-cashno-cache-calls-alias-nocachecalls) for what qualifies — the same eligibility rule (the call must not read the statement's own fold target) applies here.

!!! note "This assumes the call is a function of its arguments"
    Keying on arguments is what makes a reorder free, so it holds for a callee that computes and returns. A callee that also **writes something outside itself** — appends to a global, writes a file — is re-executed instead, and a reorder costs full price for it.

    That is not a shortfall: such a function is not a function of its arguments, and serving it would leave the global holding the *previous* order's contents. If a reorder unexpectedly costs executions, look for a write escaping the callee. See [Mutating global state inside a function](#mutating-global-state-inside-a-function).

**If you disable it** (`# @cash:no-cache-calls`), or the call isn't eligible, you get the *statement*-level table this section used to describe unconditionally — the historical, pre-default-on shape:

| Change to the list | Iterations re-run | Cost |
|---|---|---|
| `[1, 10]` → `[1, 10, 5]` (append) | just the new one | 1 s |
| `[1, 10, 5]` → `[1, 5, 10]` (swap the last two) | 2 of 3 | 2 s |
| `[1, 10, 5]` → `[5, 10, 1]` (new first element) | all 3 | 3 s |
| back to `[1, 10, 5]` | none — those keys are still cached | 0 s |

<!-- test:skip reason="illustrative: pairs with the loop above; `compute` is the reader's own" -->
```python { .nb-cell }
s = 0
# @cash:no-cache-calls
for x in [5, 10, 1]:     # reordered: BOTH the statement AND compute(x) miss. 3s.
    s += compute(x)
```

This is the fold, not the loop: drop the accumulator (`y = compute(x)`, or a comprehension) and reordering was always fully cached even at the statement level, because each iteration then depends only on its own loop variable — that part of this entry never needed the default flip.

The same order-independent effect is also available by hand if you'd rather decorate the function directly — `@cash.cache` keys the same way and composes with `%cash_on`, and does not depend on the default:

<!-- test:skip reason="illustrative: pairs with the loop above; `compute` is the reader's own" -->
```python { .nb-cell }
@cash.cache
def compute(x):
    ...

s = 0
for x in [5, 10, 1]:     # reordered: the fold's per-iteration cache misses,
    s += compute(x)      # but every compute() call still hits. 0s.
```

### Editing without saving

Cash reads the cells it did *not* execute from the saved `.ipynb` on disk, so an
edit you haven't saved is invisible to it — **unless a live reader is available**.
Colab, JupyterLab and VS Code each have one, with the caveats below; where none
applies, this is how it shows up:

- **The upstream repair doesn't fire.** Edit a config cell, then run a downstream
  cell without saving: cash reads the old value, concludes nothing upstream
  changed, and restores the previous answer while your screen shows the new code.
- In VS Code the cell-ID match can also fail, which skips the upstream check and
  *misses* the cache for statements that did not change.

Values stay correct in the sense that you get what your kernel actually holds —
the same thing plain Jupyter would give you. What you lose is the safety net:
cash's upstream check is only ever as current as the file it read.

**What cash does now:** when the cell you run is itself unsaved, cash can prove
the file is behind and says so on the badge — a warning row naming the time the
file was last saved. That proof condemns the whole file, so the warning stands
until you save.

<!-- claim: cash/notebook/live_cells.py:handle_message @c2aa8d99, cash/notebook/server_discovery.py:_try_extension_cells @3418ae7c -->
**On JupyterLab, cash's own extension pushes your unsaved edits to the kernel.**
`pip install cash-lib` also drops a prebuilt JupyterLab extension
(`cash-live-cells`) into your environment, which JupyterLab discovers at startup
— no marketplace, no build step, no Node. It sends the notebook's current cell
sources to the kernel over a comm, and **flushes them before every
`execute_request` leaves the browser**. Shell-channel messages are processed
first-in-first-out, so the push is handled before the cell that follows it: the
fast case — edit a cell, immediately run a different one, never touch `Ctrl+S` —
is an ordering guarantee rather than a race you have to win. There is nothing to
enable and nothing to configure.

Measured on **JupyterLab 4.6.3, on Windows**: 20 consecutive edit-then-run
iterations with no pause between the last keystroke and the run, all 20 checked
against the value that existed only on screen, on two independent runs. **What
was not tested: Notebook 7, split installs, and a cold-kernel `Run All`.**
Notebook 7 is built on the same JupyterLab 4 extension API, but we have not run
it there, so treat it as unverified rather than working.

**The first execution on a fresh kernel still reads the saved file.** The
extension's comm cannot open until the kernel has a target to open it against,
and that target does not exist until the cell containing `import cash` has run —
so the first attempt is necessarily refused, and the extension re-opens on the
next execution. Live cells therefore apply from the **second** execution onward.
The consequence worth knowing: **a `Run All` on a fresh kernel gets no live cells
for the whole run**, because JupyterLab queues every `execute_request` up front,
before the comm exists. Save before a cold `Run All`.

**The extension is discovered from the *server's* environment, not the kernel's.**
If you launch JupyterLab from one environment and run the notebook against a
kernel from another — a split install — the server never loads the extension,
nothing is pushed, and cash falls back to the saved `.ipynb`. Nothing breaks, and
it is not quiet about it: the once-per-session badge notice fires, as it does for
anyone reading the saved file.

**Turning it off.** `jupyter labextension disable cash-live-cells`, then reload
the page: cash falls back to the saved file exactly as it does for a user who
never had the extension. Getting back is less obvious, because JupyterLab
*locks* an extension when it disables one, and a plain `unlock` then refuses with
*"locked at a higher level"* even when the only config on the machine is the one
`disable` just wrote. The way out is to name the level:

```bash
jupyter labextension unlock cash-live-cells --level=system
jupyter labextension enable cash-live-cells
```

That lock behaviour is JupyterLab's own and applies to any extension, not just
this one.

<!-- claim: cash/notebook/server_discovery.py:_labextension_installed @ce644745 -->
Disabling does not remove the installed directory, and the proactive `Ctrl+S` tip
`%cash_on` prints is gated on that directory being present — so a disabled
extension, like a split install, keeps the tip suppressed. You are still told,
just reactively rather than up front: the once-per-session "cash cannot see
unsaved edits here" notice keys on the read that actually happened, so it is
right in both cases.

<!-- claim: cash/notebook/vscode_backup.py:live_cells @b86cd33f, cash/notebook/staleness.py:StalenessTracker.take_unverifiable_announcement @fecc0722 -->
**On VS Code, cash reads your unsaved edits directly.** VS Code keeps dirty
editors in a backup file so it can restore after a crash, and cash reads its
cells from there instead of the saved `.ipynb` — so editing one cell and running
a different one is checked against what is on screen, not what was last saved.
It only uses that backup when the backup's own record of the saved file still
matches the file on disk, so it can never substitute one stale copy for another.

This relies on an internal detail of VS Code that could change in a future
release. When it does — or when the backup is missing because hot exit is
disabled — cash falls back to reading the saved file and says so once, rather
than silently losing the guarantee.

**What it still cannot see:** editing one cell and running a *different* one,
wherever no live reader applies — a JupyterLab session with no working extension
(split install, extension disabled, or the first execution on a fresh kernel),
and a VS Code session with no usable backup. There the cell you ran matches what
cash read, so there is nothing to compare and no warning — a real hole, not an
oversight, since there is no other copy of the notebook to check against. It is
no longer silent about the gap, though: the first time a session cannot verify
freshness this way, cash says so once on the badge.

**What to do:** save the notebook (`Ctrl+S` / `Cmd+S`) after editing a cell you
aren't about to run, in any session where cash has no live reader — and before a
cold `Run All` on JupyterLab. Autosave exists in JupyterLab but runs on a timer,
so a quick edit-then-run lands inside the window. **Google Colab is exempt** —
there cash reads cells live from the frontend via `get_ipynb`, so there is
nothing to save. **So are JupyterLab with the bundled extension, and a VS Code
session with a usable hot-exit backup** — both above. If you are unsure which
you have, the badge settles it: cash says *"cash cannot see unsaved edits here"*
once per session, and only when it is reading the saved file.

### Others

- **`from math import pi`-style imports** can block restore-after-restart where a plain `import math` restores.
- **A legitimately empty cached value** (an empty list or frame) may be recomputed rather than restored.
- **`%%cash` cell magic** does not reset cell-entry lineage on re-run, costing a recompute.

---

## Large objects are hashed by sampling

To keep hashing cheap, cash samples large values rather than reading them whole:

<!-- claim: cash/notebook/object_hashing.py:compute_hash @61e351a4 -->
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
