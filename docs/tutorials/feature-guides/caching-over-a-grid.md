# Caching over a grid — what refining an axis can and can't reuse

You have an expensive computation over a grid, and you're tuning the grid.
Sometimes you go back to a resolution you already tried; sometimes you make it
finer. Cash handles those two cases very differently, and the difference is not
obvious from the outside.

This page is short on theory and long on one question: **does the refined axis
still contain the points you already computed?** Everything follows from that.

## Going back to a resolution you already ran is already free

This is the case that matters most and needs no technique at all.

```python
import cash
import numpy as np

c = cash.Cash(cache_dir="./.cash-grid-demo")
CALLS = []

@c.cache(assume_safe=True)
def field(axis):
    CALLS.append(len(axis))
    return np.sin(np.arange(1, 400)[:, None] * axis[None, :]).sum(axis=0)

field(np.linspace(0.0, 1.0, 200))     # first call — computes
field(np.linspace(0.0, 1.0, 100))     # a fresh grid — computes
CALLS.clear()
field(np.linspace(0.0, 1.0, 200))     # back to 200 — cache hit
assert CALLS == []                    # nothing ran
```

<!-- claim: cash/core.py:Cash._try_hash_numpy @ca38104d -->
`np.linspace(0.0, 1.0, 200)` reproduces the same array bit for bit every time,
so the third call is a plain cache hit. Sweeping a resolution downward until
accuracy breaks, then stepping back to the last good one, is exactly this
pattern — and it costs nothing to get.

## Refining is a different question

```python
CALLS.clear()
field(np.linspace(0.0, 1.0, 240))     # refined from 200 — recompute
assert CALLS == [240]                 # all 240 recomputed
```

<!-- claim: cash/core.py:Cash._hash_arg_payload @8be5a896 -->
The axis is a *single argument*. Ask for 240 points instead of 200 and every
value in the array is different, so there is no earlier result that corresponds
to any part of it. Cash isn't declining to reuse the old work; there is no old
work that matches.

The instinct — *"I only made it finer, keep the points I had"* — assumes the
refined grid contains the coarse one. Usually it doesn't:

```python
coarse = np.linspace(0.0, 1.0, 200)
assert len(np.intersect1d(coarse, np.linspace(0.0, 1.0, 240))) == 2
```

**Two points survive**, the endpoints. `linspace(0, 1, 240)` isn't `linspace(0,
1, 200)` plus 40 more — it is 240 entirely different coordinates. No caching
strategy can find reuse that isn't there.

## The precondition

Reuse across a refinement is possible only when the new axis **bitwise
contains** the old one. Two grid constructions have that property:

```python
# Extending the domain at a fixed step: the old axis is a prefix.
dx = 1.0 / 200
assert np.arange(0.0, 1.0, dx).tobytes() == np.arange(0.0, 2.0, dx)[:200].tobytes()

# Doubling a linspace: every old point survives, interleaved with new ones.
assert len(np.intersect1d(np.linspace(0.0, 1.0, 200),
                          np.linspace(0.0, 1.0, 399))) == 200
```

If your refinement doesn't have this property, stop here — the rest of this
page won't help, and changing *how you build the axis* is the real fix.

## Recipe 1 — chunk by index, when the old axis is a prefix

Evaluate the axis in fixed-size blocks and cache each block. Extending the
domain then only computes the new blocks.

Written out block by block, so you can see which ones run:

```python
CHUNK = 100

@c.cache(assume_safe=True)
def field_chunk(block):
    CALLS.append(len(block))
    return np.sin(np.arange(1, 400)[:, None] * block[None, :]).sum(axis=0)

axis = np.arange(0.0, 1.0, dx)                  # 200 points = 2 blocks
field_chunk(axis[0:CHUNK])                      # first call — computes
field_chunk(axis[CHUNK:2 * CHUNK])              # fresh block — computes

wider = np.arange(0.0, 2.0, dx)                 # 400 points = 4 blocks
CALLS.clear()
field_chunk(wider[0:CHUNK])                     # cache hit — same block as before
field_chunk(wider[CHUNK:2 * CHUNK])             # cache hit
field_chunk(wider[2 * CHUNK:3 * CHUNK])         # new territory — computes
field_chunk(wider[3 * CHUNK:4 * CHUNK])         # computes
assert CALLS == [CHUNK, CHUNK]                  # only the two new blocks ran
```

In real code you'd write the loop rather than the four calls:

<!-- test:skip reason="the block-by-block version above is the executed one; this is the same thing rolled into a loop" -->
```python
def field_chunked(axis):
    return np.concatenate([field_chunk(axis[i:i + CHUNK])
                           for i in range(0, len(axis), CHUNK)])
```

Pick `CHUNK` large enough that numpy still does real work per call — 10⁴ to 10⁶
elements is a reasonable band — and small enough that a localised change
doesn't invalidate the whole axis.

**This recipe does nothing for interleaved refinement.** Doubling a linspace
puts the old points at even indices, so no block matches a block you cached
before, and everything recomputes despite every old point being present.

## Recipe 2 — split into "what I had" and "what's new"

The general form. It finds the old work whenever the new axis contains the old
one, however the new points are interleaved.

```python
def field_split(axis, previous):
    new = np.setdiff1d(axis, previous)
    old_values = field(previous)                      # cache hit: same axis as before
    out = np.empty(len(axis), dtype=old_values.dtype)
    out[np.searchsorted(axis, previous)] = old_values
    if len(new):
        out[np.searchsorted(axis, new)] = field(new)  # computed: only the new points
    return out

CALLS.clear()
field_split(np.linspace(0.0, 1.0, 399), coarse)
assert CALLS == [199]                                 # 199 new, 200 reused
```

The price is in the signature: you have to still have `previous`, bit for bit.
Regenerating it with the identical call (`np.linspace(0.0, 1.0, 200)`) is
enough — but you do have to know what it was, which means tracking the axis
history yourself.

## What it's worth

Measured by `benchmarks/bench_grid_refinement.py`: 800 points, 200 000 modes,
a 1.18 s cold run.

| Workflow | Strategy | Time | Recomputed |
|---|---|---|---|
| Revisit a previous resolution | whole axis | **0.00 s** | nothing |
| Refine `linspace(n)` → `linspace(n+40)` | whole axis | 1.19 s | 840 points |
| Refine `linspace(n)` → `linspace(n+40)` | index chunks | 1.16 s | 840 points |
| Extend `arange(0,1,dx)` → `arange(0,2,dx)` | index chunks | **1.10 s** | 800 of 1600 |
| Refine `linspace(n)` → `linspace(2n-1)` | whole axis | 2.27 s | 1599 points |
| Refine `linspace(n)` → `linspace(2n-1)` | index chunks | 2.15 s | 1599 points |
| Refine `linspace(n)` → `linspace(2n-1)` | old + new split | **1.03 s** | 799 of 1599 |

Read the last three rows together: with 800 of 800 old points present in the
refined axis, whole-axis caching and index chunking both reuse *nothing*, and
only the explicit split gets the halving. Containment is necessary but not
sufficient — you also have to ask for the old set the same way twice.

## When not to bother

- **Your refinement changes every coordinate** (`linspace` with an arbitrary
  new `n`). Two points survive. There is nothing to reuse, and adding chunking
  machinery buys you complexity and no time.
- **A physical parameter changed**, not the grid. A different Poisson ratio
  means every value is genuinely different; recomputation is correct.
- **Your grid is small enough that the computation is fast.** Caching earns its
  keep on the expensive axis, not the cheap one.

## Cleaning up

```python
import shutil
shutil.rmtree("./.cash-grid-demo", ignore_errors=True)
```
