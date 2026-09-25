# Caching over a grid

!!! info "Applies to: decorator"
    Code that runs an expensive cached computation over a grid and tunes the
    grid's resolution.

When you tune a grid, you sometimes go back to a resolution you already ran and
sometimes make it finer. Cash handles the two very differently, and one
question decides which: **does the new axis still contain the points you already
computed?**

## Going back to a resolution is free

```python
import numpy as np
from cash import Cash

app = Cash()
CALLS = []

@app.cache(assume_safe=True)
def field(axis):
    CALLS.append(len(axis))
    return np.sin(np.arange(1, 400)[:, None] * axis[None, :]).sum(axis=0)

field(np.linspace(0.0, 1.0, 200))     # first call: computes
field(np.linspace(0.0, 1.0, 100))     # a new grid: computes
CALLS.clear()
field(np.linspace(0.0, 1.0, 200))     # back to 200: cache hit
assert CALLS == []
```

<!-- claim: cash/object_hashing.py:hash_numpy @f6df9c37 -->
`np.linspace(0.0, 1.0, 200)` builds the same array bit for bit every time, so
the third call is a plain hit. Sweeping a resolution down until accuracy breaks
and stepping back costs nothing.

## Refining usually reuses nothing

```python
CALLS.clear()
field(np.linspace(0.0, 1.0, 240))     # refined from 200: recompute
assert CALLS == [240]

coarse = np.linspace(0.0, 1.0, 200)
assert len(np.intersect1d(coarse, np.linspace(0.0, 1.0, 240))) == 2
```

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.hash_payload @c4f48efb -->
The axis is one argument, and 240 points from `linspace` are 240 new
coordinates: only the two endpoints survive. There is no earlier work to reuse.

Reuse is possible only when the new axis **contains the old points bit for
bit**. Two constructions do:

```python
# Extending the domain at a fixed step: the old axis is a prefix.
dx = 1.0 / 200
assert np.arange(0.0, 1.0, dx).tobytes() == np.arange(0.0, 2.0, dx)[:200].tobytes()

# Doubling a linspace: every old point survives, between new ones.
assert len(np.intersect1d(np.linspace(0.0, 1.0, 200),
                          np.linspace(0.0, 1.0, 399))) == 200
```

If your refinement has neither property, change how you build the axis; no
caching recipe helps.

## Recipe 1: cache blocks, when the old axis is a prefix

Evaluate the axis in fixed blocks and cache each one. Extending the domain then
computes only the new blocks:

```python
CHUNK = 100

@app.cache(assume_safe=True)
def field_chunk(block):
    CALLS.append(len(block))
    return np.sin(np.arange(1, 400)[:, None] * block[None, :]).sum(axis=0)

axis = np.arange(0.0, 1.0, dx)                  # 200 points = 2 blocks
field_chunk(axis[0:CHUNK])                      # first call: computes
field_chunk(axis[CHUNK:2 * CHUNK])              # a new block: computes

wider = np.arange(0.0, 2.0, dx)                 # 400 points = 4 blocks
CALLS.clear()
field_chunk(wider[0:CHUNK])                     # cache hit: same block
field_chunk(wider[CHUNK:2 * CHUNK])             # cache hit
field_chunk(wider[2 * CHUNK:3 * CHUNK])         # new territory: computes
field_chunk(wider[3 * CHUNK:4 * CHUNK])         # computes
assert CALLS == [CHUNK, CHUNK]
```

In real code, loop over the blocks and `np.concatenate` the results. Make a
block big enough that numpy does real work (10⁴ to 10⁶ elements). This recipe
does nothing for a doubled `linspace`: the old points sit at even indices, so
no block matches an old one.

## Recipe 2: split into old and new points

This works whenever the new axis contains the old one, however the points are
interleaved:

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

The price: you must still have `previous`, bit for bit. Rebuilding it with the
same call is enough, but you have to remember what it was.

## What it's worth

One measurement: 800 points, 200,000 modes, a 1.18 s cold run.

| Workflow | Strategy | Time | Recomputed |
|---|---|---|---|
| Revisit a previous resolution | whole axis | **0.00 s** | nothing |
| `arange(0,1,dx)` → `arange(0,2,dx)` | blocks | **1.10 s** | 800 of 1600 |
| `linspace(n)` → `linspace(2n-1)` | whole axis | 2.27 s | 1599 points |
| `linspace(n)` → `linspace(2n-1)` | blocks | 2.15 s | 1599 points |
| `linspace(n)` → `linspace(2n-1)` | old + new split | **1.03 s** | 799 of 1599 |

Don't bother when every coordinate changes, when a physical parameter changed
rather than the grid, or when the computation is fast anyway.
