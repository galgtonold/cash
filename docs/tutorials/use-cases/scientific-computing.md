# Scientific Computing

Scientific computing means simulations that take hours, parameter sweeps with hundreds of variations, and the need to reproduce results months later. Cash caches each simulation run, so re-running an analysis notebook doesn't trigger a fresh hour-long simulation.

## Why this matters in scientific computing

- **Compute cost.** Hour-long simulations don't tolerate accidental re-runs. The point of caching is that "I closed the notebook and re-opened it" never costs you an afternoon.
- **Parameter sweeps.** Each `(param_a, param_b, seed)` tuple is cached independently. Re-running the sweep skips combinations you've already computed and only fills in the gaps.
- **Reproducibility.** Six months later, when a reviewer asks how you produced Figure 3, the cached result is *exactly* what you used. Same args, same key, same bytes.
- **Iteration on analysis.** You usually iterate on the plot, the statistical test, or the summary table — not the underlying simulation. Cash makes the simulation a one-time cost and lets the analysis layer churn.

## Quick start

```python
import cash
import numpy as np

@cash.cache
def simulate(n_steps: int, dt: float, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    state = np.zeros((n_steps, 3))
    for i in range(1, n_steps):
        state[i] = state[i-1] + alpha * dt * rng.standard_normal(3)
    return state

trajectory = simulate(n_steps=10_000, dt=0.01, alpha=0.5, seed=42)
```

First call runs the simulation. Every subsequent call with the same four arguments returns the cached `state` array. Change `alpha` or `seed` and you get a new key and a fresh run.

## Parameter sweeps

A sweep is just a loop over arguments. Each combination is an independent cache entry:

```python
results = {}
for alpha in [0.1, 0.5, 1.0, 2.0]:
    for seed in range(10):
        results[(alpha, seed)] = simulate(10_000, 0.01, alpha, seed)
```

First run: 40 simulations execute and get cached. Re-running the same loop: 40 cache hits, no computation. Add `alpha=4.0` to the list: 10 new runs, the existing 40 still hit. This is the workflow that justifies the cache directory's existence.

For an embarrassingly parallel sweep you can dispatch the same loop across workers — see the parallel-sweep note in [Caveats](#caveats).

## Determinism: always seed the RNG

The hard rule for cacheable simulations: **the seed is an argument, not a global**.

```python
# Good — seed is an argument, result is reproducible
@cash.cache
def simulate(n_steps: int, dt: float, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_steps)

# Bad — np.random.seed() inside the function bakes the *first* observed state
@cash.cache
def simulate_bad(n_steps: int):
    np.random.seed(42)        # don't do this inside a cached function
    return np.random.randn(n_steps)
```

Cash scans your notebook for unseeded RNG calls and warns when it finds them — the warning is the cache telling you that what's saved won't match what a fresh re-run would produce. See [Controlling Cache Behavior](../feature-guides/controlling-cache-behavior.md) for the `# @cash:allow-random` escape hatch and the full list of detected RNG calls.

If you need an unseeded run for a one-off exploration, do it outside the cached function. Inside the cache, always pass the seed.

## Large arrays and persistence

Simulation outputs are usually arrays — and often big ones. Cash's smart-persistence layer decides automatically when an in-memory entry is worth writing to disk, and multi-megabyte arrays cross that threshold quickly. You normally don't have to do anything.

When you *want* to force persistence — for example, a long-running simulation whose output absolutely must survive a kernel restart — annotate the cell:

```python { .nb-cell }
# @cash:persist
trajectory = simulate(n_steps=1_000_000, dt=0.001, alpha=0.5, seed=42)
```

See [Smart Persistence](../feature-guides/smart-persistence.md) for the heuristics Cash uses and the difference between L1 (in-memory) and L2 (on-disk) tiers.

## Custom numerical types

Cash hashes `numpy` arrays, plain numbers, lists, tuples, and built-ins by default. For specialised numerical objects — `mpmath` arbitrary-precision numbers, JAX arrays, PyTorch tensors on GPU, sparse matrices — register a custom hasher so the cache key reflects the array's contents instead of its Python `id`.

```python
import cash

def hash_my_array(arr):
    return arr.tobytes()  # or arr.numpy().tobytes(), etc.

cash.register_hasher(MyArrayType, hash_my_array)
```

See [Custom Hashers](../feature-guides/custom-hashers.md) for the full hook surface and worked examples for the common scientific types.

## File-based input data

Initial conditions, mesh files, observation data — anything you read with `pandas.read_*` is auto-tracked. The source file's mtime folds into the cache key, so editing `initial_conditions.csv` invalidates every cached call downstream of it.

For non-pandas formats (HDF5 via `h5py`, NetCDF, Zarr, custom binary), declare the dependency explicitly with `file_depends_on=` so Cash sees the file. See [Custom File Sources](../feature-guides/custom-file-sources.md).

## Pipeline pattern: simulate → analyse → reduce → plot

The high-leverage layout for a research notebook is one cached function per stage:

```python
# test:inject: import cash
@cash.cache
def simulate(params): ...

@cash.cache
def analyse(trajectory): ...

@cash.cache
def reduce(analyses): ...

# @cash:no-cache
def plot(summary): ...    # plotting is cheap, side-effect-y, skip the cache
```

Edit the plot: nothing else re-runs. Edit `reduce`: `simulate` and `analyse` are cache hits, only `reduce` and `plot` execute. Edit the simulation: everything cascades, which is the correct behaviour because the underlying physics changed.

Plotting itself rarely benefits from caching — the figure is a side effect, not a return value, and rendering is fast compared to simulation. Mark the cell with `# @cash:no-cache` to make the intent explicit.

## Reproducibility over months

The cache directory *is* your reproducibility artefact. Two practical habits:

- **Back up the cache directory.** It contains the inputs and outputs of every run you've made. Treat it like any other research output — include it in backups, or commit a small one to the repo.
- **Pin dependencies.** Cash's keys reflect *your* source code; they don't reflect a library upgrade that changed numpy's RNG implementation. Pin `numpy`, `scipy`, and any solver libraries in a lockfile, and you've closed the most common reproducibility gap.

If you genuinely need library versions to fold into the cache key — e.g. you're running across a numpy major-version bump and want both eras cached separately — wrap "library version" in a `DataSource` subclass and pass a resolver via `dynamic_depends_on=`. Note that `dynamic_depends_on` requires a `DataSource` instance, not a raw string. See [Dynamic Dependencies](../feature-guides/dynamic-dependencies.md) for the subclass shape.

## Common patterns

- **Iterative solvers with checkpoints.** Cache each checkpoint as a function of `(initial_state, n_steps, params)`. A solver that crashes at step 10,000 can resume from the last cached checkpoint instead of from step 0.
- **Monte Carlo with multiple seeds.** Wrap the per-seed run in `@cash.cache`, loop over seeds. Adding more seeds is incremental — the existing ones stay cached.
- **Sensitivity analysis.** Identical to a parameter sweep — vary one input at a time, each combination cached independently, the analysis layer iterates freely.
- **Embarrassingly parallel sweeps.** Dispatch the cached function across processes with `multiprocessing` or `joblib`. If you want a *guarantee* that two workers don't both compute the same `(alpha, seed)` combination, you need `Cash(use_locking=True)` against `RedisBackend`. Every backend single-flights concurrent callers *within* one process, but a `multiprocessing`/`joblib` sweep puts the workers in **separate processes**, and Redis is the only shipped backend whose lock spans them. See [Thread Safety](../feature-guides/thread-safety.md) for the backend table and the redundancy semantics.

## Caveats

- **Don't pass 1 GB arrays as cache-key arguments.** The key hashes every argument. If your "input" is a huge precomputed array, pass a fingerprint instead — a hash, a file path, or the parameter tuple that *produced* the array — and resolve the array inside the cached function.
- **GPU non-determinism.** CUDA kernels are non-deterministic by default; the same input can produce slightly different outputs across runs. Set `torch.use_deterministic_algorithms(True)` (and the corresponding env vars) before you cache GPU computations, or accept that cache hits may diverge from fresh re-runs at the level of floating-point noise.
- **Floating-point reproducibility across hardware.** Results depend on your BLAS, your CPU, and your compiler flags. A cache built on one machine and read on another may not bit-match if the architectures differ. For most analyses this is below the noise floor; for high-precision work, document the hardware alongside the cache.
- **Don't cache the file write.** `np.save`, `h5py.File(...).create_dataset(...)`, `xarray.to_netcdf` — these are side effects. Cache the computation that produces the array; leave the write outside the cached function.

## Related

- [Custom Hashers](../feature-guides/custom-hashers.md) — for `mpmath`, JAX, PyTorch, sparse matrices, and other specialised numerical types.
- [Smart Persistence](../feature-guides/smart-persistence.md) — when Cash decides to push a large array from memory to disk.
- [Thread Safety](../feature-guides/thread-safety.md) — `use_locking=True` semantics for parallel sweeps; every backend locks in-process, Redis is the one that locks across processes.
- [Controlling Cache Behavior](../feature-guides/controlling-cache-behavior.md) — RNG-detection warnings, `@cash:no-cache`, `@cash:persist`, and TTL.
- [Custom File Sources](../feature-guides/custom-file-sources.md) — declaring HDF5, NetCDF, and other non-pandas readers as dependencies.
- [Dynamic Dependencies](../feature-guides/dynamic-dependencies.md) — folding a library or schema version into the cache key.
