# Scientific computing

!!! info "Applies to: decorator"
    Researchers running simulations and parameter sweeps from Python scripts or
    modules.

Simulations take hours, sweeps have hundreds of variations, and a reviewer may
ask for Figure 3 again in six months. Cache each simulation run, and re-running
the analysis never repeats a simulation you already have.

## A cached simulation

```python
import cash
import numpy as np

@cash.cache
def simulate(n_steps: int, dt: float, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    state = np.zeros((n_steps, 3))
    for i in range(1, n_steps):
        state[i] = state[i - 1] + alpha * dt * rng.standard_normal(3)
    return state

trajectory = simulate(n_steps=10_000, dt=0.01, alpha=0.5, seed=42)
```

<!-- claim: cash/backends/persistence_policy.py:PersistencePolicy.decide @2270c6c5 -->
The first call runs the simulation and writes the result to disk; every later
call with the same four arguments, in this run or the next, loads it. Change
`alpha` or `seed` and it runs again. Large results are stored like small ones:
a decorated result is always written to disk unless it exceeds the disk tier's
size cap; see
[Where results are stored](../../decorator.md#where-results-are-stored).

<!-- claim: cash/source_norm.py:normalize_source_for_hash @a1b4e588 -->
Editing a comment or the docstring of `simulate` re-runs nothing. Editing its
code, or a helper it calls, re-runs every combination the next time you ask
for it.

## Parameter sweeps

A sweep is a loop over arguments, and each combination is its own entry:

```python
results = {}
for alpha in [0.1, 0.5, 1.0, 2.0]:
    for seed in range(3):
        results[(alpha, seed)] = simulate(1_000, 0.01, alpha, seed)
```

The first run computes 12 simulations. Running the loop again computes none.
Add `alpha=4.0` and only its 3 runs are new. The same holds for Monte Carlo
seeds and one-at-a-time sensitivity runs.

To spread a sweep over worker processes (`multiprocessing`, joblib), see
[Threads and processes](../feature-guides/thread-safety.md#across-processes-pool-processpoolexecutor-joblib).
Workers share results through the disk. Two workers can still compute the same
combination at the same moment, unless the backend is Redis with
`use_locking=True`.

## Disk space

<!-- claim: cash/backends/adaptive_caps.py:adaptive_disk_cap @0d13d1d2, cash/__main__.py:cmd_clear @a08b9044 -->
A sweep writes one entry per combination, and every entry goes to disk. By
default cash lets the cache use a quarter of the free room on its volume,
between 8 GiB and 100 GiB. When it is full, cash evicts the entries that are
cheapest to recompute per byte.

- **See it.** `cash info` in a terminal shows the folder, what it holds and
  the cap:

    ```text title="Output"
      Cache dir:  /home/me/study/.cash
      Holds:      120 entries, 3.2 GiB
      Max size:   auto -- disk 26.0 GiB, RAM 3.1 GiB
    ```

- **Pin it.** Set `max_cache_size`, for the whole project in
  `pyproject.toml`:

    ```toml title="pyproject.toml"
    [tool.cash]
    max_cache_size = "50GB"
    ```

- **Drop one study.** `cash clear --function simulate` deletes every entry of
  `simulate`. After you edit `simulate`, its old entries stay on disk until
  the cap evicts them; this frees the space now.

See [Where your cache lives](../../how-it-works/storage.md#when-the-disk-fills-up)
for how eviction picks entries.

<!-- claim: cash/tracking/randomness/detect.py:RNG_CARRIER_CONSTRUCTORS @620106b9 -->
## Seed through an argument

The seed must be an **argument**, and the generator local:

```python
@cash.cache
def sample(n: int, seed: int):
    rng = np.random.default_rng(seed)     # good: seeded from an argument
    return rng.standard_normal(n)
```

An unseeded draw inside a cached function stores the first value and returns
it on every later call, with a warning
([`RANDOM-UNSEEDED`](../../warnings.md#random-unseeded)). Calling the global
`np.random.seed(42)` inside it is worse: a hit skips the reseed, so every draw
after it in your program depends on whether the cache was warm. See
[`allow_random=`](../../decorator.md#allow_random).

## One function per stage

<!-- test:skip reason="illustrative: the stage bodies are elided" -->
```python
@cash.cache
def simulate(params): ...

@cash.cache
def analyse(trajectory): ...

@cash.cache
def reduce(analyses): ...

def plot(summary): ...    # cheap, and a figure is an effect: leave it undecorated
```

Edit `reduce`: `simulate` and `analyse` are hits, and only `reduce` and `plot`
run. Edit the simulation and everything after it recomputes, as it should.

## Inputs and types

- **Data files** read through numpy, pandas or `open()` are tracked by content:
  editing `initial_conditions.csv` invalidates the calls that read it. HDF5
  through `h5py`, NetCDF and other C-level readers are not seen; name those files
  with `file_depends_on=`. See
  [File dependencies](../feature-guides/custom-file-sources.md).
- **numpy arrays** are hashed by their full content. Types cash can't pickle
  (GPU tensors, some C extension types) need a hasher:

    ```python
    import hashlib

    def hash_my_array(arr):
        return hashlib.sha256(arr.tobytes()).hexdigest()

    cash.register_hasher(MyArrayType, hash_my_array)
    ```

    See [Custom hashers](../feature-guides/custom-hashers.md).

- **Huge arguments** are hashed on every call. Mark the producing cached
  function `frozen=True`, or pass a path or the parameters that produced the
  array and load it inside; see
  [`frozen=` and large arguments](../../decorator.md#frozen-and-large-arguments).

## Reproducibility over months

- **Pin your libraries.** Keys cover your code, not numpy's or your solver's. A
  library upgrade that changes results does not invalidate anything.
- **Keep the cache folder** with the project's other outputs if you rely on it
  to reproduce a figure.
- **GPU and hardware.** CUDA kernels are non-deterministic unless you ask
  otherwise (`torch.use_deterministic_algorithms(True)`), and BLAS results can
  differ between machines at the level of rounding. A cache read on another
  machine returns the stored bits, not what that machine would compute.
- **Writes stay outside.** `np.save` and `to_netcdf` inside a cached function
  are skipped on a hit. Cache the computation and write outside it.

## Related

- [Caching over a grid](../feature-guides/caching-over-a-grid.md): reusing work when you refine a grid.
- [Threads and processes](../feature-guides/thread-safety.md)
- [Deploying](../feature-guides/deploying.md): running sweeps on a cluster or in CI.
