# Wheel-gate harness (CAS-190)

An automated, assertion-driven reproduction of the manual "gate rounds" — a
fresh **wheel venv** + a **real Jupyter server** + real **kernel restarts** —
that a developer or CI can run with one command.

## Why this exists

The fast suite (`tests/test_notebook_integration/`) is structurally **blind** to
two whole classes of bug, proven four times this release (CAS-185, CAS-196,
CAS-202, the packaging P0):

1. **Kernel-restart behaviour.** `nb_runner` boots a fresh kernel per test and
   has **no restart method** (`grep 'def restart' conftest.py` → nothing). Every
   restart-path bug is invisible: CAS-196's restart re-fire and CAS-202's
   restart-retrain both shipped green.
2. **Wheel-venv install layout.** The suite runs the **editable dev install**
   against `C:\Python314`; testers run a **fresh wheel venv**. `importlib.metadata`
   phantom file-dep probes (81 in a venv vs 0 in dev) only exist in the venv.
   Every install-layout bug is invisible. The dev env doesn't even have
   `jupyter_server` installed — so whatever the CAS-171 36-config sweep drove, it
   was **not a real Jupyter server**. A harness that reports green on a bug that
   reproduces on the first real kernel is itself the defect (that is CAS-190).

The manual gate catches these but needs five human-like agents ~30 min. This
harness encodes that methodology as four assertion-driven scenarios, each proven
by an **external signal** — a counter written from *inside* a function/cell to a
file and read from *outside* the kernel. Never a badge or a `print`; those are
restored on a cache hit and so cannot witness a silent re-run.

## What it does

1. Builds a wheel from the current tree (`python -m build --wheel`) into its own
   `C:\Temp\wheelgate\dist` — **not** the repo `dist/`, so it can't disturb
   release hygiene — or accepts `--wheel <path>`.
2. Creates a **fresh venv on a short path** (`C:\Temp\wheelgate\venv`; deep repo
   paths blow MAX_PATH 260 and yield bogus `ModuleNotFoundError`) and installs
   `<wheel>[all]` + `pandas numpy scikit-learn jupyter-server jupyter-client
   nbformat ipykernel`. **Never** installs into `C:\Python314`.
3. Registers a **unique `wheelgate` kernelspec into the venv**
   (`ipykernel install --sys-prefix`, never `--user`) so the kernel can only
   resolve to the venv interpreter — and a **guard cell asserts `sys.prefix` is
   the venv**, so the install-layout is genuinely exercised.
4. Drives a **real `jupyter server` + `BlockingKernelClient`** via
   `wheel_gate_driver.py` (a parametrized copy of the proven
   `C:/Temp/cashut/driver_reference.py`), with a real kernel `restart` between
   run phases. `PYTHONUTF8=1` in the server env so cash's emoji badges can't
   crash the harness on cp1252 (CAS-192).
5. Cleans up: `quit` + tree-kill the server (idempotent, rerun-safe; leftover
   sockets are only TIME_WAIT, never a live process).

## The scenarios

| id | invariant (external signal) | issue | baseline |
|----|-----------------------------|-------|----------|
| **S1** | multi-cell `make_classification → DataFrame → train_test_split → @cash.cache train()` restores after a **kernel restart** (fit counter unchanged) | CAS-202 | **GREEN** (was RED until CAS-202 fixed) |
| **S2** | a downstream reader does **not** re-fire an upstream `df.to_csv('audit.log', mode='a')` — `audit.log` byte-stable vs a `%cash_off` baseline | CAS-196 | **RED** |
| **S3** | the **single-cell** version of the same sklearn `@cash.cache` work survives a restart (fit counter unchanged) | CAS-202 control | **GREEN** |
| **S4** | a plain `@cash.cache` int fn survives a restart (call counter unchanged) | control | **GREEN** |

`RED` = the invariant is violated = the bug is present. S1 was **RED** until
CAS-202 was fixed: the decorator arg-hash keyed a DataFrame argument on its
per-session `_cash_lineage_hash` instead of its stable content, so the persisted
entry was never found after a restart and the model re-trained. Now the
**S1 (GREEN) vs the still-**RED** S2** keeps the harness non-vacuous; **S4
(GREEN)** shows restart-survival works when no sklearn import poisons the
file-dep set. A harness that passed everything would be the exact CAS-190
blindness it is meant to cure — so **S2 staying RED is the proof of
non-vacuity.**

## How to run

```bash
# full run: build the wheel, provision a fresh venv, all four scenarios
python scripts/wheel_gate.py

# accept a prebuilt wheel (skips the ~1-2 min build)
python scripts/wheel_gate.py --wheel dist/cash_lib-0.5.0b1-py3-none-any.whl

# fast iteration: reuse an already-provisioned venv
python scripts/wheel_gate.py --reuse-venv --wheel dist/cash_lib-0.5.0b1-py3-none-any.whl

# a subset
python scripts/wheel_gate.py --scenarios S1,S2
```

**Exit code 0** iff the observed RED/GREEN matrix matches the recorded baseline
(S2 RED, S1/S3/S4 GREEN — S1 flipped to GREEN when CAS-202 was fixed). A
mismatch exits 1 — either a green invariant regressed, or a known-open bug got
fixed and the baseline needs updating.

### From pytest / CI

Kept out of default collection (it is slow). A shim under `tests/test_wheel_gate/`
runs it only when opted in:

```powershell
$env:CASH_WHEEL_GATE = "1"
pytest -m wheel_gate tests/test_wheel_gate -n0
# optional: forward args to the harness
$env:CASH_WHEEL_GATE_ARGS = "--reuse-venv --wheel dist/cash_lib-0.5.0b1-py3-none-any.whl"
```

The default fast suite collects the shim and **skips it in ~0.02 s**.

## Timing

| step | cold | with `--reuse-venv` |
|------|------|---------------------|
| wheel build | ~1–2 min | n/a (`--wheel`) |
| venv create + `pip install [all]` + sklearn/jupyter | ~2–3 min | skipped |
| four scenarios (each: real server boot + restart) | ~2–3 min | ~2–3 min |
| **total** | **~6–8 min** | **~2–3 min** |

## Proof output (against `57a9823`, wheel `cash_lib-0.5.0b1`)

```
id  status  expected  match  title
------------------------------------------------------------------------------
S1  RED     RED       yes    restart survival of @cash.cache sklearn pipeline (CAS-202)
S2  RED     RED       yes    to_csv audit-log not re-fired during reconstruction (CAS-196)
S3  GREEN   GREEN     yes    single-cell sklearn @cash.cache survives a restart (CAS-202 control)
S4  GREEN   GREEN     yes    plain @cash.cache int fn survives a restart (control)

[S1] @cash.cache re-trained after restart: fit body ran 2x (cold 1 + retrain 1)
      evidence={'cold_steps': [0, 0, 0, 1], 'fit_calls_cold': 1, 'fit_calls_after_restart': 2}
[S2] cash re-fired the non-idempotent to_csv append: no-cash=1 line vs cash run-all=2, after-reader=3 lines
      evidence={'baseline_off_lines': 1, 'on_runall_lines': 2, 'on_after_reader_lines': 3, 'loud_refusal': False}
[S3] single-cell model restored after restart (fit ran 1x total)
[S4] restored after restart (body ran 1x total)

RED (bug reproduced): ['S1', 'S2']
GREEN (invariant held): ['S3', 'S4']
```

S2's `no-cash=1 vs cash=2` exactly reproduces CAS-196's measured signature (and
goes to 3 after the downstream reader). S1's fit counter going 1→2 across the
restart is CAS-202. Both are invisible to the fast suite.

## Files

- `scripts/wheel_gate.py` — the orchestrator (build → venv → scenarios → matrix).
- `scripts/wheel_gate_driver.py` — the persistent-kernel driver (real Jupyter
  server + `BlockingKernelClient`; env-parametrized copy of the gate's
  `driver_reference.py`).
- `tests/test_wheel_gate/` — opt-in pytest shim + local `wheel_gate` marker.

Only `src/` is off-limits — this is the harness, not the fix. The CAS-202 /
CAS-196 fixes land in later tasks; this harness is what will prove them.
```
