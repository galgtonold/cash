# Mutation harness

Break a subsystem on purpose and see whether the suite notices. A green suite
proves nothing by itself; it means something once you know what turns it red.

The integration suite drives a **real Jupyter kernel in another process**, so a
monkeypatch applied from a pytest plugin never reaches the code under test.
This harness gets in via `sitecustomize`, which Python imports automatically at
interpreter startup, and which the kernel inherits because `KernelManager`
passes the parent environment through. Nothing is added to `src/` — test hooks
do not belong in production code, and the only `CASH_*` variable cash itself
reads is a real configuration knob.

## Running one

```bash
CASH_MUTATION=upstream-dead \
CASH_MUTATION_MARKER=/tmp/mut/m.json \
PYTHONPATH=tests/mutations \
pytest tests/test_notebook_integration -q -n 8
```

Then read the markers — **always**:

```bash
python -c "import json,glob; rows=[json.load(open(f)) for f in glob.glob('/tmp/mut/m.*.json')]; print(f\"applied={sum(r['applied'] for r in rows)}/{len(rows)} calls={sum(r['calls'] for r in rows)}\")"
```

## Reading the result

Three outcomes, and only the marker tells them apart:

| Failures | Calls | Meaning |
|---|---|---|
| some | > 0 | The suite covers this mechanism. |
| **0** | **> 0** | **The suite does not.** The code ran, broken, and nothing objected. |
| any | **0** | **Void.** The mutation never executed; the run measured nothing. |

The last row is why the call count exists. "Nothing failed" reads identically
whether the suite is tolerant or the patch never landed, and those mean
opposite things. This is not hypothetical — an earlier attempt patched
`_backward_scan_pass` from a pytest plugin, watched 57 upstream integration
tests pass, and nearly reported a coverage hole that did not exist. The patch
was in the pytest process; the code was in the kernel.

A `calls=0` result also catches the subtler failure: a mutation whose target
was renamed silently patches a *new* attribute nobody calls. `_catalogue.py`
declares what each mutation `replaces`, and
`tests/test_notebook_integration/test_mutation_harness.py` asserts those
attributes exist — because that mistake already produced one void run here.

## Measured, 2026-08-24

Against the 56 upstream / invalidation / file-dependency integration files:

| Mutation | Result | Evidence |
|---|---|---|
| `upstream-dead` | 47 failed | 98 calls — covered |
| `file-deps-blind` | 55 failed | 379 calls — covered |
| `restore-dead` | **0 failed** | 257 calls — **not covered** |

`restore-dead` disables virtual restore, so every upstream value re-executes
instead of coming back from cache. Correctness is untouched, which is exactly
why nothing fails: the answers stay right and only the speed is gone. That is
the same silent-degradation shape as the Windows write bug — cash keeps working
and quietly stops paying for itself. See the tracker.

## Adding a mutation

Add a `Mutation` to `_catalogue.py`. Break the mechanism **bluntly** — a
mutation that only breaks an edge case tells you nothing about the mainline.
Route every call through `record()`, list what it `replaces`, and add its name
to the parametrised guard in the harness test.
