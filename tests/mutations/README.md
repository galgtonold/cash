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

| Failures | Calls | Changes behaviour? | Meaning |
|---|---|---|---|
| some | > 0 | — | The suite covers this mechanism. |
| **0** | **> 0** | **yes** | **The suite does not.** A real gap. |
| 0 | > 0 | **no** | **Nothing.** The mutation is inert — see below. |
| any | **0** | — | **Void.** It never executed; the run measured nothing. |

### The third row is the one that will fool you

`calls > 0` proves the mutated code *ran*. It does **not** prove the mutation
*mattered*. A function can be called, return something useless, and have its
failure absorbed by a fallback — in which case zero failures means the code is
redundant, not that the suite is blind.

That is exactly what happened with `restore-dead` on 2026-08-24. It reported 0
failures across 362 tests with 257 confirmed calls, which reads like a glaring
coverage hole. It was filed as one. But putting the badge side by side with and
without the mutation showed **identical output** — `^CACHED: mid = ... (saved
0.11s)` either way. Killing `_try_virtual_restore` changes nothing a user or a
test can see, because the scheduler compensates.

So before reading a zero as a gap, **produce a positive control**: one concrete
scenario where the mutation visibly changes what cash does. If you cannot, the
mutation is inert and the run is as uninformative as `calls=0`.

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
| `restore-dead` | 0 failed | 257 calls — **inert, see above** |

And against the 29 files that explicitly assert `CACHED` / `RESTORED`:

| Mutation | Result | Evidence |
|---|---|---|
| `statement-cache-dead` | 21 failed | 2456 calls — covered |
| `restore-dead` | 0 failed | 65 calls — **inert, see above** |

Those two are the reason to keep mutations narrow. Run alone, `restore-dead`
reads as "the suite never asserts caching", which is false — it asserts it hard
for the cell you ran. One mutation cannot tell a narrow hole from a broad one.

`restore-dead` looked like the headline finding and was not one. Kept in the
catalogue as the worked example of an inert mutation, since recognising that
shape is most of the skill in reading these results.

## Adding a mutation

Add a `Mutation` to `_catalogue.py`. Break the mechanism **bluntly** — a
mutation that only breaks an edge case tells you nothing about the mainline.
Route every call through `record()`, list what it `replaces`, and add its name
to the parametrised guard in the harness test.
