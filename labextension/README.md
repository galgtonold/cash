# cash-live-cells — the JupyterLab half of the live-cell read

cash reads the cells it did **not** execute from the saved `.ipynb`. An unsaved
edit is therefore invisible to it, and a downstream cell can silently restore a
value computed from source that no longer exists. Colab exposes live cells
natively and VS Code leaves a hot-exit backup on disk; on JupyterLab the only
route is a frontend component, because the kernel holds no copy of the document.

This extension pushes the notebook's current cell sources over a comm named
`cash_live_cells`. The kernel side is `src/cash/notebook/live_cells.py`, and
`src/cash/notebook/server_discovery.py` reads that store before it falls back to
the file.

It **pushes** rather than answering a request: a comm sent while a cell is
executing is queued until that cell ends (pinned by
`tests/test_notebook_integration/test_comm_reply_during_execution.py`), so a
request/response design could never serve a read at the moment cash needs one.
It pushes on change (debounced 150 ms) and flushes synchronously on
`NotebookActions.executionScheduled`.

---

## The one line you must not remove

```ts
comm.commsOverSubshells = 'disabled';
```

JupyterLab 4.6 defaults `commsOverSubshells: perCommTarget`, which delivers
comms on a **subshell thread**. Under that default the whole design collapses in
two ways at once:

1. **Ordering stops being a guarantee.** The flush-before-execute trick works
   only because shell messages are FIFO. On a subshell the push and the
   `execute_request` race. A spike measured a 0.4–7.4 ms lead that was won 130
   times out of 130 — exactly the kind of evidence that reads as a guarantee
   until a slower machine proves otherwise. Forcing the comm onto the main shell
   restored true FIFO, verified 56/56.
2. **The kernel-side store becomes cross-thread mutable state.** `live_cells.py`
   keeps a plain dict with no lock, which is only sound because `handle_message`
   runs on the kernel's own thread.

Nothing on the Python side can enforce this — over there it is a comment. Three
things make its removal fail loudly instead:

| Guard | Needs Node? | Catches |
| --- | --- | --- |
| `labextension/scripts/check-comms-over-subshells.js` (run by `npm run build`, before *and* after the bundle is produced) | yes | removal in the source, and a bundle built without it |
| `tests/test_notebook/test_labextension_packaging.py` | no | the same two, from the ordinary `pytest` run |
| the comment block around the line in `src/index.ts` | — | a reader about to delete it |

If you genuinely intend to change this, change all of them together, plus the
thread-safety note in `live_cells.py`.

---

## The first comm_open is always refused — that is normal

On a fresh kernel the extension's first `comm_open` **cannot** succeed, and the
flush-before-execute ordering is exactly why: the push is emitted before the
`execute_request`, so it necessarily arrives before the cell that runs
`import cash` registers the target. ipykernel answers `comm_close`
("No such comm target registered") and JupyterLab disposes the handler.

So the comm is treated as disposable, not as a handle to keep:

- `onClose` drops it from the per-panel `WeakMap`, so the next push builds a new
  one — by which time the import has run. An identity check stops a late close
  for an already-replaced comm from evicting its successor.
- the `catch` drops it too, because a disposed handler raises `Cannot send` and
  latching on to it would reproduce the same muteness.
- **only** `executionScheduled` may *open* a comm (`send(panel, true)`). The
  debounced keystroke path passes `false`. Opening from there as well would
  still work, but it would spray that refusal through the kernel log of every
  user who never imports cash — and buy nothing, since the debounced push is a
  latency optimisation while the flush is the correctness path.
- a kernel restart only *deletes* the comm. Rebuilding it immediately would hit
  a kernel that has likewise lost its `import cash`.

The matching kernel-side rule is in `live_cells.py`: **`_on_open` calls
`reset()`**. A `comm_open` is a new frontend connection superseding the old one,
and the extension's `seq` counter is a closure variable in the plugin
activation — it restarts at 0 on every browser page load, while the kernel store
lives as long as the kernel. Without the reset, every push after an F5 is dropped
as "older" and cash keeps serving the pre-reload snapshot: source text that, since
a reload discards unsaved edits, no longer exists anywhere. Clearing is the safe
direction, because an empty store makes cash fall through to the saved `.ipynb`
rather than believe the notebook is empty.

---

## Building

Node is **not** required to develop cash, to run `pytest`, or to build the
wheel. The built bundle is committed under `src/cash/labextension/` precisely so
that none of those ever need a JavaScript toolchain. You only need Node when you
change `src/index.ts`.

```bash
cd labextension
npm install                     # @jupyterlab/builder + typescript
npm run build                   # guard -> tsc -> jupyter labextension build -> guard
cd ..
git add src/cash/labextension   # the rebuilt bundle is part of the commit
```

`npm run build:labextension` shells out to `jupyter labextension build`, so the
`jupyterlab` **Python** package must be importable in the active environment:

```bash
pip install "jupyterlab>=4,<5"
```

`node_modules/`, `lib/` (the intermediate `tsc` output) and `package-lock.json`
are ignored; only `src/cash/labextension/` is committed.

## Shipping

`pyproject.toml` maps the built output into the wheel as
[shared data](https://hatch.pypa.io/latest/config/build/#shared-data):

```
src/cash/labextension        -> share/jupyter/labextensions/cash-live-cells
labextension/install.json    -> share/jupyter/labextensions/cash-live-cells/install.json
```

which is where JupyterLab discovers prebuilt extensions, so
`pip install cash-lib` is the entire installation step — no `jupyter labextension
install`, no rebuild of JupyterLab.

Verify with:

```bash
jupyter labextension list        # expect: cash-live-cells vX.Y.Z enabled ok
```

An **editable** install (`pip install -e .`) carries the same shared data, so a
contributor's environment gets the extension too — as a copy made at install
time, refreshed by reinstalling.

For a *symlinked* dev install that picks up a rebuild without reinstalling, use
`cash._jupyter_labextension_paths()` via:

```bash
cd src && jupyter labextension develop --overwrite cash
```

Note the invocation. The usual `jupyter labextension develop --overwrite .` from
the repo root does **not** work here: it resolves the module by importing the
distribution name (`cash_lib`) and then each top-level directory, and cash's
importable package is `cash` under `src/`, so neither matches. Passing the
package directory directly is what finds the hook. On Windows this step needs
Developer Mode enabled, or it fails at the symlink; the editable-install copy
above is the fallback.

## Versioning

This package versions **independently** of `cash-lib`; it is bumped when the
extension itself changes, not on every cash release. Do not wire it to
`cash.__version__` — the release process treats the `__version__` line in
`src/cash/__init__.py` as the single source of truth for the *Python* package,
and adding a second literal that must be edited in lockstep would just be a
second thing to forget.
