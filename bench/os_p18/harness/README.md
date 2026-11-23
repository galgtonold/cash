# Round-34 replay and scoring harness

Replays a project's `trace.json` (format in `../BRIEF.md`) plain and with cash, then scores
correctness (`compare.py`) and speed (`speed.py`). Code under test: cash `02cd17da`.
Self-contained: the harness itself needs only Python 3.13 with `jupyter_client` (the
venv `/home/user/r34/venv-harness` has everything); the files here use nothing else from this machine.

## Quick start

```bash
# once per machine
git -C /home/user/cash fetch origin main && git worktree add /home/user/r34/cash 02cd17da
uv venv /home/user/r34/venv-harness --python 3.13
VIRTUAL_ENV=/home/user/r34/venv-harness uv pip install -e /home/user/r34/cash jupyter_client ipykernel pandas numpy

# per project (PROJECT_DIR has trace.json, and ideally .venv with the project's packages)
bash /mnt/project-files/tester-round-34/harness/run_all.sh /mnt/project-files/tester-round-34/pNN-slug
#   -> pNN-slug/runs/{plain1,cash1,plain2,cash2}.jsonl + .log, compare.{json,md}, speed.{json,md}
```

Call `run_all.sh` through `bash`: `/mnt/project-files` is mounted without exec permission.
`R34_PY=/path/to/python` picks another harness interpreter. Run nothing else heavy meanwhile.

The project's Python is `PROJECT_DIR/.venv/bin/python` when it exists (used in place, not
copied), else the harness interpreter. It must import **`ipykernel`** (for any `run`/`restart`
step) and **`cash`** (installed from `/home/user/r34/cash`, e.g.
`uv pip install -e /home/user/r34/cash ipykernel`). replay.py checks this up front and stops
with a message if not. Commands in `setup`/`shell`/`script` find that venv's `python` first on PATH.

## Individual tools

| command | does |
|---|---|
| `replay.py TRACE PROJECT_DIR {plain,cash} OUT.jsonl` | one replay |
| `compare.py RUNDIR` | correctness oracle over `RUNDIR/{plain1,plain2,cash1,cash2}.jsonl` |
| `speed.py RUNDIR` | P, C, Z1, Z2, top overhead / missed-reuse steps, noise |
| `stats.py RUNDIR` | per-cell stats (served or not and why), time split first / repeat / after restart, cash's work counts, warnings, cache size -> `stats.json`, `stats.md` |
| `run_all.sh PROJECT_DIR [TRACE]` | plain1, cash1, plain2, cash2 into `PROJECT_DIR/runs/`, then both scorers |

compare.py and speed.py also work on partial runs (e.g. only plain1 + cash1), with a note.

## What replay.py does

* `mktemp -d /tmp/r34-XXXX` -> copies PROJECT_DIR to `<tmp>/proj` (skipping `.venv`, `runs`,
  `__pycache__`, `.git`, `.cash*`); empty `<tmp>/cache` is `CASH_CACHE_DIR` for the whole replay
  (kept across restarts and script runs); empty `IPYTHONDIR`; `MPLBACKEND=Agg`,
  `PYTHONHASHSEED=0`; inherited `CASH_*`/`PYTHON*` variables dropped. The temp dir is deleted by
  its exact path at the end (also on errors).
* `setup` commands run once, with `CASH_DISABLE=1` in both modes, not timed.
* Plain mode: `CASH_DISABLE=1` for the kernel and every command (so a `both` project's
  `@cash.cache` helpers do not cache in plain either).
* Cash mode: the kernel runs `import cash` + `%cash_on` at start and after every restart
  (the HTML badge, cash's default; `R34_BADGE=print` switches to the text badge).
  `CASH_VERBOSE=1` everywhere, for cash's per-call lines.
* Steps: `cell` stores the source; `run` executes the stored source; `restart` = new kernel;
  `write_file` writes; `shell`/`script` run `bash -c CMD` in the project copy. One kernel at a time.
* Timeout per step: `R34_STEP_TIMEOUT` (default 1800 s). A cell is interrupted (kernel restarted
  if it does not go idle within 120 s), a command's process group is killed; the step is
  recorded as `TIMEOUT ...` error. A dead kernel (e.g. OOM) is recorded as `KERNEL_DIED` and restarted.

### Record per step (one JSON line, appended when the step finishes)

First line `op: meta` (mode, python, cash version + git commit, setup results, `files0`: digest of
every project file after setup, `problems`). Then `op: start` (kernel start, `i = -1`) and one
line per step `i = 0..`:

| field | meaning |
|---|---|
| `i`, `op`, `cell` / `cmd` / `path` | step index, op, cell id |
| `wall` | seconds (cell: execute request to idle + reply; command: process wall; restart: new kernel ready + cash init) |
| `wall_adj` | `wall` minus the plain call hook's own time (plain mode, steps with decorated calls) |
| `status`, `error` | `ok`/`error`; `Ename: message` or `exit N: last stderr line` |
| `out` | normalized stdout (ANSI, `0x...` addresses, timings like `1.2 s`/`3 ms`, `CPU times:`/`Wall time:` lines, cash badge/notice lines, the temp path removed) |
| `disp` | displayed values: normalized `text/plain` + digest of any image; cash's badge displays are dropped (`cash_disp` counts them) |
| `stderr` | normalized stderr tail (not compared) |
| `values` | `{name: digest}` of every user-namespace name after the step (run/start/restart steps). Names starting with `_` and the harness's own `cash` import are skipped |
| `codes`, `strs` | code identity of functions/modules/classes, short string values (for Z2 only) |
| `files` | files under the project dir whose content changed in this step: `{path: digest or null}` |
| `src` | the cell source that ran |
| `stmts` (plain run) | `texts` (top-level statements as `ast.unparse`d after IPython's transforms) and `times` (perf_counter around each `run_code` of IPython, i.e. each top-level statement, last-expression display included) |
| `cash` (cash run) | cash's `_last_cell_metrics` for the cell: `statements` with `code`, `status` (`COMPUTED`/`RESTORED`/`SKIPPED`/`MIXED`/...), `execution_time`, `saved_time`, `is_upstream`; `upstream_metrics`; `{stale: true}` when cash recorded nothing for the cell (e.g. it raised) |
| `calls` | decorated calls. Plain: `{fn, q, a: argument digest, f: function identity, t, d: depth, id, p: parent id, ok, s: statement index in the cell, pid}`. Cash: `{fn, q, outcome: HIT/MISS/RAISE, key, t, why}` from `CASH_VERBOSE` |

Value digests (`r34digest.py`): DataFrame/Series/Index via `pandas.util.hash_pandas_object`
+ dtypes/columns/shape (+ index); numpy arrays dtype+shape+bytes; polars `hash_rows`/`hash`
+ schema; scalars/strings by repr; list/tuple/dict/set recursively; sklearn estimators by
`get_params()` + every fitted `*_` attribute (`coef_`, `feature_importances_`, `classes_`, ...);
scipy sparse by its arrays; dataclasses by fields; numpy Generators by state; anything else
(functions, figures, user objects) by type name only.

### Where the cash per-statement / per-call data comes from

* **Notebook, per statement:** `CashMagics._last_cell_metrics`, the same dict `%cash_status dict`
  returns as `last_cell`. A `post_run_cell` hook is registered after `%cash_on` through a
  `user_expressions` request (which does not go through `run_cell`, so cash never sees harness
  code) and deep-copies it; replay.py fetches the copy after the cell. (`%cash_status` in a
  separate request does not work: that request becomes the "last cell".)
* **Decorator, per call (cash mode):** `CASH_VERBOSE=1` lines on stderr,
  `cash.calls: HIT|MISS|RAISE module.fn [key] reason (ran|saved X.XXs)`, in completion order. In
  the kernel they arrive on the cell's stderr, so decorated calls inside notebook cells are counted too.
* **Notebook, calls inside statements (v2):** `_last_cell_metrics` drops each statement's
  `decorator_calls` (cash's intercepted helper calls and `@cash.cache` calls with `cache_hit` and
  `time_saved`). replay.py wraps `_update_last_cell_metrics` on the `CashMagics` instance
  (read-only, harness side, cash's code is unchanged) to keep them as `cash.subcalls`, one list per
  entry of `cash.statements`.
* **Plain, per call:** `r34hook_site.py` is installed as `sitecustomize` (PYTHONPATH, plain mode
  only) and patches `cash.core.Cash.cache` when cash is imported, so every decorated function
  is wrapped in a timer that records the function name, argument digest and function identity.
  Its own time is subtracted from the step's wall time (`wall_adj`). An existing
  `sitecustomize` in the venv still runs.

## compare.py

Compares per step: `status`, `error`, `out`, `disp`, `rc`, every `value` (plus the set of names),
every project file's digest **as the cumulative state after the step** (plus the set of files).
Every item that differs between plain1 and plain2 is masked. The rest is diffed for
plain1-cash1, plain1-cash2 and cash1-cash2 -> `compare.json` (`diffs`, `groups` with the first
step and number of steps each difference persists, `blocking`) and `compare.md`. A step that is
`ok` in both plain runs and `error` under cash is a BLOCKING candidate. stderr is not compared.

## speed.py

All seconds are the replayed steps' wall time (kernel start, restarts, cells, commands; not
setup, not the harness's own digest requests). P = min(plain1, plain2), C = min(cash1, cash2).

* **Z1** = P minus the plain time of each statement/call the cash run (the one giving C) served.
  **served = label-based + wall-based credit for inner hits.** cash's statements are mapped to
  the plain top-level statements (cash reports a loop body once per iteration, marked
  `# __iteration_context__`, and cuts code at 100 characters; same count without loop rows ->
  by position). Per plain statement the credit is the largest of:
  (a) label: all mapped rows `RESTORED`/`SKIPPED` -> full plain time; otherwise the saved share
  `saved/(saved+execution)`; (b) wall: cash time clearly below plain time
  (`cash < 0.7 x plain` and `plain - cash > 0.2 s`) -> `plain - cash` (this catches a statement
  labelled `COMPUTED` whose intercepted helper calls or `@cash.cache` calls hit, e.g. a loop of
  helper calls); (c) sub-calls: the `time_saved` of cash's own call hits inside the statement
  (`cash.subcalls`, recorded by replays from v2 on); capped at the plain time. A cell with no
  usable cash statements (e.g. it raised: `stale`) falls back to the same wall rule on the step
  walls. Script calls are matched by step + completion order (aligned on the function name);
  a `HIT` subtracts the plain call's inclusive time (its children are not counted again). A
  decorated call that hits inside a cell statement that got no other credit counts too.
  speed.json `served_by_method` gives the seconds per method.
* **Z2** = P minus the plain time of each statement/call that repeats one already run earlier in
  the session with the same code and inputs. Statement key: its text, the digests of the names it
  reads (values before the cell, or as written by an earlier statement in the same cell; functions
  and modules by code identity), and the contents of project files named in its text, in string
  variables it reads, or imported by it. Import statements are never free (a new process must
  import). Call key: function identity (code, defaults, globals it reads, recursively) + argument
  digest (with file contents for path arguments). A free call/statement does not count its
  children. Calls inside a statement that is not free are checked one by one. Code identities
  are paired with the current project `.py` files (a module by its name + those files), since
  the namespace digests are only refreshed at run steps.
  Per step, Z2 uses max(model free time, Z1 served time): a perfect cache saves at least what
  cash saved (cash also reuses intercepted helper calls inside a statement, a finer grain than
  the statement model). `Z2_model` in speed.json is the model alone; missed reuse = Z1 - Z2 is
  computed after the served credit.
* Top 10 steps by overhead (`cash - (plain - served)`) and by missed reuse (`free - served`), with
  cell id and statement/call excerpts; noise = spread between the two runs of each mode (totals and
  top steps).

Z2 is a model, not a measurement; known gaps: a name read only inside a function body that is
not a global of that function, global RNG state, and files reached through computed paths are not
inputs, so Z2 may be slightly optimistic there; it may be pessimistic where cash proves a changed
input irrelevant. Compare ratios across machines, not seconds.

## Example projects (`example/`)

* `example/nb`: notebook, ~7 s plain: cells, re-runs, an error then fix, an in-notebook function
  edit, out-of-order runs, a `script`, a helper-module edit (`write_file`), a data change
  (`shell`), a restart then Run All.
* `example/deco`: decorator, ~14 s plain: script runs with changed args, an error run, a function
  body edit, a data change, a new cached function, then `cell`/`run` (with an out-of-order
  NameError), `restart`, `run`.
* `example/make_traces.py` regenerates both `trace.json` files. Their `runs/` folders hold the
  last end-to-end results (both: no differences, no BLOCKING candidates).

### Per-cell stats (v4, 2026-10-16)

Cash mode records, per cell, next to `cash`:

* `cash.raw`: cash's own decision record per statement, read in the same tap on
  `_update_last_cell_metrics`: `status`, `execution_time`, `saved_time`, `miss_reason`,
  `uncacheable_reasons`, `skipped_reason` (why it was not stored, e.g. below the store floor),
  `guard_cause`, `storage`, `cash_tax`, `compute_cost`, `cost_model_*`; `cash.subcalls` entries
  gained `miss_reason`.
* `cash.work`: counts from `cs_stats.py`, installed in the kernel after `%cash_on` through a
  `user_expressions` request: `{counter: [calls, seconds, bytes]}` since the previous cell, for
  `hash_bytes` (bytes through `cash.bulk_digest.leaf_hasher`; small values cash hashes directly
  are not counted), `frame_copy` / `value_copy` / `frame_copy_values` (cash's copies before
  storing), `deepcopy`, `pickle` (from cash's own code), `referrer_scan` (`gc.get_referrers`
  from cash; the wrapper takes its own argument tuple out of the answer, so cash sees what it
  saw before), `source_parse` / `source_read`, `store` / `restore` seconds; `_own` = the
  counters' own time. `CS_STATS=0` turns the counters off.
* `warnings`: `{"Category [CODE]": count}` parsed from the cell's whole stderr.
* A last line `op: end` with `cache_bytes` / `cache_files` (the cache dir at the end).

`stats.py RUNDIR` classifies each cell run as `first` (first run of that cell id), `repeat`
(ran before in the same kernel) or `after_restart` (ran before, first time since the latest
restart), sums P, C, Z1, Z2 per class from speed.json, and gives every cell that was not served
its reason: cash's own (`skipped_reason`, `miss_reason`, `guard_cause`, `uncacheable_reasons`),
else an inferred one marked "(inferred)": first run, edited code, or after a restart whether
the earlier result was kept in memory only, on disk, or not stored. `tools/stats_rollup.py ROOT`
rolls every stats.json under ROOT into one REPORT section (`ROOT/ROLLUP.md`).

replay.py runs on Windows too (temp dir via `tempfile`, `Scripts\python.exe` venvs,
`platform.node()`, UTF-8 files; trace shell commands go to bash when it is on PATH, else cmd).

## Limitations

* Traces must not contain cash magics (`%cash_on`, `%load_ext cash`) in cells: the harness adds
  them in cash mode, and plain mode would fail on them.
* Cash's per-statement record is missing for a cell that raised (`stale`): such steps count as
  nothing served.
* Statement times in plain mode exclude IPython's own per-cell overhead (`wall` includes it).
* The `/tmp` disk on machine A had 3 GB free; cash sizes its cache to 80% of free disk.

## Changelog

* v1 (READY): first complete version, tested end to end on both examples.
* v2 (23:40Z): speed.py: served = label-based + wall-based credit for inner hits (cash labels a
  loop of helper-call hits `COMPUTED` with `saved_time` 0, found by machine E on p09: Z1 was = P);
  cash statements grouped to plain statements across loop iterations; step-wall fallback; Z2 per
  step >= served, module identities refreshed after edits (`Z2_model` kept). replay.py records
  `cash.subcalls`. The nb example gained a loop whose helper calls hit. Backward compatible:
  re-run `speed.py RUNDIR` on existing runs (no replay needed).

**Project venv outside the shared folder (added 23:35Z):** `/mnt/project-files` has no symlinks, so `uv venv
project/.venv` fails there. Put the venv under `/home/user/r34/venvs/<ID>` and name it in trace.json as
`"venv": "/home/user/r34/venvs/<ID>"` (or set `R34_VENV`). Order: `$R34_VENV`, trace.json `venv`, `PROJECT_DIR/.venv`.
- v3 (00:40Z): call-line regex accepts logging prefixes ("INFO cash.calls: ..."); script error = last exception line, not a trailing cash summary line. Re-score affected runs; replays with prefixed logging need a re-replay of the cash passes.
- v4 (2026-10-16): per-cell stats (`cash.raw`, `cash.work`, `warnings`, `op: end`), `stats.py`, Windows support. compare.py / speed.py skip the `end` line; results otherwise unchanged.
