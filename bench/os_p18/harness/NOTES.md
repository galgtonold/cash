# harness progress notes (machine A)

- 22:45Z worktree /home/user/r34/cash at 02cd17da, venv /home/user/r34/venv-harness (py3.13, cash editable, jupyter_client, ipykernel, pandas, numpy, polars, sklearn, scipy, matplotlib, pyarrow, pillow).
- cash data sources found:
  - notebook per statement: `CashMagics._last_cell_metrics` (the dict `%cash_status dict` returns as `last_cell`), snapshotted by a post_run_cell hook registered after `%cash_on`; fetched with a silent user_expressions request. (Calling `%cash_status` from a separate request returns an empty last_cell, so the snapshot is needed.)
  - decorator per call: `CASH_VERBOSE=1` stderr lines `cash.calls: HIT|MISS|RAISE fn [key] ... (ran|saved Xs)`, in completion order.
  - plain per call: sitecustomize hook patches `cash.core.Cash.cache` under CASH_DISABLE=1 to time each decorated call (+ argument digest, function identity).
- HTML badge displays are recognised by the `c3-wrap` css class and dropped from outputs.
- 23:05Z decorator example replays plain + cash ok.
- 22:55Z in-cell decorated calls: plain kernel also gets the call hook (calls tagged with the statement index), cash kernel parses CASH_VERBOSE lines from the cell's stderr; speed.py counts both.
- 22:58Z cash fills `_last_cell_metrics` before IPython's pre/post_run_cell events fire, so the snapshot hook is registered after %cash_on via user_expressions; a cell cash did not record (it raised) is marked `stale`.
- 23:00Z fixed: timing normalization ate "1.012\nmin" in describe() output (\s matched a newline); now space only. Z2 never counts import statements as free (a new process must import).
- 23:02Z Z2 function identity no longer includes the whole module file (was too pessimistic after adding an unrelated function); it is code + defaults + globals read, recursively.
- 23:03Z tested with a project .venv (uv venv + editable cash + ipykernel): ok. Oracle sensitivity checked by injecting a value/display/error change: all flagged.
- 23:06Z both examples end to end: 0 masked, 0 differences, 0 BLOCKING. READY written.
- 23:40Z v2 (coordinator request, p09 finding): cash's `_last_cell_metrics` reports each loop iteration as its own `COMPUTED` row with saved_time 0 even when the iteration's helper call was an intercepted cache hit (the hit is only in the raw `decorator_calls`, which magics.py:1240-1252 drops). speed.py now credits label + wall (cash < 0.7 x plain and > 0.2 s faster) + sub-call hits, groups iteration rows to their loop, falls back to step walls; Z2 per step >= served; module identity refreshed after file edits. p09: Z1 346.3 -> 266.9. replay.py taps `_update_last_cell_metrics` to record `cash.subcalls`. Both examples re-run: 0 differences.
- 2026-10-13 06:05Z (round 35, from machine H's p25 finding): speed.py scales call-based served and free time by a per-step parallel factor ((parent top-level call time + busiest worker's) / sum over processes), credits a clearly faster script/shell step at least plain - cash, and caps served and free at the step's plain wall. p25: Z1 -104 -> 254.6, Z2 -518 -> 183.2 (H's manual estimate 250 / 182). Single-process projects unchanged (round-34 p02, p10 checked). Re-run: `R34_PY=... python speed.py <runs dir>` (no replay needed).

2026-10-13 ~17:30Z: r34digest.py digested only estimators from modules starting with "sklearn"; LightGBM/XGBoost/CatBoost sklearn wrappers fell to the type-only digest, so every fitted model of a class digested equal and Z2 counted calls on new models as free. Now any object with get_params+fit uses _sk_digest. Published round-35 Z2 numbers for p17 (and projects with those libraries) overstate missed reuse; p17 corrected: after 6525d53 missed reuse 93.7 -> 1.5 s, before 166.3 -> 73.2 s (ledger2-nbscript.md).
- 2026-10-14 03:15Z: run_all.sh honours R35_RUNS (output folder override), for the cold-run measurement (round-35/cold/).
