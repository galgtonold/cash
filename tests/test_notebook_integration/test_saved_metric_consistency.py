"""The time a statement shows on COMPUTE must line up with what it saves on RESTORE.

A cell that builds a large DataFrame used to report two numbers that looked
contradictory to a user:

  * on the first run it showed the *total* wall-clock — the compute PLUS cash's
    own one-time cost of hashing and serialising the (large) result into the
    cache, e.g. "0.9s";
  * on restore the same statement credited only the compute it avoided, e.g.
    "saved 0.3s" — and, in the text badge, actually printed the tiny
    *deserialise* time under the "saved" label ("saved 0.01s").

Neither number was wrong on its own, but "executed 0.9s" next to "saved 0.3s"
(or "saved 0.01s") reads as a bug. The statement row now shows the pure compute
on COMPUTE and the avoided compute on RESTORE, so the two line up; cash's
serialisation cost is attributed to the cell's overhead section instead.

The ``time.sleep`` gives the compute a floor well above the serialisation and
restore noise so the assertion is meaningful, not measuring jitter.
"""
from __future__ import annotations

import re


BIG_DF = (
    "def _mk():\n"
    "    time.sleep(0.4)\n"
    "    return pd.DataFrame(np.random.default_rng(0).integers(0, 1000, (2_000_000, 4)))\n"
    "df = _mk()"
)


def _stream_text(cell) -> str:
    return "".join(
        o.get("text", "") for o in cell.get("outputs", [])
        if o.get("output_type") == "stream"
    )


def _set_print_badge(nb_runner) -> None:
    nb_runner._run_async(nb_runner.client.kc._async_execute_interactive(
        "get_ipython().run_line_magic('cash_badge', 'print')",
        store_history=False, output_hook=lambda m: None,
    ))


def test_computed_time_matches_saved_time(nb_runner):
    nb_runner.create_notebook(["import time, numpy as np, pandas as pd", BIG_DF])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_cell(1)
    _set_print_badge(nb_runner)

    nb_runner.run_cell(2)  # compute
    compute_badge = _stream_text(nb_runner.get_cell(2))
    nb_runner.run_cell(2)  # restore
    restore_badge = _stream_text(nb_runner.get_cell(2))

    exec_m = re.search(r"EXECUTED: df = _mk\(\)\s+\(([\d.]+)s\)", compute_badge)
    saved_m = re.search(r"(?<!NOT )CACHED: df = _mk\(\)\s+\(saved ([\d.]+)s\)", restore_badge)
    assert exec_m, f"no EXECUTED row for df = _mk():\n{compute_badge}"
    assert saved_m, f"no CACHED row for df = _mk():\n{restore_badge}"

    exec_t, saved_t = float(exec_m.group(1)), float(saved_m.group(1))
    # Both must reflect the ~0.4s compute, not the serialisation-inflated total
    # (which would push the computed number up) nor the deserialise time (which
    # would drop the saved number to ~0.01s).
    assert exec_t > 0.35, f"computed time {exec_t}s should reflect the ~0.4s compute"
    assert saved_t > 0.35, f"saved time {saved_t}s should reflect the ~0.4s compute"
    assert abs(exec_t - saved_t) < 0.1, (
        f"computed {exec_t}s and saved {saved_t}s should line up"
    )
