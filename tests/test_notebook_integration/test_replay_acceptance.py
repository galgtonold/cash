"""Edit one cell, run a later one: cash must give what a plain top-to-bottom run gives.

See ``replay_harness`` for what is compared and ``replay_corpus`` for the
notebooks and edits. ``CASH_REPLAY_REPORT=<path>`` appends one JSON line per
scenario (verdict, what was re-run, seconds) for measuring a change.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from replay_corpus import SCENARIOS  # noqa: E402
from replay_harness import fresh_trace_file, run_with_cash  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_replay_matches_a_plain_run(scenario, nb_runner):
    nb_runner._force_fresh_kernel = True      # the kernel must inherit CASH_TRACE_FILE
    trace = fresh_trace_file()
    try:
        result = run_with_cash(scenario, nb_runner, trace)
    finally:
        os.unlink(trace)
    report = os.environ.get("CASH_REPLAY_REPORT")
    if report:
        with open(report, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"scenario": result.scenario, "ok": result.ok,
                                 "stdout_ok": result.stdout_ok, "bad_files": result.bad_files,
                                 "rerun": result.rerun, "seconds": round(result.seconds, 2),
                                 "written": sorted(result.written),
                                 "explain": result.explain()}) + "\n")
    assert result.ok, result.explain()
