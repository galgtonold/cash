"""Edit one cell, run a later one: cash must give what a plain top-to-bottom run gives.

See ``replay_harness`` for what is compared and ``replay_corpus`` for the
notebooks and edits. ``CASH_REPLAY_REPORT=<dir>`` writes one JSON file per
scenario (verdict, what was re-run and really recomputed, seconds) for
measuring a change.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from replay_corpus import SCENARIOS, expected_recompute  # noqa: E402
from replay_harness import fresh_trace_file, run_with_cash  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_replay_matches_a_plain_run(scenario, nb_runner):
    nb_runner._force_fresh_kernel = True  # the kernel must inherit CASH_TRACE_FILE
    trace = fresh_trace_file()
    try:
        result = run_with_cash(scenario, nb_runner, trace)
    finally:
        os.unlink(trace)
    report = os.environ.get("CASH_REPLAY_REPORT")
    if report:
        # One file per scenario: parallel workers appending to one file
        # interleave their lines.
        os.makedirs(report, exist_ok=True)
        name = "".join(c if c.isalnum() else "_" for c in result.scenario) + ".json"
        with open(os.path.join(report, name), "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "scenario": result.scenario,
                        "ok": result.ok,
                        "stdout_ok": result.stdout_ok,
                        "bad_files": result.bad_files,
                        "rerun": result.rerun,
                        "seconds": round(result.seconds, 2),
                        "written": sorted(result.written),
                        "recomputed": result.recomputed,
                        "explain": result.explain(),
                    }
                )
                + "\n"
            )
    assert result.ok, result.explain()
    assert sorted(result.recomputed) == sorted(expected_recompute(scenario)), (
        "recomputed more (or less) than the cell needs\n" + result.explain()
    )
