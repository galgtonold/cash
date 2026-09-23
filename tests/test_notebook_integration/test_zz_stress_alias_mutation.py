"""The alias / in-place mutation scenarios, many times at once, under load.

A stress sweep saw ``scen_alias_downstream_consumer`` print a total from the
pre-edit lineage once (342 instead of 642) -- a wrong answer -- and it never
came back in 72 repeats run alone. A failure that shows only when the machine
is busy needs the machine busy, and needs its evidence kept the one time it
happens: every repeat here records cash's decision trace, and a repeat that
disagrees with the top-to-bottom oracle leaves the trace and every cell's
output (badges included) in ``CASH_STRESS_KEEP`` (default: the test's tmp
folder) and names them in the failure.

Off unless asked for, as it takes minutes:

    CASH_STRESS_REPEAT=24 python -m pytest tests/test_notebook_integration/test_zz_stress_alias_mutation.py -n 16
    python scripts/run_integration_sweep.py --stress 24
"""

import json
import os
import pathlib
import shutil

import pytest
from test_eda_mutation_reconstruction import (
    _oracle,
    scen_alias_downstream_consumer,
    scen_alias_reflects_upstream_edit,
    scen_helper_edit_mutating,
    scen_multihop_inplace_reconstruction,
    scen_mutate_upstream_recompute,
)

from tests._nbharness.runner import NotebookTestRunner

pytestmark = [pytest.mark.libraries, pytest.mark.stress]

REPEAT = int(os.environ.get("CASH_STRESS_REPEAT", "0") or 0)

SCENARIOS = [
    scen_alias_downstream_consumer,
    scen_alias_reflects_upstream_edit,
    scen_mutate_upstream_recompute,
    scen_multihop_inplace_reconstruction,
    scen_helper_edit_mutating,
]


def _keep(tmp_path, name, r, trace):
    keep = pathlib.Path(os.environ.get("CASH_STRESS_KEEP") or tmp_path) / name
    keep.mkdir(parents=True, exist_ok=True)
    if trace.exists():
        shutil.copy(trace, keep / "trace.jsonl")
    cells = [{"cell": i, "source": c.source, "output": r.get_raw_output(i)} for i, c in enumerate(r.nb.cells, 1)]
    (keep / "cells.json").write_text(json.dumps(cells, indent=1), encoding="utf-8")
    return keep


@pytest.mark.timeout(300)
@pytest.mark.skipif(REPEAT <= 0, reason="set CASH_STRESS_REPEAT=N to run the stress repeats")
@pytest.mark.parametrize("repeat", range(max(REPEAT, 1)))
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.__name__ for s in SCENARIOS])
def test_repeated_under_load(scenario, repeat, tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    # a fresh kernel (no pool) inherits it, so the trace is this repeat's own
    monkeypatch.setenv("CASH_TRACE_FILE", str(trace))
    on_dir = tmp_path / "on"
    on_dir.mkdir(parents=True, exist_ok=True)
    r = NotebookTestRunner(work_dir=on_dir)
    try:
        captured, check = scenario(r)
        final_sources = [c.source for c in r.nb.cells]
        monkeypatch.delenv("CASH_TRACE_FILE")
        oracle = _oracle(tmp_path / "oracle", final_sources, check)
        diffs = {k: (oracle[k], captured[k]) for k in check if oracle[k] != captured[k]}
        if diffs:
            kept = _keep(tmp_path, f"{scenario.__name__}-{repeat}", r, trace)
            pytest.fail(
                f"cash-ON diverged from the top-to-bottom oracle in {scenario.__name__} "
                f"(repeat {repeat}); trace and cell outputs kept in {kept}:\n"
                + "\n".join(f"  [{k}] oracle={o!r}  cash={c!r}" for k, (o, c) in diffs.items())
            )
    finally:
        r.shutdown()
