"""Decorator shapes: the steps run three times in one process, then twice
in a new process on the same cache folder (the restart), under cash or
plain Python."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools.plain_checker import CONFIGS, RUNS, Outcome, Shape

_SESSION = Path(__file__).with_name("_deco_session.py")


def run_decorator(shape: Shape, work: Path, config: str | None) -> Outcome:
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(CONFIGS[config] if config is not None else {})
    env["CASH_CACHE_DIR"] = str(work / ".cash")
    outcome = Outcome()
    for session, runs in enumerate((RUNS[:3], RUNS[3:])):
        job = {"setup": shape.setup, "steps": shape.steps, "runs": list(runs), "config": config}
        done = subprocess.run(
            [sys.executable, str(_SESSION)],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            cwd=work,
            env=env,
            timeout=600,
        )
        if done.returncode != 0:
            raise RuntimeError(f"session {session + 1} failed: {done.stderr[-2000:]}")
        got = json.loads(done.stdout.splitlines()[-1])
        outcome.seen.update(got["seen"])
        for run_name, statuses in got["statuses"].items():
            outcome.statuses[run_name] = {int(k): v for k, v in statuses.items()}
    return outcome
