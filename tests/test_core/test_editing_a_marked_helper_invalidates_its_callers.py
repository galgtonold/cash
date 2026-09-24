"""Editing a helper marked ``@cash.pure`` or ``@cash.stateful`` invalidates the
cached functions that call it, exactly as editing an unmarked helper does.

A marker says what a helper may DO. It does not say its code cannot change what
it returns, so the helper's code stays in every caller's key: in the process
that sees the edit, and in the next process that runs the edited file.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import cash
from cash import Cash
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.core]

MARKERS = ["@cash.pure", "@cash.stateful", ""]
IDS = ["pure", "stateful", "unmarked"]

HELPER = """\
import cash

{marker}
def helper(x):
    return x + {step}
"""


CALLER = """\
import {helpers}


def total(x):
    return {helpers}.helper(x)


def make_total(module):
    def total(x):
        return module.helper(x)

    return total
"""


@pytest.mark.parametrize("reach", ["global", "closure"])
@pytest.mark.parametrize("marker", MARKERS, ids=IDS)
def test_an_edit_in_this_process_invalidates_the_caller(tmp_path, monkeypatch, marker, reach):
    """``closure``: the caller reaches the helper through a captured module,
    so no name in its globals is rebound by the reload and only the helper's
    own identity can move the key."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(tmp_path))
    helpers = f"_marked_helpers_{tmp_path.name}"
    caller = f"_marked_caller_{tmp_path.name}"
    for name in (helpers, caller):
        monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{helpers}.py").write_text(HELPER.format(marker=marker, step=1), encoding="utf-8")
    (tmp_path / f"{caller}.py").write_text(CALLER.format(helpers=helpers), encoding="utf-8")

    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    caller_module = importlib.import_module(caller)
    if reach == "global":
        total = c.cache(caller_module.total)
    else:
        total = c.cache(caller_module.make_total(importlib.import_module(helpers)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # @stateful is reported; not the point here
        assert total(1) == 2
        assert total(1) == 2
        assert total.cache_info()["hits"] == 1  # the entry is really served

        (tmp_path / f"{helpers}.py").write_text(HELPER.format(marker=marker, step=100), encoding="utf-8")
        importlib.reload(sys.modules[helpers])

        assert total(1) == 101


JOB = """\
import os
import sys
import time

import cash

{marker}
def helper(x):
    return x + {step}

@cash.cache
def total(x):
    fd = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")  # @cash:assume-safe
    os.close(fd)  # @cash:assume-safe
    time.sleep({sleep})  # @cash:assume-safe
    return helper(x)

print("ANSWER", total(1))
"""


def _run(project: Path, marker: str, step: int) -> str:
    (project / "job.py").write_text(
        JOB.format(marker=marker, step=step, sleep=ABOVE_PERSISTENCE_FLOOR_S), encoding="utf-8"
    )
    src = str(Path(cash.__file__).resolve().parents[1])  # the cash under test
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(project / ".cash")
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")])
    env["PYTHONWARNINGS"] = "ignore"
    done = subprocess.run(
        [sys.executable, "job.py", str(project / "runs")],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.split("ANSWER")[-1].strip()


@pytest.mark.timeout(300)
@pytest.mark.parametrize("marker", MARKERS, ids=IDS)
def test_an_edit_between_two_runs_invalidates_the_caller(tmp_path, marker):
    assert _run(tmp_path, marker, 1) == "2"
    assert _run(tmp_path, marker, 1) == "2"
    assert (tmp_path / "runs").read_bytes() == b"x"  # the second run was served from disk

    assert _run(tmp_path, marker, 100) == "101"
    assert (tmp_path / "runs").read_bytes() == b"xx"
