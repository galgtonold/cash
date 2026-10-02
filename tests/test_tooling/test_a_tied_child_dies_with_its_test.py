"""A build or install a test starts does not outlive the test.

``pip install`` in a subprocess kept running for minutes after the xdist
worker that started it was killed: a timeout on ``subprocess.run`` is
enforced by the process waiting on it, so it dies with that process.
``run_tied`` kills the command's whole tree on a timeout and when the test
process itself dies.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from tests._scripts import run_tied

REPO = Path(__file__).resolve().parents[2]

# Writes its own pid, then a child's, then sleeps: a command with a tree.
SLEEPER = (
    "import os, subprocess, sys, time\n"
    "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    "open(sys.argv[1], 'w', encoding='utf-8').write(f'{os.getpid()} {kid.pid}')\n"
    "time.sleep(600)\n"
)


def _procs(path: Path, within: float = 30.0) -> list[psutil.Process]:
    """The command's processes, as it reported them: those still running
    when asked, each held as a process so a pid reused later is not it."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text(encoding="utf-8").split()) == 2:
            procs = []
            for pid in path.read_text(encoding="utf-8").split():
                try:
                    procs.append(psutil.Process(int(pid)))
                except psutil.NoSuchProcess:
                    pass
            return procs
        time.sleep(0.05)  # poll for the command to report its pids
    raise AssertionError("the command never started")


def _alive(proc: psutil.Process) -> bool:
    """Is *proc* running? A process can exit between any two questions about
    it, so one that is gone by the time it is asked counts as gone."""
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _gone(procs: list[psutil.Process], within: float = 15.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in procs):
            return True
        time.sleep(0.1)  # poll for the processes to exit
    return False


def _sleeper(tmp_path: Path) -> tuple[list[str], Path]:
    pid_file = tmp_path / "pids"
    return [sys.executable, "-c", SLEEPER, str(pid_file)], pid_file


def test_a_timeout_kills_the_whole_tree(tmp_path):
    argv, pid_file = _sleeper(tmp_path)

    with pytest.raises(subprocess.TimeoutExpired):
        run_tied(argv, timeout=5)

    assert pid_file.exists()
    assert _gone(_procs(pid_file)), "the command outlived its timeout"


@pytest.mark.timeout(90)
def test_the_tree_dies_with_a_killed_test_process(tmp_path):
    argv, pid_file = _sleeper(tmp_path)
    test_process = subprocess.Popen(
        [sys.executable, "-c", f"from tests._scripts import run_tied\nrun_tied({argv!r}, timeout=600)"],
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])},
    )
    procs: list[psutil.Process] = []
    try:
        procs = _procs(pid_file)
        test_process.kill()  # what a crashed xdist worker amounts to
        test_process.wait(timeout=30)

        assert _gone(procs), "the command outlived the process that started it"
    finally:
        for proc in procs:
            try:
                proc.kill()
            except psutil.Error:
                pass
