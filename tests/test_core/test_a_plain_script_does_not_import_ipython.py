"""A plain script that uses cash imports neither IPython nor psutil for it.

Finding out that no notebook is running imported IPython (50-160 ms on every
script start, in any environment that has it installed), and ``import cash``
imported psutil (about 11 ms) for readings most runs never take. A running
shell has always imported IPython, so ``sys.modules`` answers the question;
psutil is imported the first time a reading needs it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import cash
from cash import _location

pytestmark = [pytest.mark.core]

JOB = """\
import sys

import cash


@cash.cache
def f(x):
    return x + 1


print("ANSWER", f(1), f(1))
print("LOADED", "IPython" in sys.modules, "psutil" in sys.modules)
"""


def test_a_script_calling_a_cached_function_imports_neither(tmp_path):
    pytest.importorskip("IPython")  # nothing to avoid importing without it
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    src = str(Path(cash.__file__).resolve().parents[1])
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / ".cash")
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")])
    done = subprocess.run(
        [sys.executable, "job.py"], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "ANSWER 2 2" in done.stdout
    if sys.platform.startswith(("linux", "win32")):
        assert "LOADED False False" in done.stdout, done.stdout
    else:  # macOS: psutil still reads the process start time
        assert "LOADED False " in done.stdout, done.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="GetProcessTimes is the Windows reading")
def test_windows_reads_the_process_start_time_without_psutil_and_agrees_with_it():
    import psutil

    from cash.process_start import _windows_start_time

    assert abs(_windows_start_time() - psutil.Process().create_time()) < 0.1


def test_no_shell_is_running_until_ipython_is_imported(monkeypatch):
    """Answered from ``sys.modules``: asking does not import IPython."""
    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    assert _location.interactive_shell_is_running() is False
    assert "IPython" not in sys.modules


def test_a_running_shell_is_still_seen():
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.instance()
    try:
        assert _location.interactive_shell_is_running() is True
    finally:
        InteractiveShell.clear_instance()


def test_a_lazy_module_reads_and_patches_the_real_one(monkeypatch):
    import json

    from cash._lazy_module import LazyModule

    lazy = LazyModule("json")
    assert lazy.dumps is json.dumps
    monkeypatch.setattr(lazy, "dumps", lambda value: "patched")
    assert json.dumps(1) == "patched"  # the write reached the module itself
    monkeypatch.undo()
    assert json.dumps(1) == "1"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="read from /proc on Linux only")
def test_the_process_start_time_read_from_proc_is_psutils():
    """Read without psutil on Linux, and the same value psutil gives: a file
    edited after the process started is still told apart from one before."""
    import psutil

    from cash import process_start

    assert process_start._proc_start_time() == pytest.approx(psutil.Process().create_time(), abs=0.02)
