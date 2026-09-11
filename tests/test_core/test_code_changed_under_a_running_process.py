"""The code that ran is the code the entry is keyed by.

Round 19, two ways the two came apart, both persisting a wrong answer:

* A helper replaced under a running job by a copy that KEEPS an older mtime
  (``shutil.copy2``, ``cp -p``, rsync, robocopy, Explorer). The loaded-vs-disk
  check trusted "mtime older than the process" to mean "not edited", so the
  new text keyed the old code's result, and the next process -- running the
  new code -- was served it (r19s4).
* A helper edited while a cached call ran: a pool worker started during the
  call imported the new code, and the result was stored under the key read
  from the old text. Reverting the edit then served the edited code's numbers
  (r19s5).
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

JOB = '''\
import sys, time
from pathlib import Path
import cash
from helper import smooth

@cash.cache
def work(n):
    time.sleep(0.2)  # @cash:assume-safe
    return smooth(n)

go = Path(sys.argv[1]) if len(sys.argv) > 1 else None
print("READY", flush=True)
while go is not None and not go.exists():
    time.sleep(0.02)
print("ANSWER", work(5), flush=True)
'''

V1 = "def smooth(x):\n    return x + 1\n"
V2 = "def smooth(x):\n    return x + 1000\n"


def _env(proj):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CASH_") and k != "PYTHONDONTWRITEBYTECODE"}
    env["CASH_CACHE_DIR"] = str(proj / ".cash")
    return env


def _answer(out):
    return out.split("ANSWER")[-1].strip()


@pytest.mark.parametrize("warm_first", [False, True], ids=["pyc-written-by-this-run", "pyc-older-than-the-run"])
def test_a_helper_replaced_by_a_copy_that_keeps_its_old_mtime_is_not_keyed_by_the_new_text(
        tmp_path, warm_first):
    """`warm_first` runs the job once beforehand, so the helper's .pyc is
    older than the process that sees the replacement: the fast path that reads
    the .pyc header must still notice it."""
    (tmp_path / "job.py").write_text(JOB, encoding="utf-8")
    (tmp_path / "helper.py").write_text(V1, encoding="utf-8")
    staged = tmp_path / "staged.py"
    staged.write_text(V2, encoding="utf-8")
    two_hours_ago = time.time() - 7200
    os.utime(staged, (two_hours_ago, two_hours_ago))
    env = _env(tmp_path)
    if warm_first:
        first = subprocess.run([sys.executable, "job.py"], cwd=str(tmp_path), env=env,
                               capture_output=True, text=True, timeout=120)
        assert _answer(first.stdout) == "6", first.stderr[-2000:]
        time.sleep(1.1)       # the next process starts after the .pyc was written

    go = tmp_path / "GO"
    running = subprocess.Popen([sys.executable, "job.py", str(go)], cwd=str(tmp_path), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert running.stdout.readline().strip() == "READY"
    shutil.copy2(staged, tmp_path / "helper.py")        # the deploy: V2, with an OLD mtime
    go.write_text("go", encoding="utf-8")
    out, err = running.communicate(timeout=120)
    assert _answer(out) == "6", err[-2000:]                  # it still runs the code it loaded
    assert "KEY-SOURCE-CHANGED" in err

    fresh = subprocess.run([sys.executable, "job.py"], cwd=str(tmp_path), env=env,
                           capture_output=True, text=True, timeout=120)
    assert _answer(fresh.stdout) == "1005", "the old code's result was served for the new code"


# -- the store refuses what the code it was keyed by no longer is ------------ #

HELPER = "def scale(i):\n    return i * {K}\n"

JOBMOD = """\
import {helper}

def make(c, calls, on_run=None):
    @c.cache
    def total(n):
        calls.append(n)
        out = sum({helper}.scale(i) for i in range(n))
        if on_run is not None:
            on_run()
        return out
    return total
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A helper module and a module holding the cached function, which reads
    the helper as a module global -- the ordinary shape."""
    tag = f"{os.getpid()}_{time.monotonic_ns()}"
    helper, job = f"codemove_helper_{tag}", f"codemove_job_{tag}"
    path = tmp_path / f"{helper}.py"
    path.write_text(HELPER.format(K=2), encoding="utf-8")
    (tmp_path / f"{job}.py").write_text(JOBMOD.format(helper=helper), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    helper_module = importlib.import_module(helper)
    job_module = importlib.import_module(job)
    yield helper_module, job_module, path
    sys.modules.pop(helper, None)
    sys.modules.pop(job, None)


def _edit(path, k):
    path.write_text(HELPER.format(K=k) + "# edited\n", encoding="utf-8")


def _code_changed(rec):
    return [w for w in rec if "STORE-CODE-CHANGED" in str(w.message)]


def test_a_helper_edited_while_the_call_runs_is_not_stored(tmp_path, project):
    _helper, job, path = project
    calls = []
    # A worker process a pool started at this point would import the edit.
    total = job.make(Cash(cache_dir=str(tmp_path / "cache")), calls, on_run=lambda: _edit(path, 3))

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert total(10) == 90
        assert total(10) == 90
    assert len(calls) == 2, "the result was stored under a key its code no longer matches"
    assert len(_code_changed(rec)) == 1
    assert str(path) in str(_code_changed(rec)[0].message)


def test_an_edit_elsewhere_in_the_helper_file_still_stores(tmp_path, project):
    """Control: only the code the call runs counts. A new function appended
    to the helper's file leaves `scale` as it was, so a worker would run the
    same code and the entry is right."""
    _helper, job, path = project
    calls = []

    def append_elsewhere():
        path.write_text(HELPER.format(K=2) + "\ndef unrelated():\n    return 0\n", encoding="utf-8")

    total = job.make(Cash(cache_dir=str(tmp_path / "cache")), calls, on_run=append_elsewhere)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert total(10) == 90
        assert total(10) == 90
    assert calls == [10]
    assert not _code_changed(rec)


def test_an_unedited_helper_still_stores_and_hits(tmp_path, project):
    """Control: the check must not refuse an ordinary store."""
    _helper, job, _path = project
    calls = []
    total = job.make(Cash(cache_dir=str(tmp_path / "cache")), calls)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert total(10) == 90
        assert total(10) == 90
    assert calls == [10]
    assert not _code_changed(rec)


def test_a_reloaded_helper_is_keyed_afresh_and_stores(tmp_path, project):
    """Control: an edit the process picks up (importlib.reload, %autoreload)
    makes new code objects, keyed by the new text -- they store normally."""
    helper, job, path = project
    calls = []
    total = job.make(Cash(cache_dir=str(tmp_path / "cache")), calls)

    assert total(10) == 90
    _edit(path, 3)
    importlib.reload(helper)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert total(10) == 135
        assert total(10) == 135
    assert calls == [10, 10]
    assert not _code_changed(rec)
