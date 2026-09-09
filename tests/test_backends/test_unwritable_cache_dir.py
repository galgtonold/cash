"""An unwritable cache directory must cost a recompute, never the process.

Round-15 gate finding (BLOCKING). Point the cache at a directory the process
may read but not write and the job ran to completion, printed its result, and
then never exited: **28.7s writable, still running at 150s, 200s and -- the
first, accidental observation -- eleven minutes.** A minimal case was still
running at 420s for 24s of work. 5/5, with a clean writable control arm.

Two independent causes, either of which is enough on its own:

* ``tempfile.mkstemp`` SWALLOWS ``PermissionError`` and tries the next name, up
  to ``TMP_MAX`` = 10,000 times. The guard that is supposed to stop that is
  ``os.access(dir, W_OK)``, which on Windows reports the read-only attribute
  and knows nothing about ACLs, so it says "writable" about a directory that
  denies every write.
* ``ThreadPoolExecutor`` registers an ``atexit`` hook that joins every worker
  thread it started, with no timeout and no way to opt out. A task stuck in the
  OS holds the interpreter open however ``shutdown(wait=...)`` was spelled.

For a scheduled job this is worse than a crash: the work SUCCEEDED, so the logs
look healthy, the process never returns, and the next tick piles up behind it.
The contract these tests pin is the one the rest of this backend already
follows -- a cache write is best-effort, a failure warns and the computed value
still reaches the caller.

The subprocess arm is the load-bearing one. An in-process test can pass while
interpreter shutdown still hangs, because it never reaches interpreter
shutdown.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
import warnings

import pytest

from cash.backends._base import PendingWrites, discarded_writes
from cash.backends.file_backend import _TEMP_NAME_ATTEMPTS, FileBackend, _create_temp_file

# --------------------------------------------------------------------------- #
# Making a directory unwritable, on either platform                           #
# --------------------------------------------------------------------------- #

def _deny_writes(path: str) -> "callable":
    """Deny this account write access to *path*; return the undo.

    Windows needs an ACL (``icacls``) -- the read-only ATTRIBUTE does not apply
    to directories, and is exactly what ``os.access`` misreads. POSIX needs only
    a mode change, and root ignores it, so that case skips.
    """
    if sys.platform == "win32":
        user = f"{os.environ.get('USERDOMAIN', '')}\\{os.environ['USERNAME']}"
        result = subprocess.run(
            ["icacls", path, "/deny", f"{user}:(W,AD,WD,WEA,WA)"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"could not deny writes with icacls: {result.stderr.strip()}")

        def undo() -> None:
            subprocess.run(["icacls", path, "/remove:d", user],
                           capture_output=True, text=True)
        return undo

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root: file modes do not deny anything")
    previous = os.stat(path).st_mode
    os.chmod(path, 0o555)

    def undo() -> None:
        os.chmod(path, previous)
    return undo


@pytest.fixture
def unwritable_dir(tmp_path):
    """A real directory this process cannot write into, restored afterwards."""
    d = tmp_path / "ro_cache"
    d.mkdir()
    undo = _deny_writes(str(d))
    try:
        yield str(d)
    finally:
        undo()


# --------------------------------------------------------------------------- #
# The temp-file creation itself                                               #
# --------------------------------------------------------------------------- #

def test_a_permission_error_is_not_retried(tmp_path, monkeypatch):
    """One attempt, then out. Retrying it is what produced the hang.

    Injected rather than provoked so this runs identically on every platform;
    the ACL-driven arm below is the one that proves the injection matches
    reality.
    """
    attempts = []

    def denied(path, flags, mode=0o777, **kwargs):
        attempts.append(path)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "open", denied)
    with pytest.raises(PermissionError):
        _create_temp_file(str(tmp_path))
    assert len(attempts) == 1, (
        f"{len(attempts)} attempts on a permission error; tempfile.mkstemp makes "
        f"10,000 of them and that is the reported bug"
    )


def test_a_name_collision_is_retried(tmp_path, monkeypatch):
    """The one failure a different name can fix, and the only one retried."""
    real_open = os.open
    calls = {"n": 0}

    def collide_twice(path, flags, mode=0o777, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise FileExistsError(17, "File exists")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", collide_twice)
    fd, created = _create_temp_file(str(tmp_path))
    os.close(fd)
    assert calls["n"] == 3
    assert os.path.exists(created)


def test_endless_collisions_still_terminate(tmp_path, monkeypatch):
    """Bounded even in the case it does retry."""
    calls = {"n": 0}

    def always_exists(path, flags, mode=0o777, **kwargs):
        calls["n"] += 1
        raise FileExistsError(17, "File exists")

    monkeypatch.setattr(os, "open", always_exists)
    with pytest.raises(FileExistsError):
        _create_temp_file(str(tmp_path))
    assert calls["n"] == _TEMP_NAME_ATTEMPTS


def test_two_temp_files_do_not_collide(tmp_path):
    """The control for the above: distinct names, and both really created."""
    fd_a, a = _create_temp_file(str(tmp_path))
    fd_b, b = _create_temp_file(str(tmp_path))
    os.close(fd_a)
    os.close(fd_b)
    assert a != b
    assert os.path.exists(a) and os.path.exists(b)


# --------------------------------------------------------------------------- #
# The backend against a genuinely unwritable directory                        #
# --------------------------------------------------------------------------- #

@pytest.mark.expects_failed_writes
def test_a_write_to_an_unwritable_directory_gives_up_quickly(unwritable_dir):
    """The reported hang, at its source: a bounded write, on real permissions."""
    backend = FileBackend(cache_dir=unwritable_dir)
    before = len(discarded_writes())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started = time.monotonic()
        backend.set("k", b"x" * 1024, {})
        backend.shutdown()
        elapsed = time.monotonic() - started

    assert elapsed < 20, f"the write took {elapsed:.1f}s; it used to never finish"
    assert len(discarded_writes()) > before, (
        "the write neither succeeded nor was recorded as discarded"
    )


def test_the_unwritable_directory_is_announced(unwritable_dir):
    """A silent unwritable cache has no symptom but 'always slow'."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        FileBackend(cache_dir=unwritable_dir)._ensure_initialized()

    text = "\n".join(str(w.message) for w in rec)
    assert "CACHE-DIR-UNWRITABLE" in text, f"nothing said the directory is unwritable:\n{text}"
    assert unwritable_dir in text, "the warning must name the directory"


def test_a_writable_directory_says_nothing(tmp_path):
    """The control: the probe must not warn about ordinary cache directories."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        backend = FileBackend(cache_dir=str(tmp_path / "cache"))
        backend._ensure_initialized()
        backend.set("k", b"payload", {})
        backend.shutdown()

    assert "CACHE-DIR-UNWRITABLE" not in "\n".join(str(w.message) for w in rec)
    assert backend.get("k")[1] == b"payload", "and it still caches"


def test_the_probe_leaves_nothing_behind(tmp_path):
    """A writability probe that littered would be a new bug of its own."""
    cache_dir = tmp_path / "cache"
    backend = FileBackend(cache_dir=str(cache_dir))
    backend._ensure_initialized()
    assert [p.name for p in cache_dir.iterdir() if p.name.startswith(".probe-")] == []


# --------------------------------------------------------------------------- #
# Shutdown must not be able to block forever                                  #
# --------------------------------------------------------------------------- #

def test_shutdown_abandons_a_write_that_will_not_finish():
    """The second cause, isolated: a task that never returns.

    ``ThreadPoolExecutor`` would join this thread at interpreter exit whatever
    the timeout said, which is why the pool underneath is our own.
    """
    release = threading.Event()
    writes = PendingWrites()
    writes.submit("stuck", release.wait)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            started = time.monotonic()
            writes.shutdown(wait=True, timeout=0.5)
            elapsed = time.monotonic() - started
        assert elapsed < 10, f"shutdown blocked for {elapsed:.1f}s"
        assert "CACHE-WRITE-ABANDONED" in "\n".join(str(w.message) for w in rec), (
            "abandoning a write must be said out loud, not silently"
        )
    finally:
        release.set()


def test_shutdown_still_waits_for_a_write_that_finishes():
    """The control: the deadline must not truncate ordinary writes.

    Without this, the test above passes on a shutdown that waits for nothing.
    """
    done = []
    writes = PendingWrites()
    writes.submit("slow", lambda: (time.sleep(0.3), done.append("written")))
    writes.shutdown(wait=True, timeout=30)
    assert done == ["written"]


def test_the_writer_threads_are_daemons():
    """Structural, because the property is only observable at interpreter exit.

    A non-daemon writer is joined by the interpreter no matter what cash does,
    so this is the one thing the subprocess arm below cannot isolate.
    """
    started = threading.Event()
    writes = PendingWrites()
    writes.submit("x", started.set)
    started.wait(timeout=10)
    workers = [t for t in threading.enumerate() if t.name == "cash-cache-writer"]
    assert workers, "no writer thread was started"
    assert all(t.daemon for t in workers)
    writes.shutdown(wait=True, timeout=10)


# --------------------------------------------------------------------------- #
# The property that actually failed: the process EXITS                        #
# --------------------------------------------------------------------------- #

_CHILD = """
import os, sys
import cash

@cash.cache(assume_safe=True)
def work(n):
    return list(range(n))

r = work(200000)
print("WORK-DONE", len(r), flush=True)
"""


def test_the_process_exits_with_an_unwritable_cache_dir(unwritable_dir, tmp_path):
    """End to end, in a real interpreter, all the way through shutdown.

    In-process assertions cannot reach this: the hang was in the ``atexit``
    join, which only happens when a process actually ends. The child prints
    WORK-DONE before exiting, so a timeout here means "the work finished and
    the process still would not leave" -- the reported failure exactly.
    """
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(_CHILD), encoding="utf-8")
    env = dict(os.environ, CASH_CACHE_DIR=unwritable_dir)
    env.pop("PYTHONWARNINGS", None)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True,
            text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the child never exited with an unwritable cache directory -- this "
            "is the reported hang"
        )
    elapsed = time.monotonic() - started

    assert "WORK-DONE" in proc.stdout, (
        f"the child did not even finish its work:\n{proc.stdout}\n{proc.stderr}"
    )
    assert proc.returncode == 0, f"child exited {proc.returncode}:\n{proc.stderr}"
    assert elapsed < 90, f"the child took {elapsed:.1f}s to exit"
