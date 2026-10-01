"""A forked child writes its own stored-key records and exits.

The record of stored keys (``<cache>/.keys/``) is rewritten by a background
writer holding ``StoredKeyRecord._io_lock``. A program that forked while that
writer was mid-rewrite -- ``os.fork()`` right after a burst of misses -- gave
the child a lock nobody would release: its first record write blocked, and
``sys.exit`` then waited on that write forever.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from cash.decorator.stored_keys import StoredKeyRecord

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")


def _in_child(body) -> int:
    """Run *body* in a forked child; its exit code, or -1 if it hung."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        code = 1
        try:
            code = 0 if body() else 3
        except BaseException:
            code = 2
        finally:
            os._exit(code)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.02)  # poll for the child's exit
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    return -1


def test_a_child_forked_mid_record_write_writes_and_exits(tmp_path):
    record = StoredKeyRecord(lambda: str(tmp_path))
    record.note_stored("mod.f", "call:s1:a:k1", None, lambda state: {})
    record.flush()

    # The parent's writer is mid-rewrite: it holds the I/O lock at the fork.
    holding, release = threading.Event(), threading.Event()

    def writer_mid_rewrite():
        with record._io_lock:
            holding.set()
            release.wait(30)

    thread = threading.Thread(target=writer_mid_rewrite, daemon=True)
    thread.start()
    assert holding.wait(5)
    try:

        def body():
            record.note_stored("mod.f", "call:s1:a:k2", None, lambda state: {})
            record.close()
            with open(record.path("mod.f"), encoding="utf-8") as fh:
                return "call:s1:a:k2" in json.load(fh)["keys"]

        assert _in_child(body) == 0
    finally:
        release.set()
        thread.join(5)
        record.close()
