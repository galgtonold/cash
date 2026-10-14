"""An object kept as pickle bytes reaches disk in the background, and stays.

The RAM tier keeps an object of a class of its own as its pickle bytes,
and the disk tier writes those very bytes. The write runs on the writer
thread: the store returns before it is done, so the next statement does
not wait for it. One the caller asked to keep (``@cash:persist``) is
written even when the cost model says it is not worth its disk, and a new
process reads it back.
"""

from __future__ import annotations

import sys
import textwrap
import threading

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from tests._scripts import run_python

MODULE = """
import numpy as np


class Fit:
    def __init__(self):
        self.design = np.arange(1 << 18, dtype=float)
        self.formula = "y ~ x"
"""


def _fit_class(tmp_path):
    (tmp_path / "fitlib.py").write_text(textwrap.dedent(MODULE), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        import fitlib
    finally:
        sys.path.remove(str(tmp_path))
    return fitlib.Fit


def test_the_store_returns_before_the_write_is_done(tmp_path, monkeypatch):
    Fit = _fit_class(tmp_path)
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    tiers = TieredBackend([InMemoryBackend(), disk])
    release = threading.Event()
    write = disk._do_set_sync

    def held_write(*args):
        release.wait(30)
        return write(*args)

    monkeypatch.setattr(disk, "_do_set_sync", held_write)
    store = threading.Thread(target=tiers.set, args=("k", Fit(), {"execution_time": 60.0, "force_persist": True}))
    store.start()
    store.join(10)
    try:
        assert not store.is_alive(), "the store waited for the disk write"
        assert isinstance(tiers.backends[0].get("k")[1], Fit), "the RAM tier waited for the disk write"
    finally:
        release.set()
        store.join(30)
        disk.shutdown()
    assert FileBackend(str(tmp_path / "c")).get("k")[1].formula == "y ~ x"


def test_one_the_caller_keeps_is_on_disk_for_a_new_process(tmp_path):
    """A 0.001 s compute: the cost model alone would keep 2 MiB off disk."""
    Fit = _fit_class(tmp_path)
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    tiers = TieredBackend([InMemoryBackend(), disk])
    metadata = {"execution_time": 0.001, "force_persist": True}
    tiers.set("k", Fit(), metadata)
    disk.shutdown()
    assert "DISK" in metadata["storage"]
    done = run_python(
        "-c",
        "from cash.backends import FileBackend; "
        f"fit = FileBackend({str(tmp_path / 'c')!r}).get('k')[1]; "
        "print(type(fit).__name__, fit.design.sum(), fit.formula)",
        cwd=tmp_path,  # where ``fitlib`` is
    )
    assert done.stdout.split() == ["Fit", str(float(sum(range(1 << 18)))), "y", "~", "x"]
