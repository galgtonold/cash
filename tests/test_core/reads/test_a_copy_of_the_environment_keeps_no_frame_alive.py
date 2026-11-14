"""Reading the whole environment keeps no frame, nor what it held, alive.

The read hook tells a copy of ``os.environ`` (each name looked up after
``__iter__``) from reads of one name by remembering the copying frame and
counting its lookups down. ``os.environ.items()`` looks every name up
through ``ItemsView.__iter__``, which was never counted down: the frame
stayed remembered, and with it every frame it was called from and their
locals. Cash's own config load, run in a cached call's key, read
``items()``, so the result of the cached call around it was held by that
frame and read as held elsewhere: it was never stored.
"""

from __future__ import annotations

import gc
import importlib
import os
import weakref

from cash.tracking import env_reads, io_watch


class _Held:
    pass


def _copy_all(held):
    return dict(os.environ.items())


def _copy_part(held):
    for i, _name in enumerate(os.environ):
        if i:
            break


def _gone_after(reader) -> bool:
    held = _Held()
    ref = weakref.ref(held)
    io_watch.hold()
    try:
        with env_reads.recording():
            reader(held)
    finally:
        io_watch.release()
    del held
    gc.collect()
    return ref() is None


def test_items_of_the_environment_leave_the_frame_free():
    assert _gone_after(_copy_all)


def test_a_copy_left_before_its_end_leaves_the_frame_free():
    assert _gone_after(_copy_part)


READER = (
    "import os\n"
    "def copy_then_read():\n"
    "    dict(os.environ.items())\n"
    "    return os.environ.get('CASH_UT_COPY_THEN_READ')\n"
)


def test_a_package_read_after_its_copy_still_counts(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_copy_then_read.py").write_text(READER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("CASH_UT_COPY_THEN_READ", "1")
    reader = importlib.import_module("cash_ut_copy_then_read")
    io_watch.hold()
    try:
        with env_reads.recording() as reads:
            assert reader.copy_then_read() == "1"
    finally:
        io_watch.release()
    assert reads == {"CASH_UT_COPY_THEN_READ"}
