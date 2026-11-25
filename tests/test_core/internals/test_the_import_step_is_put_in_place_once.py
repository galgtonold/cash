"""cash's step in the import system is put in place once, whichever
threads take their first look at the process registries together.

Around a cached body cash compares the process's registries (loggers,
warnings filters, ``atexit``), and leaves out what a module does to them
while it is first imported. For that it puts a step of its own in front of
``importlib``'s loader, on the first comparison of the process, and keeps
the loader it replaced to call on. Two threads making their first cached
call at once each put the step in place, the second over the first, and
kept the step itself as the loader to call on: from then on every first
import of a module, anywhere in the process, raised RecursionError.

The second thread's part is replayed here as it ran: it had seen that
nothing was in place yet, then found the first thread's step there.
"""

from __future__ import annotations

import importlib
import sys
import threading

import pytest

from cash.tracking import process_registries


@pytest.fixture
def step_as_installed(monkeypatch):
    """The module's state put back after the test, whatever the test did
    to it; a first snapshot of the process taken before."""
    process_registries.changed(process_registries.snapshot())
    bootstrap = sys.modules["importlib._bootstrap"]
    assert bootstrap._load_unlocked is process_registries._load_unlocked
    monkeypatch.setattr(process_registries, "_original_load", process_registries._original_load)
    monkeypatch.setattr(process_registries, "_open", process_registries._open)
    return bootstrap


def _import_fresh(tmp_path, monkeypatch, name: str):
    (tmp_path / f"{name}.py").write_text("VALUE = 7\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)


def test_a_second_thread_arriving_late_does_not_make_the_step_call_itself(step_as_installed, tmp_path, monkeypatch):
    # The second thread: it saw no step in place, and puts one in now.
    process_registries._open = None
    process_registries._watch_imports()

    assert process_registries._original_load is not process_registries._load_unlocked
    before = process_registries.snapshot()  # the step does its work only while one is open
    try:
        assert _import_fresh(tmp_path, monkeypatch, "imported_after_two_first_snapshots").VALUE == 7
    finally:
        process_registries.changed(before)


def test_snapshots_taken_and_ended_on_other_threads_do_not_fail_an_import(step_as_installed, tmp_path, monkeypatch):
    """The step lists the open snapshots while other threads add theirs
    and take them out: the list is made under the lock they change it
    under."""
    held = process_registries.snapshot()
    order: list[str] = []
    real_list = process_registries._open.__class__.__iter__

    def listing(self):
        # The class is shared by every weak set of cash's: only this one.
        if self is process_registries._open:
            order.append("listed under the lock" if process_registries._open_lock._is_owned() else "listed unlocked")
        return real_list(self)

    monkeypatch.setattr(process_registries._open.__class__, "__iter__", listing)
    try:
        assert _import_fresh(tmp_path, monkeypatch, "imported_while_snapshots_come_and_go").VALUE == 7
    finally:
        process_registries.changed(held)
    assert order and set(order) == {"listed under the lock"}, order

    # And adding or ending one waits for that lock.
    blocked = threading.Event()
    done = threading.Event()

    def other():
        blocked.set()
        process_registries.changed(process_registries.snapshot())
        done.set()

    with process_registries._open_lock:
        worker = threading.Thread(target=other)
        worker.start()
        assert blocked.wait(10)
        assert not done.wait(0.2), "a snapshot was added while the lock was held"
    assert done.wait(10)
    worker.join()
