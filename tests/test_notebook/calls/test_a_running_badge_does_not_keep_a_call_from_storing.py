"""A RUNNING badge published while a ``@cash.cache`` body runs does not keep
the call's result from being stored.

A statement still running after ``BadgePresenter.MIN_RENDER_INTERVAL`` gets
its RUNNING badge from a timer thread, in the middle of whatever the
statement calls. cash refuses to store a decorated result when a
``unittest.mock`` object was called while the body ran, from any thread;
with the unit tests' shell publishing through a ``MagicMock``, that badge
made every decorated call slower than the interval unstored (on a loaded
CI runner, even a 0.2 s body), and a test that then looked for the call's
entry found none. The shell's publisher records instead
(`tests.conftest.RecordingDisplayPub`), as a real kernel's is no mock.

The badge's timer is released by the body starting rather than by the
interval, so the publish lands inside the body however long cash's own
work before it takes on a loaded machine.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from cash._active import default_cash, set_default_cash
from tests._cell_driver import run_cash_cell

#: Long enough for the released timer thread to publish inside it.
BODY_S = 1.0

LIB = (
    "import cash\n"
    "import time\n"
    "SPANS = []\n"
    "@cash.cache\n"
    "def make(n):\n"
    "    SPANS.append(time.perf_counter())\n"
    f"    time.sleep({BODY_S})\n"
    "    SPANS.append(time.perf_counter())\n"
    "    return list(range(n))\n"
)


@pytest.fixture
def lib(cash_magics, cash_instance, mock_shell, tmp_path, monkeypatch):
    """A module holding a cached ``make``, under the notebook's Cash as the
    default one."""
    previous = default_cash()
    set_default_cash(cash_instance)
    name = f"_running_badge_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"import {name} as lib")
    yield sys.modules[name]
    set_default_cash(previous)
    sys.modules.pop(name, None)


@pytest.fixture
def badge_fires_in_the_body(lib, monkeypatch):
    """The badge's timer fires once ``make``'s body has started, not after
    its interval; every other timer is left as it is."""
    real_timer = threading.Timer

    class BodyTimer(real_timer):
        def run(self):
            if getattr(self.function, "__module__", None) != "cash.notebook.ipython.badges":
                return super().run()
            deadline = time.monotonic() + 10
            while not lib.SPANS and not self.finished.is_set() and time.monotonic() < deadline:
                time.sleep(0.001)
            if not self.finished.is_set():
                self.function(*self.args, **self.kwargs)
            self.finished.set()

    monkeypatch.setattr(threading, "Timer", BodyTimer)


def test_the_call_is_stored_although_the_badge_was_published_while_it_ran(
    lib, badge_fires_in_the_body, cash_magics, cash_instance, mock_shell, monkeypatch
):
    stamps: list[float] = []
    publish = mock_shell.display_pub.publish

    def stamped(*args, **kwargs):
        stamps.append(time.perf_counter())
        return publish(*args, **kwargs)

    monkeypatch.setattr(mock_shell.display_pub, "publish", stamped)
    run_cash_cell(cash_magics, "x = lib.make(3)")

    start, end = lib.SPANS
    assert any(start < t < end for t in stamps), "no badge was published while the body ran"
    entries = [m for m in cash_instance.backend.list_entries() or () if m.get("decorator_entry")]
    assert len(entries) == 1, lib.make.cache_info()
    run_cash_cell(cash_magics, "y = lib.make(3)")
    assert len(lib.SPANS) == 2, "the second call ran the body again"
    assert mock_shell.user_ns["y"] == [0, 1, 2]
