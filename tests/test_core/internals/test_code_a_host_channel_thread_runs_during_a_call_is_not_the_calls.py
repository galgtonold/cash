"""Code a host's message channel thread runs while a cached call is in
progress is not the call's.

pytest-xdist's channel thread (execnet's, which `threading` does not list)
runs its protocol code whenever the runner sends a worker its next tests,
in the middle of whatever call is running. The recorder took that code for
the call's: the class attributes it read (``WorkerGateway.id``,
``Marker.SHUTDOWN``) went into the library settings the entry is checked
on, and its code into the call's trace. Once the channel's own state moved,
or for a place no import finds (``__channelexec__``), every lookup found a
setting moved and the call ran again; and a trace naming code no scan
watches kept the hit memo from serving it. In a mixed worker a test here
and there missed a hit it had every right to.

Here a thread of a host package (`trace_recorder.HOST_PACKAGES`), started
unlisted as execnet's is, runs a library's code for each message, which a
body sends mid-call; between the calls the channel's state moves. The
entry is checked on none of what that thread read and the call hits. Work
the body hands to a thread of its own is still the call's.
"""

from __future__ import annotations

import _thread
import importlib
import os
import sys
import textwrap
import threading
import types

import pytest

from cash import Cash
from cash.decorator import hit_memo
from cash.tracking import trace_recorder

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

#: A library: the channel's protocol code, and a setting the bodies read.
LIBRARY = textwrap.dedent("""
    class Config:
        factor = 2


    def scale(x):
        return x * Config.factor


    class Marker:
        SHUTDOWN = "shutdown"


    class Gateway:
        id = "<worker>"

        def handle(self, message):
            return (self.id, Marker.SHUTDOWN, message)
""")

#: Body runs. A dict: a global the memo watches.
RUNS = {"n": 0}
#: The pipes a body talks to the channel through: (ask, done), each (read, write).
PIPES: dict[str, int] = {}
#: The library's ``scale``, set per test by the ``lib`` fixture.
scale = None


def ask_the_channel():
    """What the runner does mid-call: the channel thread handles a
    message while the body waits."""
    os.write(PIPES["ask_w"], b"a")
    os.read(PIPES["done_r"], 1)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    name = f"cash_ut_channel_{tmp_path.name.replace('-', '_')}"
    (site / f"{name}.py").write_text(LIBRARY, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    module = importlib.import_module(name)
    monkeypatch.setitem(globals(), "scale", module.scale)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def channel(lib, monkeypatch):
    """A host's message channel: a thread of a host package, started with
    ``_thread`` as execnet's is, that runs the library's protocol code for
    each message."""
    package = "cash_ut_host"
    monkeypatch.setattr(trace_recorder, "HOST_PACKAGES", trace_recorder.HOST_PACKAGES | {package})
    host = types.ModuleType(f"{package}.gateway")
    exec("def serve(run):\n    run()\n", host.__dict__)
    ask_r, ask_w = os.pipe()
    done_r, done_w = os.pipe()
    PIPES.update(ask_w=ask_w, done_r=done_r)
    ended = threading.Event()

    def run():
        gateway = lib.Gateway()
        while os.read(ask_r, 1) == b"a":
            gateway.handle("next tests")
            os.write(done_w, b"d")
        ended.set()

    _thread.start_new_thread(host.serve, (run,))
    yield lib
    os.write(ask_w, b"q")
    assert ended.wait(10)
    for fd in (ask_r, ask_w, done_r, done_w):
        os.close(fd)
    PIPES.clear()


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _scaled(x):
    RUNS["n"] += 1
    ask_the_channel()  # the runner sends its next tests, mid-call
    return scale(x)


def test_what_the_channel_read_is_not_checked_and_the_call_hits(c, channel):
    RUNS["n"] = 0
    f = c.cache(_scaled)
    assert f(3) == 6
    channel.Marker.SHUTDOWN = "closing"  # the channel's own state moves
    channel.Gateway.id = "<worker 2>"
    assert f(3) == 6
    info = f.cache_info()
    assert RUNS["n"] == 1, info["miss_reasons"]
    assert info["hits"] == 1


def test_a_setting_the_call_read_is_still_checked(c, channel):
    RUNS["n"] = 0
    f = c.cache(_scaled)
    assert f(3) == 6
    channel.Config.factor = 5
    assert f(3) == 15
    assert RUNS["n"] == 2
    assert "library setting changed" in f.cache_info()["miss_reasons"]


@pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension")
def test_the_hit_is_served_from_the_memo(c, channel):
    RUNS["n"] = 0
    f = c.cache(_scaled)
    assert f(2) == 4
    channel.Gateway.id = "<worker 3>"
    served = 0
    for _ in range(20):
        before = hit_memo.status()["fast_hits"]
        assert f(2) == 4
        served += hit_memo.status()["fast_hits"] > before
    assert RUNS["n"] == 1
    assert served, "a hit after a miss the channel ran code in was never served from the memo"


class Settings:
    level = 1


def level_of(x):
    return x + Settings.level


def _handed(x):
    RUNS["n"] += 1
    out = []
    worker = threading.Thread(target=lambda: out.append(level_of(x)))
    worker.start()
    worker.join()
    return out[0]


def test_work_the_body_hands_to_a_thread_of_its_own_is_still_the_calls(c):
    RUNS["n"] = 0
    f = c.cache(_handed)
    try:
        assert f(1) == 2
        assert f(1) == 2
        assert RUNS["n"] == 1
        Settings.level = 5
        assert f(1) == 6
        assert RUNS["n"] == 2
    finally:
        Settings.level = 1
