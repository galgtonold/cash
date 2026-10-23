"""Tornado's selector thread is the kernel's, not another thread.

On Windows the kernel's event loop cannot watch sockets itself, and tornado
runs a thread that does (`SelectorThread`): it waits on them and hands
what is ready back to the loop's own thread, running nothing of the
user's. It is alive the whole session there. Counted as another thread
that might change a value between two checks, it kept every statement
from handing its digests on: ``y = normalize(x)`` read ``x`` four times
on a first run, where Linux read it twice. The same thread is started
here, on any system, through the same wrapper.
"""

from __future__ import annotations

import asyncio
import socket
import threading

import pytest

from cash.tracking import trace_recorder

tornado_asyncio = pytest.importorskip("tornado.platform.asyncio")


def test_the_selector_thread_is_not_another_thread():
    async def look():
        a, b = socket.socketpair()
        wrapped = tornado_asyncio.AddThreadSelectorEventLoop(asyncio.get_running_loop())
        try:
            wrapped.add_reader(a.fileno(), lambda: None)
            await asyncio.sleep(0.05)
            names = [t.name for t in threading.enumerate()]
            return "Tornado selector" in names, trace_recorder.other_thread_alive({})
        finally:
            wrapped.remove_reader(a.fileno())
            wrapped._selector.close()
            a.close()
            b.close()

    started, alive = asyncio.run(look())
    assert started
    assert not alive
