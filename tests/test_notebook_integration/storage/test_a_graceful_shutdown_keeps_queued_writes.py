"""What a kernel's cache writes survive: a graceful shutdown and a hard kill.

A decorated call returns before its entry is on disk: the file backend writes
in the background. Once cash is imported in a kernel, every cell ends by
draining those writes, so what a finished cell cached is on disk whether the
kernel is then asked to exit (JupyterLab's restart button sends
``shutdown_request``) or killed outright (a crash, the OOM killer,
force-quit). The writer is slowed there, so the entries are on disk only
because the cell waited for them.

Writes can still be queued when the shutdown comes if the cell has not
finished. That is simulated by holding the writer at a gate and taking the
end-of-cell drain away. A graceful exit then opens the gate (an exit handler
registered after ``Cash()``, so it runs before cash's own exit drain) and cash
drains the queue: nothing is lost. A hard kill cannot drain anything, so the
queued entries are lost. What holds even then: every entry file on disk is
whole (each is written to a temporary file and renamed into place), and a new
kernel serves what is there and recomputes the rest correctly.
"""

import asyncio

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

_CALLS = (1, 2, 3)
_FUNCTIONS = ("load", "mid", "top")

# The writer waits before each write: at a gate an exit handler opens (queued
# cases), or for a fixed time long enough that the entries could not all be
# on disk when the cell ends unless something waited for them.
_HELD_WRITER = (
    "import atexit, threading\n"
    "from cash.backends.file_backend import FileBackend as _FB\n"
    "_gate = threading.Event()\n"
    "_write = _FB._do_set_sync\n"
    "def _held_write(self, *args, **kwargs):\n"
    "    _gate.wait({hold})\n"
    "    return _write(self, *args, **kwargs)\n"
    "_FB._do_set_sync = _held_write\n"
)

# After Cash(), which registers cash's magics and with them the end-of-cell drain.
_NO_END_OF_CELL_DRAIN = (
    "\n_ip = get_ipython()\n"
    "for _hook in list(_ip.events.callbacks['post_run_cell']):\n"
    "    if getattr(_hook, '__name__', '') == '_flush_pending_writes':\n"
    "        _ip.events.unregister('post_run_cell', _hook)\n"
    "atexit.register(_gate.set)  # after Cash(): runs before its exit drain"
)


def _chain_cells(cache_dir: str, queued: bool):
    cdir = cache_dir.replace("\\", "/")
    setup = "import time\nfrom cash import Cash, FileBackend\n"
    setup += _HELD_WRITER.format(hold=30 if queued else 0.2)
    setup += f"c = Cash(backend=FileBackend(cache_dir='{cdir}'))"
    if queued:
        setup += _NO_END_OF_CELL_DRAIN
    return [
        setup,
        "def base(x):\n    return x + 1\n"
        "@c.cache\n"
        "def load(x):\n"
        "    time.sleep(0.15)\n"
        "    return base(x) * 2\n"
        "@c.cache(depends_on=[load])\n"
        "def mid(x):\n    return load(x) + 10\n"
        "@c.cache(depends_on=[mid])\n"
        "def top(x):\n    return mid(x) + 100",
        f"vals = [top(s) for s in {_CALLS!r}]\ninfo = top.cache_info()",
    ]


def _expected(x):
    load = (x + 1) * 2
    return {"load": load, "mid": load + 10, "top": load + 110}


_ALL = {(fn, _expected(x)[fn]) for x in _CALLS for fn in _FUNCTIONS}
_VALS = str([_expected(x)["top"] for x in _CALLS])


def _stored(cache_dir):
    """``({(function, value)}, entry files)`` for the cache, read through the
    backend: every entry it can read, and how many entry files there are."""
    from cash import FileBackend

    backend = FileBackend(cache_dir=cache_dir)
    try:
        found = set()
        for entry in backend.entries():
            _meta, value = backend.get(entry.key)
            found.add((entry.metadata["func_name"].rsplit(".", 1)[-1], value))
        return found, backend.entry_count()
    finally:
        backend.shutdown()


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    ("queued", "graceful"),
    [(False, True), (False, False), (True, True), (True, False)],
    ids=["finished_graceful", "finished_hard_kill", "queued_graceful", "queued_hard_kill"],
)
def test_what_a_shutdown_keeps(nb_runner, tmp_path, queued, graceful):
    # This test's whole subject is killing the kernel, so it has to own the one
    # it kills. Under CASH_TEST_REUSE_KERNEL=1 it would otherwise destroy the
    # worker's SHARED warm kernel -- the only test in the suite that reaches
    # past the runner to `km.shutdown_kernel` directly.
    cache_dir = str(tmp_path / "cache")
    nb_runner.create_notebook(_chain_cells(cache_dir, queued))
    nb_runner.start_kernel(with_cash=False)  # the decorator alone
    nb_runner.run_all()
    assert nb_runner.peek("vals") == _VALS

    before, _ = _stored(cache_dir)
    if not queued:
        assert before == _ALL, f"not on disk when the cell finished: {sorted(_ALL - before)}"
    else:
        assert before == set(), f"writes landed before the shutdown began: {sorted(before)}"

    # The fresh-boot kernel manager is jupyter_client's AsyncKernelManager, so
    # shutdown_kernel() returns a coroutine: await it on the loop that drives
    # this kernel. Bounded, since that await can hang on a wedged provisioner
    # (see _force_kill_kernel). graceful is what JupyterLab's restart button
    # does: ask the kernel to exit and let it run its own shutdown.
    km = nb_runner.client.km
    nb_runner._run_async(asyncio.wait_for(km.shutdown_kernel(now=not graceful), timeout=60))
    assert not km.has_kernel, "the kernel is still running"

    after, files = _stored(cache_dir)
    assert files == len(after), "an entry file on disk is unreadable: a torn write"
    assert before <= after <= _ALL, sorted(after ^ _ALL)
    if graceful or not queued:
        assert after == _ALL, f"lost by the shutdown: {sorted(_ALL - after)}"

    # A new kernel serves what is on disk and recomputes only what is not.
    nb_runner.shutdown()
    nb_runner.start_kernel(with_cash=False)  # the decorator alone
    nb_runner.run_all()
    assert nb_runner.peek("vals") == _VALS
    tops_on_disk = sum(1 for fn, _ in after if fn == "top")
    assert int(nb_runner.peek("info['hits']")) == tops_on_disk, (sorted(after), nb_runner.peek("info"))
    assert int(nb_runner.peek("info['misses']")) == len(_CALLS) - tops_on_disk
