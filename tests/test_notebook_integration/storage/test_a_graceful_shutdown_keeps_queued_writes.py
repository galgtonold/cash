"""What a kernel's cache writes survive: a graceful shutdown and a hard kill.

A decorated call returns before its entry is on disk: the file backend writes
in the background, and a cell does not wait for its writes either. So writes
can still be queued when the shutdown comes. That is simulated by holding the
writer at a gate. A graceful exit (JupyterLab's restart button sends
``shutdown_request``) opens the gate (an exit handler registered after
``Cash()``, so it runs before cash's own exit drain) and cash drains the
queue: nothing is lost. A hard kill (a crash, the OOM killer, force-quit)
cannot drain anything, so the queued entries are lost. What holds even then:
every entry file on disk is whole (each is written header last, or to a
temporary file renamed into place), and a new kernel serves what is there and
recomputes the rest correctly.
"""

import asyncio

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

_CALLS = (1, 2, 3)
_FUNCTIONS = ("load", "mid", "top")

# The writer waits before each write, at a gate an exit handler opens.
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


def _chain_cells(cache_dir: str):
    cdir = cache_dir.replace("\\", "/")
    setup = "import time\nfrom cash import Cash, FileBackend\n"
    setup += _HELD_WRITER.format(hold=30)
    setup += f"c = Cash(backend=FileBackend(cache_dir='{cdir}'))\n"
    setup += "atexit.register(_gate.set)  # after Cash(): runs before its exit drain\n"
    # A cell leaves at most MAX_BACKLOG_S of writing queued; lifted, so the
    # held writes are all still queued when the shutdown comes.
    setup += "import cash.backends._writes as _w\n_w.MAX_BACKLOG_S = float('inf')"
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
@pytest.mark.parametrize("graceful", [True, False], ids=["queued_graceful", "queued_hard_kill"])
def test_what_a_shutdown_keeps(nb_runner, tmp_path, graceful):
    # This test's whole subject is killing the kernel, so it has to own the one
    # it kills. Under CASH_TEST_REUSE_KERNEL=1 it would otherwise destroy the
    # worker's SHARED warm kernel -- the only test in the suite that reaches
    # past the runner to `km.shutdown_kernel` directly.
    cache_dir = str(tmp_path / "cache")
    nb_runner.create_notebook(_chain_cells(cache_dir))
    nb_runner.start_kernel(with_cash=False)  # the decorator alone
    nb_runner.run_all()
    assert nb_runner.peek("vals") == _VALS

    before, _ = _stored(cache_dir)
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
    if graceful:
        assert after == _ALL, f"lost by the shutdown: {sorted(_ALL - after)}"

    # A new kernel serves what is on disk and recomputes only what is not.
    nb_runner.shutdown()
    nb_runner.start_kernel(with_cash=False)  # the decorator alone
    nb_runner.run_all()
    assert nb_runner.peek("vals") == _VALS
    tops_on_disk = sum(1 for fn, _ in after if fn == "top")
    assert int(nb_runner.peek("info['hits']")) == tops_on_disk, (sorted(after), nb_runner.peek("info"))
    assert int(nb_runner.peek("info['misses']")) == len(_CALLS) - tops_on_disk
