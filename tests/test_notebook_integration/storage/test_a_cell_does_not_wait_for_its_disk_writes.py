"""A cell does not wait for its cache writes; whatever reads them still does.

The file tier writes in the background, and the next cell runs while it
does. Every reader of the disk copy still sees the writes queued before it:
a later cell listing the cache folder, a child process the kernel starts,
and the kernel after a restart (a graceful shutdown finishes the writes).

A cell does wait for what would take longer than ``MAX_BACKLOG_S`` to
write: Jupyter gives a restarting kernel 2.5 s to exit, and a killed kernel
loses what is queued. The writer is slowed so the nine entries take longer
than that, or less.
"""

import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_SLOW_WRITER = (
    "import time\n"
    "from cash.backends.file_backend import FileBackend as _FB\n"
    "_write = _FB._do_set_sync\n"
    "def _slow_write(self, *args, **kwargs):\n"
    "    time.sleep({write_s})\n"
    "    return _write(self, *args, **kwargs)\n"
    "_FB._do_set_sync = _slow_write\n"
)

# Run in a child process: every entry it can read back, as "fn=value" lines.
_CHILD = (
    "import sys\n"
    "from cash import FileBackend\n"
    "b = FileBackend(cache_dir=sys.argv[1], flush_interval=0)\n"
    "for e in b.entries():\n"
    "    print(e.metadata['func_name'].rsplit('.', 1)[-1] + '=' + str(b.get(e.key)[1]))\n"
)


def _cells(cache_dir: str, write_s: float = 0.15):
    cdir = cache_dir.replace("\\", "/")
    return [
        _SLOW_WRITER.format(write_s=write_s)
        + f"from cash import Cash, FileBackend\nc = Cash(backend=FileBackend(cache_dir='{cdir}'))",
        "def base(x):\n    return x + 1\n"
        "@c.cache\n"
        "def load(x):\n"
        "    return base(x) * 2\n"
        "@c.cache(depends_on=[load])\n"
        "def mid(x):\n    return load(x) + 10\n"
        "@c.cache(depends_on=[mid])\n"
        "def top(x):\n    return mid(x) + 100",
        "vals = [top(s) for s in (1, 2, 3)]\ninfo = top.cache_info()",
        # 4: a later cell reading the folder itself
        f"import glob\non_disk = len(glob.glob('{cdir}/*.entry'))",
        # 5: a child process reading the cache
        f"import subprocess, sys\nchild = subprocess.run([sys.executable, '-c', {_CHILD!r}, '{cdir}'], "
        "capture_output=True, text=True)\nseen = sorted(child.stdout.split())",
    ]


_ALL = sorted(
    f"{fn}={v}"
    for x in (1, 2, 3)
    for fn, v in (("load", (x + 1) * 2), ("mid", (x + 1) * 2 + 10), ("top", (x + 1) * 2 + 110))
)


def _pending(nb_runner) -> int:
    return int(
        nb_runner.peek(
            "sum(q.pending_count() for q in __import__('cash.backends._writes', fromlist=['x']).all_pending_writes())"
        )
    )


@pytest.mark.fresh_kernel
def test_the_cell_returns_before_its_writes_and_every_reader_sees_them(nb_runner, tmp_path):
    cache_dir = str(tmp_path / "cache")
    nb_runner.create_notebook(_cells(cache_dir))
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])

    nb_runner.run_cell(3)
    assert nb_runner.peek("vals") == "[114, 116, 118]"
    assert _pending(nb_runner) > 0, "every write had landed when the cell returned: it waited for them"

    # A later cell listing the folder sees all nine.
    nb_runner.run_cell(4)
    assert nb_runner.peek("on_disk") == "9"


@pytest.mark.fresh_kernel
def test_a_child_process_and_a_restarted_kernel_see_every_write(nb_runner, tmp_path):
    cache_dir = str(tmp_path / "cache")
    nb_runner.create_notebook(_cells(cache_dir))
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])
    assert _pending(nb_runner) > 0, "every write had landed: the writer was not slowed"

    nb_runner.run_cell(5)
    assert nb_runner.peek("seen") == str(_ALL), nb_runner.peek("child.stderr")

    # Queued again, then a restart: the kernel's shutdown finishes them.
    nb_runner.peek("[c.backend.delete(e.key) for e in c.backend.entries()] and None")
    nb_runner.run_cell(3)
    assert _pending(nb_runner) > 0
    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3])
    assert nb_runner.peek("info['hits']") == "3", nb_runner.peek("info")


@pytest.mark.fresh_kernel
def test_a_cell_waits_for_what_would_outlast_a_restart(nb_runner, tmp_path):
    """Nine one-second writes: the cell ends with about two seconds of them
    left, and a restart right away keeps all nine."""
    cache_dir = str(tmp_path / "cache")
    nb_runner.create_notebook(_cells(cache_dir, write_s=1.0))
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    t0 = time.perf_counter()
    nb_runner.run_cell(3)
    took = time.perf_counter() - t0
    left = _pending(nb_runner)
    assert 1 <= left <= 4, f"{left} one-second writes left when the cell returned"
    assert took >= 4.0, f"the cell took {took:.1f}s for 9s of writes"

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3])
    assert nb_runner.peek("info['hits']") == "3", nb_runner.peek("info")
