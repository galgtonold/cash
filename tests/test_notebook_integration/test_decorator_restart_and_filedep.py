"""Integration tests (real kernel) for the decorator-path stress-test fixes.

These run ``@cash.cache`` chains inside a live Jupyter kernel and use a real
kernel restart (``shutdown()`` + ``start_kernel()``) so they exercise the
cross-process / restart behaviour the unit tests can only approximate.

* #7  — after a restart, the FIRST call to every cached function in a
  ``depends_on`` chain restores from disk (no warm-up recompute).
* #10 — the restored run reports a non-zero ``total_time_saved``.
* #11 — a file change invalidates a downstream cached function that only reads
  the file via a nested cached call.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]


def _chain_cells(cache_dir: str, sleep: float = 0.15):
    cdir = cache_dir.replace("\\", "/")
    return [
        "import cash\nfrom cash import Cash, FileBackend\n"
        f"c = Cash(backend=FileBackend(cache_dir='{cdir}'))",
        # A real depends_on chain; the leaf does enough work to be worth caching.
        "import time\n"
        "def base(x):\n    return x + 1\n"
        "@c.cache\n"
        "def load(x):\n"
        f"    time.sleep({sleep})\n"
        "    return base(x) * 2\n"
        "@c.cache(depends_on=[load])\n"
        "def mid(x):\n    return load(x) + 10\n"
        "@c.cache(depends_on=[mid])\n"
        "def top(x):\n    return mid(x) + 100",
        # Consume the chain for a few args, then report stats.
        "vals = [top(s) for s in (1, 2, 3)]\n"
        "info = top.cache_info()\n"
        "print(f'RESULT hits={info[\"hits\"]} misses={info[\"misses\"]} "
        "saved={info[\"total_time_saved\"]:.3f} vals={vals}')",
    ]


@pytest.mark.restore
def test_decorator_chain_restores_on_first_call_after_restart(nb_runner, tmp_path):
    """#7/#10: a fresh kernel restores the chain on the first call (misses=0)
    and credits the avoided compute (saved>0)."""
    nb_runner.create_notebook(_chain_cells(str(tmp_path / "cache")))
    nb_runner.start_kernel()

    # Cold run: everything computes.
    nb_runner.run_all()
    cold = nb_runner.get_output(3)
    assert "RESULT hits=0 misses=3" in cold, cold

    # Real kernel restart: a brand-new process pointed at the same on-disk cache.
    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.run_all()
    warm = nb_runner.get_output(3)

    # #7: the first call to top() after restart hits - zero warm-up recompute.
    assert "hits=3 misses=0" in warm, warm
    # #10: the restored run reports real saved time (~3 x 0.15s), not ~0.
    saved = float(warm.split("saved=")[1].split(" ")[0])
    assert saved > 0.2, warm
    # And the values are identical across the restart.
    assert cold.split("vals=")[1] == warm.split("vals=")[1]


@pytest.mark.files
def test_decorator_file_dep_propagates_through_chain(nb_runner, tmp_path):
    """#11: changing a file invalidates a downstream cached function that reads
    it only via a nested cached call."""
    data = tmp_path / "data.txt"
    data.write_text("hello")
    dpath = str(data).replace("\\", "/")
    cdir = str(tmp_path / "cache").replace("\\", "/")

    nb_runner.create_notebook([
        "import cash\nfrom cash import Cash, FileBackend\n"
        f"c = Cash(backend=FileBackend(cache_dir='{cdir}'))\n"
        f"DATA = '{dpath}'",
        "@c.cache\n"
        "def load():\n"
        "    with open(DATA) as f:\n        return f.read().strip()\n"
        "@c.cache(depends_on=[load])\n"
        "def upper():\n    return load().upper()",
        "print('upper =', upper())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "upper = HELLO" in nb_runner.get_output(3)

    # Change the underlying file, then re-run only the consumer cell.
    import time
    time.sleep(0.05)
    data.write_text("world-changed")

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "upper = WORLD-CHANGED" in out, out


def test_no_spurious_impurity_warnings_in_kernel(nb_runner, tmp_path):
    """#3/#6/#9: in a live kernel, a cached chain with local-only mutations and a
    @c.cache(depends_on=[...]) decorator must not raise an impurity warning. We
    promote CashImpurityWarning to an error so any regression fails the cell."""
    cdir = str(tmp_path / "cache").replace("\\", "/")
    nb_runner.create_notebook([
        "import warnings\nimport cash\n"
        "from cash import Cash, FileBackend, CashImpurityWarning\n"
        "warnings.simplefilter('error', CashImpurityWarning)\n"  # warning -> exception
        f"c = Cash(backend=FileBackend(cache_dir='{cdir}'))",
        "import numpy as np\n"
        "@c.cache\n"
        "def build(n):\n"
        "    pos = np.zeros(n)\n"          # fresh local array
        "    for i in range(n):\n        pos[i] = i\n"   # #3 local subscript mutation
        "    rows = []\n    rows.append(1)\n"            # #6 local list.append
        "    return float(pos.sum()) + len(rows)\n"
        "@c.cache(depends_on=[build])\n"   # #9 decorator line must not be analyzed
        "def consume(n):\n    return build(n) + 1",
        "print('OK', consume(5))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    # If any of #3/#6/#9 regressed, the cell would have raised instead.
    # consume(5) = (sum([0..4])=10.0 + len([1])=1) + 1 = 12.0
    assert "OK 12.0" in out, out
