"""CAS-209 step 0: does the REAL restart path lose queued writes?

The harness kills with `shutdown_kernel(now=True)`. JupyterLab's restart button
goes through `restart_kernel()`, which by default sends a graceful
`shutdown_request` first — and a graceful exit runs atexit, which drains the
write queue.

If graceful drains, the data-loss window is crashes / OOM / force-quit only,
not the ordinary restart every user performs, and CAS-209 is a much smaller
problem than the hard-kill measurement suggests.

Run with the ATOMIC write applied (git checkout 2fce7ea -- src/cash/backends/
file_backend.py) — that is the configuration where loss is visible.
"""
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]


def _chain_cells(cache_dir: str, sleep: float = 0.15):
    cdir = cache_dir.replace("\\", "/")
    return [
        "import cash\nfrom cash import Cash, FileBackend\n"
        f"c = Cash(backend=FileBackend(cache_dir='{cdir}'))",
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
        "vals = [top(s) for s in (1, 2, 3)]\n"
        "info = top.cache_info()\n"
        "print(f'RESULT hits={info[\"hits\"]} misses={info[\"misses\"]}')",
    ]


def _counts(cache_dir):
    files = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
    return (len([f for f in files if f.endswith(".data")]),
            len([f for f in files if f.endswith(".meta")]),
            len([f for f in files if f.endswith(".part")]))


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("graceful", [False, True], ids=["hard_kill", "graceful"])
def test_probe_shutdown_path_durability(nb_runner, tmp_path, graceful):
    # This test's whole subject is killing the kernel, so it has to own the one
    # it kills. Under CASH_TEST_REUSE_KERNEL=1 it would otherwise destroy the
    # worker's SHARED warm kernel -- the only test in the suite that reaches
    # past the runner to `km.shutdown_kernel` directly.
    cache_dir = str(tmp_path / "cache")
    nb_runner.create_notebook(_chain_cells(cache_dir))
    nb_runner.start_kernel()
    nb_runner.run_all()

    before = _counts(cache_dir)

    km = getattr(nb_runner.client, "km", None)
    assert km is not None, "could not reach the KernelManager"

    if graceful:
        # What JupyterLab's restart button does: ask the kernel to exit, let it
        # run its own shutdown (and therefore atexit).
        km.shutdown_kernel(now=False)
    else:
        km.shutdown_kernel(now=True)

    after = _counts(cache_dir)
    label = "GRACEFUL" if graceful else "HARD KILL"
    print(f"\n  [{label}] before={before[0]}d/{before[1]}m/{before[2]}p  "
          f"after={after[0]}d/{after[1]}m/{after[2]}p", flush=True)
    print(f"  [{label}] VERDICT: "
          f"{'ALL 9 LANDED' if after[0] >= 9 and after[1] >= 9 else 'ENTRIES LOST'}",
          flush=True)
