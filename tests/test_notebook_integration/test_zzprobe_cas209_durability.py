"""CAS-209 probe: does the cold kernel's cache survive shutdown?

Hypothesis under test: the atomic write is not WRONG, it is SLOWER (mkstemp +
write + replace vs one open+write), and the kernel shutdown does not drain
pending writes. If so the cold process simply dies with entries still queued,
and the warm process misses them -- no error anywhere, which matches the
observed "no write failures logged".

Counts what actually reached disk after shutdown.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


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


def test_probe_entries_on_disk_after_shutdown(nb_runner, tmp_path):
    import os

    cache_dir = tmp_path / "cache"
    nb_runner.create_notebook(_chain_cells(str(cache_dir)))
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = nb_runner.get_output(3)
    print(f"\ncold: {cold.strip()}", flush=True)

    def counts(label):
        files = os.listdir(cache_dir) if cache_dir.exists() else []
        data = [f for f in files if f.endswith(".data")]
        meta = [f for f in files if f.endswith(".meta")]
        part = [f for f in files if f.endswith(".part")]
        print(f"  {label}: {len(data)} .data  {len(meta)} .meta  {len(part)} .part",
              flush=True)
        return len(data), len(meta)

    before = counts("BEFORE shutdown")
    nb_runner.shutdown()
    after = counts("AFTER  shutdown")

    print(f"\n  expected 9 data entries (top/mid/load x 3 args)", flush=True)
    print(f"  VERDICT: {'ALL LANDED' if after[0] >= 9 else 'ENTRIES LOST — shutdown did not drain'}",
          flush=True)
