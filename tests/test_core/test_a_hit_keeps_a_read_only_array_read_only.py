"""A numpy array returned read-only is read-only on every hit too.

A constant table marked ``a.flags.writeable = False`` came back writable from
every hit: neither the RAM tier's copy nor a pickle keeps the flag, so an
accidental write that the miss refused succeeded silently on a hit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

np = pytest.importorskip("numpy")

import cash

SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import cash

    @cash.cache
    def table():
        a = np.arange(5)
        a.flags.writeable = False
        return a

    @cash.cache
    def scratch():
        return np.arange(5)

    for _ in range(2):
        assert table().flags.writeable is False
        assert scratch().flags.writeable is True
    print(table.cache_info()["hits"])
    """
)


def test_a_read_only_array_stays_read_only_on_ram_and_disk_hits(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(SCRIPT, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    src = os.path.dirname(os.path.dirname(cash.__file__))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    hits = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
        assert p.returncode == 0, p.stderr
        hits.append(int(p.stdout.strip().splitlines()[-1]))
    assert hits == [1, 2], "the second process did not hit the disk entry"
