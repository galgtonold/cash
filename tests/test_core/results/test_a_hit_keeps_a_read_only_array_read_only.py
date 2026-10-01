"""A numpy array returned read-only is read-only on every hit too.

A constant table marked ``a.flags.writeable = False`` came back writable from
every hit: neither the RAM tier's copy nor a pickle keeps the flag, so an
accidental write that the miss refused succeeded silently on a hit.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

np = pytest.importorskip("numpy")


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
    hits = []
    for _ in range(2):
        p = run_python(script, cwd=tmp_path, cache_dir=tmp_path / "cache")
        hits.append(int(p.stdout.strip().splitlines()[-1]))
    assert hits == [1, 2], "the second process did not hit the disk entry"
