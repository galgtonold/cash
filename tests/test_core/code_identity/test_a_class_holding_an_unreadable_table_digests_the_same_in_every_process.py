"""A class holding a table cash cannot read digests the same in every process.

Reported against scanpy checked out as the user's code: its pydantic
settings class holds dicts filled from sets of strings, in an order that
follows the process's string hashing, with values cash cannot digest. Each
attribute was written into the class's digest as far as it got before
turning out unreadable, so the class read as changed in every new process
and a traced call that ran one of its methods never hit. An attribute now
counts whole or as unreadable, and a change to one cash can read is still
seen.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

SCRIPT = textwrap.dedent("""
    import sys
    import threading

    from cash.tracking.trace_items import class_digest


    class Table:
        names = {name: len(name) for name in {"alpha", "beta", "gamma", "delta", "epsilon"}}
        names["lock"] = threading.Lock()
        scale = int(sys.argv[1])


    print(class_digest(Table))
""")


def _digest(tmp_path, seed, scale):
    (tmp_path / "table.py").write_text(SCRIPT, encoding="utf-8")
    proc = run_python("table.py", str(scale), cwd=tmp_path, env={"PYTHONHASHSEED": str(seed)})
    return proc.stdout.strip()


def test_the_digest_follows_the_class_not_the_string_hashing(tmp_path):
    digests = {_digest(tmp_path, seed, 1) for seed in (1, 2, 3, 4)}
    assert len(digests) == 1, digests
    assert _digest(tmp_path, 1, 2) not in digests
