"""Decorating a function means caching it, floor or no floor.

Found while attacking the decorator before round 26: a function under the 0.1 s
persistence floor is cached in RAM only, so running a script twice recomputes
every time and a tester sees no caching at all. Two of six agents' first sweeps
reported "no bugs found" for that reason alone -- nothing was ever stored, so
no staleness could appear. The floor belongs to the notebook, where cash caches
every statement by itself; ``@cash.cache`` is the user saying to cache this one.
The cost model still decides: a call cheaper to redo than to restore stays in
RAM, and ``explain()`` says so.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent('''
    import json
    import cash
    cash.configure(cache_dir=CACHE)

    calls = []

    @cash.cache
    def quick(n):
        import time
        time.sleep(0.02)          # well under the old 0.1s floor
        calls.append(n)
        return n * 2

    value = quick(21)
    print(json.dumps({"value": value, "ran": len(calls),
                      "misses": quick.cache_info()["misses"]}))
''')


def _run(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(PROGRAM.replace("CACHE", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(300)
def test_a_quick_function_survives_a_second_process(tmp_path):
    assert _run(tmp_path) == {"value": 42, "ran": 1, "misses": 1}
    assert _run(tmp_path) == {"value": 42, "ran": 0, "misses": 0}, "nothing reached disk"


@pytest.mark.timeout(300)
def test_the_entry_is_on_disk(tmp_path):
    _run(tmp_path)
    assert list((tmp_path / ".cash").glob("*.entry")), sorted(
        p.name for p in (tmp_path / ".cash").iterdir())


RESTORE_IS_SLOWER = textwrap.dedent('''
    import json
    import cash
    cash.configure(cache_dir=CACHE)

    @cash.cache
    def wide(n):
        return list(range(4_000_000))       # instant to build, slow to restore

    wide(1)
    print(json.dumps({"ok": True}))
''')


@pytest.mark.timeout(300)
def test_a_result_slower_to_restore_than_to_rebuild_stays_in_ram(tmp_path):
    """The guard for the above: dropping the floor must not put a value on disk
    that costs more to read back than to recompute."""
    script = tmp_path / "wide.py"
    script.write_text(RESTORE_IS_SLOWER.replace("CACHE", repr(str(tmp_path / ".cash"))),
                      encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    assert not list((tmp_path / ".cash").glob("*.entry")), "the cost model should have refused it"
