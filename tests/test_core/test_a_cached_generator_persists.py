"""A cached generator reaches disk like any decorated result.

Its manifest and chunks were judged by the notebook's compute floor instead of
the decorator's rule (decorating a function is the decision to cache it). So a
quick generator stayed in RAM, and the next process missed with "no entry yet";
a slower one had its early chunks -- written before 0.1 s of production --
kept in RAM while the manifest reached disk, and every later process found the
entry incomplete and recomputed it, forever.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.timeout(300)

PROGRAM = textwrap.dedent("""
    import json, time
    import cash
    cash.configure(cache_dir=CACHE)

    runs = []

    @cash.cache
    def quick(n):
        runs.append("quick")
        yield from range(n)

    @cash.cache(chunk_max_items=3)
    def paced(n):
        runs.append("paced")
        for i in range(n):
            time.sleep(0.02)          # 0.2 s in all: the first chunks come early
            yield i

    assert list(quick(5)) == list(range(5))
    assert list(paced(10)) == list(range(10))
    print(json.dumps({"runs": runs, "explain": paced.explain(10).reason}))
""")


def _run(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(PROGRAM.replace("CACHE", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_cached_generator_is_a_hit_in_the_next_process(tmp_path):
    assert _run(tmp_path)["runs"] == ["quick", "paced"]
    second = _run(tmp_path)
    assert second["runs"] == [], "a later process recomputed a stored generator"
    assert second["explain"] == "hit"
