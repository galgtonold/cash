"""A hit advances the global RNG as the computed call did.

Found while attacking the decorator before round 26: with ``np.random.seed(0)``
before a cached draw, the caller's NEXT draw came back equal to the cached
value -- the hit never consulted the stream, so the caller drew what the
function had drawn. ``known-limitations.md`` sends users to seeding as the
remedy ("Seed the RNG -- then the replay *is* the correct value"), which is the
notebook path, where the recorded state is replayed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent('''
    import json, time
    import numpy as np
    import random
    import cash
    cash.configure(cache_dir=CACHE_DIR)

    @cash.cache
    def draw_np(n):
        time.sleep(0.3)
        return float(np.random.rand())

    @cash.cache
    def draw_std(n):
        time.sleep(0.3)
        return random.random()

    np.random.seed(0)
    a = draw_np(1)
    b = float(np.random.rand())
    random.seed(7)
    c = draw_std(1)
    d = random.random()
    print(json.dumps({"cached_np": a, "next_np": b, "cached_std": c, "next_std": d}))
''')


def _run(tmp_path, env=None):
    import os
    script = tmp_path / "run.py"
    script.write_text(PROGRAM.replace("CACHE_DIR", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=180, cwd=str(tmp_path),
                          env={**os.environ, **(env or {})})
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(300)
def test_the_callers_next_draw_is_the_same_with_and_without_the_cache(tmp_path):
    computed = _run(tmp_path)                   # computes and stores
    assert computed["next_np"] != computed["cached_np"]
    hit = _run(tmp_path)                        # serves the stored value
    assert hit["cached_np"] == computed["cached_np"], "precondition: the call hit"
    assert hit["next_np"] == computed["next_np"], "the hit left the numpy stream unadvanced"
    assert hit["next_std"] == computed["next_std"], "the hit left random's stream unadvanced"
