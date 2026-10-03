"""A hit moves a generator the function draws from to where the body left it.

Round 32 found this with a notebook helper and the same holds for
``@cash.cache``::

    rng = np.random.default_rng(42)
    @cash.cache
    def boot(x): ...rng.integers(...)...
    res = {g: boot(x) for g, x in groups.items()}   # 10 groups, then 60

The calls for the first ten groups hit, the hits left ``rng`` where it was,
and the next fifty drew from the wrong place in the stream: most results
differed from a plain run. And the generator was dropped from the key after
its first call moved it, so a stored value could come back for a stream
position it was not computed at.

The oracle is the same program without the decorator.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

PROGRAM = textwrap.dedent("""
    import json, sys
    import numpy as np
    import cash
    cash.configure(cache_dir=CACHE_DIR)
    N, SKEW, CACHED = int(sys.argv[1]), sys.argv[2] == "skew", sys.argv[3] == "cached"

    rng = np.random.default_rng(42)
    if SKEW:
        rng.normal()

    def boot(x):
        return float(rng.normal()) + x

    if CACHED:
        boot = cash.cache(boot)
    res = [boot(g) for g in range(N)]
    print(json.dumps({"res": res, "next": float(rng.normal())}))
""")


def _run(tmp_path, n, skew=False, cached=True):
    script = tmp_path / "run.py"
    script.write_text(PROGRAM.replace("CACHE_DIR", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    argv = (script, str(n), "skew" if skew else "plain", "cached" if cached else "plain")
    done = run_python(*argv, cwd=tmp_path, timeout=180)
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(300)
def test_more_calls_after_fewer_match_a_plain_run(tmp_path):
    assert _run(tmp_path, 4) == _run(tmp_path, 4, cached=False)
    # The first four hit; the rest must draw where a plain run would.
    assert _run(tmp_path, 12) == _run(tmp_path, 12, cached=False)


@pytest.mark.timeout(300)
def test_a_stream_moved_before_the_calls_is_not_served_old_values(tmp_path):
    _run(tmp_path, 4)
    assert _run(tmp_path, 4, skew=True) == _run(tmp_path, 4, skew=True, cached=False)
