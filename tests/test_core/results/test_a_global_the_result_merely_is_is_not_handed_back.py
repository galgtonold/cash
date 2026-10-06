"""A hit is the stored value, not a module global the result happened to be.

Handing back a sentinel as itself (``return d.get(k, MISSING)``) looked for ANY
module global or closure variable that the result ``is``. ``best(candidates)``
returning the module-level ``BASELINE`` passed in as a candidate recorded that
global, and every later hit -- with fresh, equal candidates -- handed back
``BASELINE`` itself, with whatever it held by then. Only a variable the body
names is the function's sentinel; anything else is a copy of the stored value,
and CACHE-RESULT-SHARED says so on the computing run.
"""

from __future__ import annotations

import textwrap

from tests._scripts import run_python

SCRIPT = textwrap.dedent(
    """
    import os, sys
    import cash

    class Model:
        def __init__(self, w):
            self.w = w

        def score(self):
            return -abs(self.w - 1)

    BASELINE = Model(int(sys.argv[1]))

    @cash.cache
    def best(candidates):
        return max(candidates, key=Model.score)

    if sys.argv[2] == "global":
        r = best([BASELINE, Model(5)])
    else:
        r = best([Model(1), Model(5)])
    print(r.w, r is BASELINE)
    r2 = best([Model(1), Model(5)])
    print(r2.w, r2 is BASELINE)
    """
)


def test_a_hit_is_not_the_global_the_stored_result_was(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(SCRIPT, encoding="utf-8")
    first = run_python(script, "1", "global", cwd=tmp_path)
    assert first.stdout.split("\n")[:2] == ["1 True", "1 False"], first.stderr
    assert "CACHE-RESULT-SHARED" in first.stderr, "a hit gives a separate object, which the warning says"
    later = run_python(script, "7", "fresh", cwd=tmp_path)
    assert later.stdout.split("\n")[:2] == ["1 False", "1 False"], later.stderr
