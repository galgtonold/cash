"""A walk cheaper than its check is still stored when a report with its
text is stored already.

A fresh process stores a cached function's purity report only when the
walk cost more CPU time than working out the report's guards again
(`report_store._READ_COST`): walking a cheap function again costs less than
checking its report would. That holds only while nothing is stored for it.
Once a report with the function's text is (stored before a file it reads
was edited), every later process checks that report before it walks; a
walk left out of the store then spared no check and added a walk to every
process after it. On a slower machine (the macOS CI runners) a helper
reached through a subscript (``BAG[k]``, walked from ``Bag.__getitem__``)
fell just under the limit, and was walked again in every process.
"""

from __future__ import annotations

import json
import textwrap

from tests._scripts import run_python

_MODULE = """
LIMIT = 3


def helper(x):
    return x + LIMIT


def top(x):
    return helper(x) * 2
"""

_SCRIPT = """
import json, sys
import cash
from cash.analysis import report_store

# Every walk worth keeping, or every walk too cheap to keep.
report_store._READ_COST = float(sys.argv[2])
import mod

f = cash.Cash(cache_dir=sys.argv[1]).cache(mod.top)
assert f(2) == 2 * (2 + mod.LIMIT)
print(json.dumps(report_store.status()))
"""


def _run(tmp_path, read_cost: float) -> dict:
    out = run_python("main.py", str(tmp_path / "cache"), str(read_cost), cwd=tmp_path)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_cheap_walk_over_a_stale_report_is_read_back_next_time(tmp_path):
    (tmp_path / "mod.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    (tmp_path / "cache").mkdir()
    _run(tmp_path, -1.0)
    assert _run(tmp_path, -1.0)["loaded"] >= 1, "control: the report was not stored"
    # Not top's text: what the walk took from the rest of its file.
    (tmp_path / "mod.py").write_text(textwrap.dedent(_MODULE).replace("LIMIT = 3", "LIMIT = 4"), encoding="utf-8")
    walked = _run(tmp_path, 1e9)
    assert walked["stale"] >= 1, f"control: the stored report still held after the edit: {walked}"
    again = _run(tmp_path, 1e9)
    assert again["stale"] == 0 and again["loaded"] >= 1, f"the walk over a stale report was not stored: {again}"
