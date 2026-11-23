"""Whether a walk's report is stored does not hang on how the walk's CPU
time compares with working out its guards.

A report was left out of the store when its walk took less CPU time than
working out its guards plus reading a file back. But a walk left out works
its guards out again in every process that walks it (`report_store.save`
does so before it decides), just as reading the report back would: only the
read was ever worth weighing. Weighed against the guards, the cut-off hung
on two timings of a few ms: a function walked after others of its module
(their parses already held) takes about as long as its guards, and on the
macOS CI runners, where the guards' file reads cost more, a report was left
out and the next process walked again (``loaded 0, missed 1``; a module
instance's bound method read back as ``None``). Here the guards are made
slower than the walk, as there.
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
import json, sys, time
import cash
from cash.analysis import report_store

guards = report_store._guards
entries = []


def slow_guards(*args, **kwargs):
    # A runner where the guards' file reads cost more than the walk.
    start = time.thread_time()
    while time.thread_time() - start < 0.05:
        pass
    return guards(*args, **kwargs)


report_store._guards = slow_guards
entry = report_store._entry


def counted_entry(*args, **kwargs):
    entries.append(1)
    return entry(*args, **kwargs)


report_store._entry = counted_entry
if sys.argv[2] != "-":
    report_store._READ_COST = float(sys.argv[2])
import mod

f = cash.Cash(cache_dir=sys.argv[1]).cache(mod.top)
assert f(2) == 2 * (2 + mod.LIMIT)
report_store.flush()
print(json.dumps({"stats": report_store.status(), "entries": len(entries)}))
"""


def _run(tmp_path, read_cost: str = "-") -> dict:
    out = run_python("main.py", str(tmp_path / "cache"), read_cost, cwd=tmp_path)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write(tmp_path) -> None:
    (tmp_path / "mod.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    (tmp_path / "cache").mkdir()


def test_a_walk_cheaper_than_its_guards_is_read_back_next_time(tmp_path):
    _write(tmp_path)
    first = _run(tmp_path)
    assert first["stats"]["stored"] >= 1 and first["stats"]["cheap"] == 0, first
    again = _run(tmp_path)
    assert again["stats"]["loaded"] >= 1 and again["stats"]["missed"] == 0, again


def test_a_walk_too_cheap_to_store_works_out_no_guards(tmp_path):
    _write(tmp_path)
    first = _run(tmp_path, "1e9")
    assert first["stats"]["cheap"] >= 1 and first["stats"]["stored"] == 0, first
    assert first["entries"] == 0, first
