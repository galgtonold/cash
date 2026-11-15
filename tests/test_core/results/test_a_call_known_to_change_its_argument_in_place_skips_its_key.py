"""A call that changes its argument in a way a hit cannot replay is never
stored. Once one call was refused for that, later calls with the same code
and argument types run uncached without building their key, in this
process and the next -- and they still behave as plain Python: the argument
is changed every call, and the warning is shown once per process. An edit
that stops the write in place, in the function or in a helper, is stored
again in the next process; what the verdict's key cannot see is probed again
within `REPROBE_EVERY` calls."""

from __future__ import annotations

import json
import textwrap

from cash.decorator.refusal_verdicts import REPROBE_EVERY
from tests._scripts import run_python

HELPER = """
def scale(x):
    return x * 2
"""

HELPER_WRITES = """
def scale(box):
    box.log.append(1)
    return 2
"""

HELPER_READS = """
def scale(box):
    return 2
"""

JOB = """
import json, sys, warnings
import cash
import helper

c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)


class Box:
    __slots__ = ("log",)

    def __init__(self):
        self.log = []


@c.cache
def fill(box, n, copy=False):
    if copy:
        box = Box()
    box.log.append(n)            # a hit cannot put this back: Box keeps slots
    return len(box.log) * 10


@c.cache
def through_helper(box):
    return helper.scale(box)


calls = int(sys.argv[2])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    results, lengths = [], []
    for i in range(calls):
        b = Box()
        results.append(fill(b, i))
        lengths.append(len(b.log))
    copy_box = Box()
    copied = [fill(copy_box, 1, copy=True) for _ in range(2)]
    hb = Box()
    helper_results = [through_helper(hb) for _ in range(2)]
codes = [getattr(w.message, "code", None) for w in caught if "fill" in str(w.message)]
info = fill.cache_info()
print(json.dumps({
    "results": results,
    "lengths": lengths,
    "copied": copied,
    "copy_box": len(copy_box.log),
    "helper": helper_results,
    "helper_log": len(hb.log),
    "helper_hits": through_helper.cache_info()["hits"],
    "fill_warnings": codes.count("IMPURE-SIDE-EFFECTS") + codes.count("IMPURE-OBSERVED-EFFECTS"),
    "hits": info["hits"],
}))
"""


def _run(tmp_path, calls):
    out = run_python("job.py", str(tmp_path / "cache"), str(calls), cwd=tmp_path)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _setup(tmp_path, helper=HELPER_WRITES):
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    (tmp_path / "helper.py").write_text(textwrap.dedent(helper), encoding="utf-8")


def test_every_call_changes_its_argument_as_plain_python_does_in_each_process(tmp_path):
    _setup(tmp_path)
    for _ in range(2):
        got = _run(tmp_path, 4)
        assert got["results"] == [10, 10, 10, 10]
        assert got["lengths"] == [1, 1, 1, 1]
        assert got["helper"] == [2, 2] and got["helper_log"] == 2


def test_a_call_with_copy_true_is_stored_beside_the_refused_ones(tmp_path):
    """A scalar argument is part of the verdict's key: ``copy=True`` leaves
    the caller's object alone, so it is stored and its second call hits."""
    _setup(tmp_path)
    got = _run(tmp_path, 3)
    assert got["copied"] == [10, 10] and got["copy_box"] == 0
    assert got["hits"] == 1


OBSERVED = """
import json, sys, warnings
import cash

c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)


class Box:
    def __init__(self):
        self.items = []


@c.cache
def fill(box):
    list.extend(vars(box)["items"], [1])   # seen only when the call runs
    return 1


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    boxes = [Box() for _ in range(3)]
    results = [fill(b) for b in boxes]
codes = [getattr(w.message, "code", None) for w in caught]
print(json.dumps([results, [len(b.items) for b in boxes], codes.count("IMPURE-OBSERVED-EFFECTS")]))
"""


def test_the_warning_is_shown_once_in_every_process(tmp_path):
    """The change only the call shows, not the source: its warning comes
    from checking a call, which the next process's calls skip, and is
    still shown there once."""
    (tmp_path / "observed.py").write_text(textwrap.dedent(OBSERVED), encoding="utf-8")
    for _ in range(3):
        out = run_python("observed.py", str(tmp_path / "cache"), cwd=tmp_path)
        assert json.loads(out.stdout.strip().splitlines()[-1]) == [[1, 1, 1], [1, 1, 1], 1]


def test_a_helper_edited_to_stop_writing_is_stored_in_the_next_process(tmp_path):
    _setup(tmp_path, HELPER_WRITES)
    assert _run(tmp_path, 1)["helper_hits"] == 0
    assert _run(tmp_path, 1)["helper_hits"] == 0
    (tmp_path / "helper.py").write_text(textwrap.dedent(HELPER_READS), encoding="utf-8")
    got = _run(tmp_path, 1)
    assert got["helper"] == [2, 2] and got["helper_log"] == 0
    assert got["helper_hits"] == 1


def test_a_function_edited_to_stop_writing_is_stored_in_the_next_process(tmp_path):
    _setup(tmp_path)
    assert _run(tmp_path, 2)["hits"] == 1  # only the copy=True call
    job = (tmp_path / "job.py").read_text(encoding="utf-8")
    (tmp_path / "job.py").write_text(
        job.replace("    box.log.append(n)            # a hit", "    box = Box(); box.log.append(n)  # a hit"),
        encoding="utf-8",
    )
    got = _run(tmp_path, 3)
    assert got["results"] == [10, 10, 10] and got["lengths"] == [0, 0, 0]
    got = _run(tmp_path, 3)
    assert got["results"] == [10, 10, 10] and got["lengths"] == [0, 0, 0]
    assert got["hits"] == 3 + 2  # every call of the edited code, stored by the run before


PROBE = """
import json, sys, warnings
import cash

c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)


class Box:
    __slots__ = ("log",)

    def __init__(self, log):
        self.log = log


@c.cache
def tag(box):
    if box.log:                  # whether it writes depends on the contents
        box.log.append(0)
    return 1


warnings.simplefilter("ignore")
writes = int(sys.argv[2])
calls = int(sys.argv[3])
for _ in range(writes):
    tag(Box([1]))
for _ in range(calls):
    tag(Box([]))
print(json.dumps(tag.cache_info()["hits"]))
"""


def test_what_the_verdict_cannot_see_is_probed_again_across_processes(tmp_path):
    """The verdict is keyed by the argument's type, not its contents: calls
    that would be stored run uncached until a probe stores one, at most
    `REPROBE_EVERY` calls later, counted across processes."""
    (tmp_path / "probe.py").write_text(textwrap.dedent(PROBE), encoding="utf-8")
    cache = str(tmp_path / "cache")
    run_python("probe.py", cache, "2", "0", cwd=tmp_path)
    hits = []
    for _ in range(REPROBE_EVERY // 4 + 1):
        out = run_python("probe.py", cache, "0", "4", cwd=tmp_path)
        hits.append(json.loads(out.stdout.strip().splitlines()[-1]))
    assert hits[0] == 0
    assert hits[-1] == 4, hits
