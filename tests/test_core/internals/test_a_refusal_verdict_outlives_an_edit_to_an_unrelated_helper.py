"""A call refused for changing its argument in place keeps its verdict
(`refusal_verdicts`) across an edit to a helper that has nothing to do with
the change: the verdict is keyed by the function's own code, the code of
the helpers holding a change of state, and the argument types. A NiMARE
estimator's ``_fit`` reaches hundreds of library helpers, and every commit
touching any of them cost a full key build before the call was refused
again.

The verdict is dropped (the next call is keyed and checked) when what the
refusal rested on can have changed: the helper making the change is edited,
another helper starts changing state, or the argument's type differs. The
call behaves as plain Python every time, and shows the warnings a call
with no verdict shows."""

from __future__ import annotations

import json
import textwrap

from tests._scripts import run_python

N_OTHERS = 12

PREP = """
def prep(box):
    box.log.append(1)
"""

PREP_EDITED = """
def prep(box):
    box.log.append(1)
    box.log.append(2)
"""


HIDDEN = """
import bisect


def hidden(box):
    bisect.insort(box.log, 1)        # a change no finding names
"""

HIDDEN_STOPPED = """
def hidden(box):
    return None
"""


def _other(i, body):
    return f"\ndef other_{i}(x):\n    {body}\n"


def _helper(prep=PREP, edited=None, writing=None, hidden=HIDDEN):
    parts = [prep, hidden]
    for i in range(N_OTHERS):
        if i == writing:
            body = "global COUNT\n    COUNT = x\n    return x"
        elif i == edited:
            body = f"return x * {i} + 1"
        else:
            body = f"return x * {i}"
        parts.append(_other(i, body))
    return "COUNT = 0\n" + "".join(parts)


JOB = """
import json, sys, warnings
import cash
import helper

c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)


class Box:
    __slots__ = ("log",)

    def __init__(self):
        self.log = []


class OtherBox(Box):
    __slots__ = ()


def _total(n):
    return sum(getattr(helper, f"other_{i}")(n) for i in range(%(n)d))


@c.cache
def fit(box):
    helper.prep(box)                 # a hit cannot put this back: Box keeps slots
    return %(calls)s


@c.cache
def fit_hidden(box):
    helper.hidden(box)
    return %(calls)s


@c.cache
def fit_own(box):
    box.log.append(1)
    return %(calls)s


builds = []
real = c._keys.build
c._keys.build = lambda *a, **k: builds.append(1) or real(*a, **k)

which = {"fit": fit, "fit_own": fit_own, "fit_hidden": fit_hidden}[sys.argv[2]]
kind = {"box": Box, "other": OtherBox}[sys.argv[3]]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    boxes = [kind() for _ in range(3)]
    results = [which(b) for b in boxes]
shown = [w for w in caught if sys.argv[2] in str(w.message) and "in place" in str(w.message)]
print(json.dumps({
    "results": results,
    "logs": [b.log for b in boxes],
    "builds": len(builds),
    "warned": len(shown),
    "hits": which.cache_info()["hits"],
}))
"""


def _job():
    calls = " + ".join(f"helper.other_{i}(len(box.log))" for i in range(N_OTHERS))
    return textwrap.dedent(JOB) % {"n": N_OTHERS, "calls": calls}


def _run(tmp_path, which="fit", kind="box"):
    out = run_python("job.py", str(tmp_path / "cache"), which, kind, cwd=tmp_path)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _setup(tmp_path, **helper):
    (tmp_path / "job.py").write_text(_job(), encoding="utf-8")
    _edit(tmp_path, **helper)


def _edit(tmp_path, **helper):
    (tmp_path / "helper.py").write_text(_helper(**helper), encoding="utf-8")


def _settled(tmp_path, which="fit"):
    """Two processes: the first is refused and keeps a verdict, the second
    uses it from its first call."""
    first = _run(tmp_path, which)
    assert first["builds"] == 1, first
    second = _run(tmp_path, which)
    assert second["builds"] == 0, second
    return second


def _expected(prep_edited=False, edited=None):
    one = [1, 2] if prep_edited else [1]
    n = len(one)
    total = sum(n * i + (1 if i == edited else 0) for i in range(N_OTHERS))
    return {"results": [total] * 3, "logs": [one] * 3}


def test_an_edit_to_an_unrelated_helper_keeps_the_verdict(tmp_path):
    _setup(tmp_path)
    settled = _settled(tmp_path)
    _edit(tmp_path, edited=3)
    got = _run(tmp_path)
    assert {k: got[k] for k in ("results", "logs")} == _expected(edited=3)
    assert got["builds"] == 0, got
    assert got["warned"] == settled["warned"], got


def test_an_edit_to_an_unrelated_helper_keeps_the_verdict_of_a_function_changing_it_itself(tmp_path):
    _setup(tmp_path)
    settled = _settled(tmp_path, "fit_own")
    _edit(tmp_path, edited=7)
    got = _run(tmp_path, "fit_own")
    assert {k: got[k] for k in ("results", "logs")} == _expected(edited=7)
    assert got["builds"] == 0, got
    assert got["warned"] == settled["warned"], got


# --- what drops the verdict (these hold with the whole walk in the key too) ---


def test_an_edit_to_the_helper_making_the_change_drops_the_verdict(tmp_path):
    _setup(tmp_path)
    _settled(tmp_path)
    _edit(tmp_path, prep=PREP_EDITED)
    got = _run(tmp_path)
    assert {k: got[k] for k in ("results", "logs")} == _expected(prep_edited=True)
    assert got["builds"] == 1, got  # checked again, and refused again
    assert _run(tmp_path)["builds"] == 0


def test_a_helper_that_starts_changing_state_drops_the_verdict(tmp_path):
    _setup(tmp_path)
    _settled(tmp_path)
    _edit(tmp_path, writing=5)
    got = _run(tmp_path)
    assert got["builds"] == 1, got


def test_a_change_no_finding_names_keeps_every_helper_in_the_key(tmp_path):
    """Where the change was made is not among the findings: an edit to
    any helper is checked again, so the helper edited to stop changing the
    argument is stored in the next process."""
    _setup(tmp_path)
    _settled(tmp_path, "fit_hidden")
    _edit(tmp_path, edited=3)
    assert _run(tmp_path, "fit_hidden")["builds"] == 1
    _edit(tmp_path, edited=3, hidden=HIDDEN_STOPPED)
    got = _run(tmp_path, "fit_hidden")
    assert got["logs"] == [[], [], []], got
    assert got["hits"] == 2, got


def test_an_argument_of_another_type_drops_the_verdict(tmp_path):
    _setup(tmp_path)
    _settled(tmp_path)
    got = _run(tmp_path, kind="other")
    assert got["builds"] == 1, got
