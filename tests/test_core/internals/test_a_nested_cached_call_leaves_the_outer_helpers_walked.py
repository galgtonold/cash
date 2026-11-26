"""A cached call nested in another's body does not make the outer one walk
its helpers again on every call.

A missed call's body runs under cash's reader wrappers (``os.path.isdir``
and the rest). The nested call's key build, inside that body, met a module
global such as ``os.path.isdir`` holding cash's wrapper; the outer call's
next key build, after the wrappers came out, met the builtin, took the
binding for changed and walked the helpers again: twice per outer call
(NiMARE's ``_fit`` calling a cached ``_transform``). A binding to cash's
wrapper of a function is now a binding to that function; one the user puts
in still counts as a change.

Also pinned: an outer function that appends to a list it is given, which
missed on every call while its binding flipped ("code picked by a run-time
name changed"), is now served, and a hit leaves the list as plain Python
does, in this process and the next.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

_HELPERS = """
import os


def has_dir(p):
    return os.path.isdir(p)


def inner_helper(x):
    return x * 2 + int(has_dir("/no-such-dir-inner"))


def outer_helper(x):
    return x + int(has_dir("/no-such-dir-outer"))
"""

_PRELUDE = """
import json, os, sys, warnings
warnings.simplefilter("ignore")
import cash
from cash.analysis import purity_analyzer
import bfhelpers

WALKS = []
_run = purity_analyzer.HelperWalk.run


def counted_run(self):
    WALKS.append(1)
    return _run(self)


purity_analyzer.HelperWalk.run = counted_run
c = cash.Cash(register_magic=False)


@c.cache
def inner(x):
    return bfhelpers.inner_helper(x)
"""


def _run(tmp_path, body: str) -> dict:
    (tmp_path / "bfhelpers.py").write_text(textwrap.dedent(_HELPERS), encoding="utf-8")
    script = tmp_path / "main.py"
    script.write_text(textwrap.dedent(_PRELUDE) + textwrap.dedent(body), encoding="utf-8")
    out = run_python(script.name, cwd=tmp_path).stdout
    return json.loads(out.strip().splitlines()[-1])


_WALKS = """
@c.cache
def outer(x):
    return bfhelpers.outer_helper(x) + inner(x)


def walks(x):
    n = len(WALKS)
    r = outer(x)
    return r, len(WALKS) - n


first = walks(1)
missed = [walks(x) for x in (2, 3, 4)]
warm = [walks(x) for x in (1, 2, 3)]
info = outer.cache_info()
print(json.dumps({"first": first, "missed": missed, "warm": warm, "hits": info["hits"]}))
"""


def test_outer_calls_after_the_first_walk_no_helper(tmp_path):
    got = _run(tmp_path, _WALKS)
    assert got["first"][1] > 0
    assert got["missed"] == [[6, 0], [9, 0], [12, 0]]
    assert got["warm"] == [[3, 0], [6, 0], [9, 0]]
    assert got["hits"] == 3


_REBOUND = """
@c.cache
def outer(x):
    return bfhelpers.outer_helper(x) + inner(x)


outer(1)
outer(1)
real = os.path.isdir
how = sys.argv[1] if len(sys.argv) > 1 else "os"


def users_isdir(p):
    return True


if how == "os":
    os.path.isdir = users_isdir  # every module's os.path.isdir
else:
    bfhelpers.os = type(sys)("fake_os")  # this module's own `os`
    bfhelpers.os.path = type(sys)("fake_os.path")
    bfhelpers.os.path.isdir = users_isdir
n = len(WALKS)
before = outer.cache_info()["hits"]
r = outer(1)
print(json.dumps({"result": r, "walks": len(WALKS) - n, "hit": outer.cache_info()["hits"] > before}))
"""


@pytest.mark.parametrize("how", ["os", "module"])
def test_a_user_rebinding_the_global_still_misses(tmp_path, how):
    (tmp_path / "bfhelpers.py").write_text(textwrap.dedent(_HELPERS), encoding="utf-8")
    script = tmp_path / "main.py"
    script.write_text(textwrap.dedent(_PRELUDE) + textwrap.dedent(_REBOUND), encoding="utf-8")
    out = run_python(script.name, how, cwd=tmp_path).stdout
    got = json.loads(out.strip().splitlines()[-1])
    # Both helpers now see a directory: 1 + 1 + (2 + 1), run, not served.
    assert got == {"result": 5, "walks": got["walks"], "hit": False}
    assert got["walks"] > 0


_APPENDS = """
def appends_plain(x, acc):
    acc.append(bfhelpers.outer_helper(x))
    acc.append(x)
    return inner(x)


appends = c.cache(appends_plain)
calls = [1, 2, 1, 1, 2]
acc, holder = [], {}
holder["same"] = acc
got = [appends(x, acc) for x in calls]
plain = []
want = [appends_plain(x, plain) for x in calls]
fresh = [appends(1, []), appends(1, [])]
one = []
appends(1, one)
info = appends.cache_info()
print(json.dumps({
    "same": [got, acc] == [want, plain],
    "acc": acc,
    "alias": holder["same"] is acc,
    "fresh": fresh,
    "one": one,
    "hits": info["hits"],
    "misses": info["misses"],
}))
"""


def test_an_outer_call_appending_to_its_argument_replays_as_plain_python(tmp_path):
    first = _run(tmp_path, _APPENDS)
    second = _run(tmp_path, _APPENDS)
    for got in (first, second):
        assert got["same"], got
        assert got["acc"] == [1, 1, 2, 2, 1, 1, 1, 1, 2, 2]
        assert got["alias"]
        assert got["fresh"] == [2, 2]
        assert got["one"] == [1, 1]
    assert first["hits"] > 0
    # A new process serves every call, and writes the list back as it ran.
    assert second == {**first, "hits": 8, "misses": 0}
