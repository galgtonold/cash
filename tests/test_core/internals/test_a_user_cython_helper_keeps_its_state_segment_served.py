"""A cached function that calls a Cython function of the user's own build
has its state segment served by the state memo (`cash.decorator.state_memo`).

The memo's scan refused such a world (``Ineligible: a
_cython_3_x.cython_function_or_method``), so the globals were folded again
on every call, hits included: 2.5 ms a hit for hdbscan's
``_hdbscan_prims_kdtree`` built as user code. The key names the function by
the Cython code it reaches in its ``.pyx`` (or its whole built file),
worked out once per loaded module; the scan now takes it with a probe that
reads again, before every reuse, what that key read of it -- its module,
the module's file, its names, code and attributes, and the size and times of
every file its key was worked out from (`hit_memo._cython_reads`). An edit
to the ``.pyx`` breaks the memo and the segment is folded again, to what
folding without the memo gives.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import time

import pytest

from cash.decorator import state_memo
from tests._scripts import run_python

pytestmark = [
    pytest.mark.timeout(600),
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension's world watch"),
]

PYX = """\
FACTOR = 3

cdef int h(int x):
    return x * FACTOR

def f(int x):
    return h(x) + 1

def g(int x):
    return x - 1
"""

SETUP = """\
from setuptools import setup
from Cython.Build import cythonize
setup(name="fastops", ext_modules=cythonize("fastops.pyx", language_level=3, quiet=True))
"""

JOB = """\
import json, os
import cash
from cash.decorator import state_memo
from fastops import f

c = cash.Cash(register_magic=False)


@c.cache
def run(x):
    return f(x)


keys = []


def calls(xs):
    for x in xs:
        run(x)
        keys.append(run.explain(x).cache_key)


calls(range(6))
out = {"reused_before_edit": state_memo.STATS["reused"]}
if os.environ.get("EDIT"):
    st = os.stat("fastops.pyx")
    text = open("fastops.pyx", encoding="utf-8").read()
    scans = state_memo.STATS["scans"]
    try:
        with open("fastops.pyx", "w", encoding="utf-8") as fh:
            fh.write(text.replace("return x - 1", "return x - 2"))
        os.utime("fastops.pyx", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        calls(range(6, 10))
        out["scans_after_edit"] = state_memo.STATS["scans"] - scans
    finally:
        with open("fastops.pyx", "w", encoding="utf-8") as fh:
            fh.write(text)
        os.utime("fastops.pyx", ns=(st.st_atime_ns, st.st_mtime_ns))
out.update(state_memo.STATS)
out["keys"] = keys
print(json.dumps(out))
"""


class BuildFailed(Exception):
    """The extension did not build; carries the compiler's output."""


def _build(proj):
    """Build ``fastops`` in place, then date every source before the binary,
    as a build leaves them."""
    temp = str(proj.parent / (proj.name + "-build"))
    p = subprocess.run(
        [sys.executable, "setup.py", "-q", "build_ext", "--inplace", "--force", "--build-temp", temp],
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=300,
    )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if p.returncode != 0 or not (proj / f"fastops{suffix}").exists():
        raise BuildFailed((p.stdout + p.stderr)[-3000:])
    now = time.time()
    for path in proj.iterdir():
        past = 30 if path.name.endswith(suffix) else 60
        os.utime(path, (now - past, now - past))


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """``fastops`` built once; skipped when Cython or a C compiler is missing."""
    pytest.importorskip("setuptools")
    pytest.importorskip("Cython")
    proj = tmp_path_factory.mktemp("cy") / "proj"
    proj.mkdir()
    (proj / "fastops.pyx").write_text(PYX, encoding="utf-8")
    (proj / "setup.py").write_text(SETUP, encoding="utf-8")
    (proj / "job.py").write_text(JOB, encoding="utf-8")
    try:
        _build(proj)
    except BuildFailed as e:
        last = [line for line in str(e).splitlines() if line.strip()][-1:] or ["no output"]
        pytest.skip(f"no working C compiler builds a Cython extension here: {last[0][:300]}")
    return proj


def _job(proj, tmp_path, **env):
    p = run_python("job.py", cwd=proj, cache_dir=str(tmp_path / ".cash"), env=env)
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_the_state_segment_is_served(project, tmp_path):
    got = _job(project, tmp_path, CASH_STATE_MEMO_VERIFY="1")
    assert got["scans_failed"] == 0, "the scan refused the Cython function"
    assert got["reused"] > 0, "the segment was folded again on every call"
    assert got["verified"] == got["reused"]


def test_an_edit_to_the_pyx_folds_again_to_what_no_memo_gives(project, tmp_path):
    served = _job(project, tmp_path / "on", EDIT="1", CASH_STATE_MEMO_VERIFY="1")
    plain = _job(project, tmp_path / "off", EDIT="1", CASH_STATE_MEMO="0")
    assert served["reused_before_edit"] > 0
    assert served["scans_after_edit"] >= 1, "an edit to the .pyx left the remembered segment standing"
    assert served["keys"] == plain["keys"]
