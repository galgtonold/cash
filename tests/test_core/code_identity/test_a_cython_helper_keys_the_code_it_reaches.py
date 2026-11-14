"""A Cython helper is keyed by the Cython code it can reach, not the whole binary.

``from fastops import f`` with ``fastops`` a Cython module built in place
was keyed by the digest of the built file, so an edit to ANY function of
the ``.pyx`` re-ran every cached function that called any of them. It is
keyed by what ``f`` reaches now: its own block, the blocks it names
(transitively), the module header and directives, the ``.pxd`` files and
the build configuration. Where that cannot be trusted (``cdef extern``, a
``.pyx`` newer than the build) the whole built file still stands for it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import textwrap
import time

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(600)]

PYX = """\
# cython: boundscheck=False
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
import sys
import cash
from fastops import f

@cash.cache
def run(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return f(x)

print(run(2))
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
    out = proj / f"fastops{sysconfig.get_config_var('EXT_SUFFIX')}"
    if p.returncode != 0 or not out.exists():
        raise BuildFailed((p.stdout + p.stderr)[-3000:])
    now = time.time()
    for path in proj.iterdir():
        if path != out:
            os.utime(path, (now - 60, now - 60))
    os.utime(out, (now - 30, now - 30))


def _run(proj):
    p = run_python("job.py", cwd=proj)
    return p.stdout.strip(), "[RUN]" in p.stderr


def _project(tmp_path, pyx=PYX, pxd=None):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "fastops.pyx").write_text(pyx, encoding="utf-8")
    if pxd is not None:
        (proj / "fastops.pxd").write_text(pxd, encoding="utf-8")
    (proj / "setup.py").write_text(SETUP, encoding="utf-8")
    (proj / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    _build(proj)
    return proj


def _edit(proj, old, new, name="fastops.pyx"):
    path = proj / name
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    _build(proj)


@pytest.fixture(scope="module")
def cython_compiler(tmp_path_factory):
    """Skip when Cython is missing or no C compiler here builds an extension,
    found by building the module once; a build that fails after this is a
    failure, not a skip."""
    pytest.importorskip("setuptools")
    pytest.importorskip("Cython")
    try:
        _project(tmp_path_factory.mktemp("probe"))
    except BuildFailed as e:
        last = [line for line in str(e).splitlines() if line.strip()][-1:] or ["no output"]
        pytest.skip(f"no working C compiler builds a Cython extension here: {last[0][:300]}")


def test_an_edit_to_another_function_hits_and_an_edit_to_the_called_one_misses(cython_compiler, tmp_path):
    proj = _project(tmp_path)
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False), "the unedited run did not hit"
    _edit(proj, "return x - 1", "return x - 2")
    assert _run(proj) == ("7", False), "an edit to a function f never reaches re-ran it"
    _edit(proj, "return h(x) + 1", "return h(x) + 2")
    assert _run(proj) == ("8", True), "an edit to f itself was served the old result"


def test_an_edit_to_a_helper_it_calls_misses(cython_compiler, tmp_path):
    proj = _project(tmp_path)
    assert _run(proj) == ("7", True)
    _edit(proj, "return x * FACTOR", "return x * FACTOR * 2")
    assert _run(proj) == ("13", True), "an edit to the cdef helper f calls was served the old result"


def test_an_edit_to_the_module_header_misses(cython_compiler, tmp_path):
    proj = _project(tmp_path)
    assert _run(proj) == ("7", True)
    _edit(proj, "FACTOR = 3", "FACTOR = 4")
    assert _run(proj) == ("9", True), "an edit to a module constant was served the old result"
    _edit(proj, "# cython: boundscheck=False", "# cython: boundscheck=True")
    assert _run(proj) == ("9", True), "a compiler directive change did not re-run"


def test_an_edit_to_the_pxd_misses(cython_compiler, tmp_path):
    proj = _project(tmp_path, pxd="cdef int h(int x)\n")
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False)
    _edit(proj, "cdef int h(int x)\n", "ctypedef int counter\ncdef int h(int x)\n", name="fastops.pxd")
    assert _run(proj) == ("7", True), "an edit to the module's .pxd did not re-run"


def test_cdef_extern_keys_the_whole_binary(cython_compiler, tmp_path):
    pyx = PYX.replace("FACTOR = 3\n", 'FACTOR = 3\n\ncdef extern from "math.h":\n    double fabs(double)\n')
    proj = _project(tmp_path, pyx=pyx)
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False)
    _edit(proj, "return x - 1", "return x - 2")
    assert _run(proj) == ("7", True), "with cdef extern an edit anywhere in the module must re-run"


def test_a_pyx_newer_than_the_binary_keys_the_whole_binary(cython_compiler, tmp_path):
    proj = _project(tmp_path)
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False)
    # Edited after the build and not rebuilt: what the .pyx says is not what
    # runs, so the built file is keyed (a different key, so one re-run).
    pyx = proj / "fastops.pyx"
    pyx.write_text(PYX.replace("return x - 1", "return x - 2"), encoding="utf-8")
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False), "the whole-binary key is stable"
    _build(proj)
    pyx.write_text(PYX.replace("return x - 1", "return x - 3"), encoding="utf-8")
    assert _run(proj) == ("7", True), "a rebuilt binary under a newer .pyx must re-run"
