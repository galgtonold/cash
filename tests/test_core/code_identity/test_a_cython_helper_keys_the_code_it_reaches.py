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
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if p.returncode != 0 or not (proj / f"fastops{suffix}").exists():
        raise BuildFailed((p.stdout + p.stderr)[-3000:])
    now = time.time()
    for path in proj.iterdir():
        past = 30 if path.name.endswith(suffix) else 60
        os.utime(path, (now - past, now - past))


def _run(proj):
    p = run_python("job.py", cwd=proj)
    return p.stdout.strip(), "[RUN]" in p.stderr


def _project(tmp_path, pyx=PYX, pxd=None, setup=SETUP, files=None):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "fastops.pyx").write_text(pyx, encoding="utf-8")
    if pxd is not None:
        (proj / "fastops.pxd").write_text(pxd, encoding="utf-8")
    for name, text in (files or {}).items():
        (proj / name).write_text(text, encoding="utf-8")
    (proj / "setup.py").write_text(setup, encoding="utf-8")
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


#: ``fastops`` calls a cdef function of a second module, ``kernels``,
#: through ``kernels.pxd``: the call goes into the other binary.
KERNELS_PXD = "cdef int triple(int x)\n"

KERNELS_PYX = """\
cdef int triple(int x):
    return x * 3

def unrelated(int x):
    return x + 100
"""

CALLER_PYX = """\
from kernels cimport triple

def f(int x):
    return triple(x) + 1
"""

TWO_MODULES_SETUP = SETUP.replace('cythonize("fastops.pyx"', 'cythonize(["fastops.pyx", "kernels.pyx"]')


def _two_modules(tmp_path, pyx=CALLER_PYX):
    return _project(
        tmp_path,
        pyx=pyx,
        setup=TWO_MODULES_SETUP,
        files={"kernels.pxd": KERNELS_PXD, "kernels.pyx": KERNELS_PYX},
    )


def test_an_edit_to_a_cimported_module_s_function_misses(cython_compiler, tmp_path):
    proj = _two_modules(tmp_path)
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False), "the unedited run did not hit"
    _edit(proj, "return x + 100", "return x + 200", name="kernels.pyx")
    assert _run(proj) == ("7", False), "an edit to a function of the cimported module f never reaches re-ran it"
    _edit(proj, "return x * 3", "return x * 4", name="kernels.pyx")
    assert _run(proj) == ("9", True), "an edit to the cimported cdef function f calls was served the old result"


def test_an_edit_to_a_cimported_module_misses_on_the_whole_binary_key(cython_compiler, tmp_path):
    pyx = CALLER_PYX.replace("\ndef f", '\ncdef extern from "math.h":\n    double fabs(double)\n\ndef f')
    proj = _two_modules(tmp_path, pyx=pyx)
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False), "the unedited run did not hit"
    _edit(proj, "return x * 3", "return x * 4", name="kernels.pyx")
    assert _run(proj) == ("9", True), "an edit to the cimported cdef function f calls was served the old result"


def test_numpy_macros_and_headers_keep_the_per_function_key(cython_compiler, tmp_path):
    """hdbscan's shape: every module sets numpy's API macro in a
    ``# distutils:`` comment and declares a numpy header ``cdef extern``.
    Both are keyed (the comment as text, the header by its content), so an
    edit to another function still hits and a macro change misses."""
    numpy = pytest.importorskip("numpy")
    pyx = (
        "# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION\n"
        f"# distutils: include_dirs={numpy.get_include()}\n"
        + PYX
        + '\ncdef extern from "numpy/arrayobject.h":\n    ctypedef int npy_intp\n'
    )
    proj = _project(tmp_path, pyx=pyx)
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False), "the unedited run did not hit"
    _edit(proj, "return x - 1", "return x - 2")
    assert _run(proj) == ("7", False), "an edit to a function f never reaches re-ran it"
    _edit(proj, "NPY_1_7_API_VERSION", "NPY_1_8_API_VERSION")
    assert _run(proj) == ("7", True), "a changed macro did not re-run"


#: `JOB`, also saying whether the process imported Cython's compiler.
JOB_COMPILER = (
    JOB + 'print("[COMPILER]" if "Cython.Compiler.Main" in sys.modules else "[NO-COMPILER]", file=sys.stderr)\n'
)


def _run_compiler(proj):
    p = run_python("job.py", cwd=proj)
    return p.stdout.strip(), "[RUN]" in p.stderr, "[COMPILER]" in p.stderr


def test_a_warm_process_uses_the_stored_analysis_and_still_misses_after_an_edit(cython_compiler, tmp_path):
    """Parsing the ``.pyx`` with Cython cost every fresh process about a
    second per module. The analysis is stored with the content digest of
    every file it read, so a warm process never imports the compiler, and
    an edit to ``f`` is still seen."""
    proj = _project(tmp_path, files={"job.py": JOB_COMPILER})
    (proj / "job.py").write_text(JOB_COMPILER, encoding="utf-8")
    assert _run_compiler(proj)[:2] == ("7", True)
    _run_compiler(proj)  # the cache folder exists now, so the analysis is stored
    assert _run_compiler(proj) == ("7", False, False), "a warm process parsed the .pyx again"
    _edit(proj, "return x - 1", "return x - 2")
    assert _run_compiler(proj) == ("7", False, True), "an edit to a function f never reaches re-ran it"
    assert _run_compiler(proj) == ("7", False, False)
    _edit(proj, "return h(x) + 1", "return h(x) + 2")
    assert _run_compiler(proj)[:2] == ("8", True), "an edit to f was served the old result"
    assert _run_compiler(proj) == ("8", False, False)


def test_a_stored_analysis_is_not_used_when_a_pxd_appears(cython_compiler, tmp_path):
    """A stored analysis also records which files were NOT there. A
    ``.pxd`` that appears beside the ``.pyx``, dated before the binary and
    nothing rebuilt, is keyed as a fresh analysis keys it: the stored one,
    made without it, is not used."""
    proj = _project(tmp_path)
    assert _run(proj) == ("7", True)
    _run(proj)
    assert _run(proj) == ("7", False)
    pxd = proj / "fastops.pxd"
    pxd.write_text("cdef int h(int x)\n", encoding="utf-8")
    past = time.time() - 60
    os.utime(pxd, (past, past))
    assert _run(proj) == ("7", True), "a new .pxd was not keyed"
    assert _run(proj) == ("7", False)
