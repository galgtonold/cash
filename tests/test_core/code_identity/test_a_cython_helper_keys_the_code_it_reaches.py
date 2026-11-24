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
    # The first process analysed before the cache folder existed; what it
    # worked out is written once the folder is there.
    assert _run_compiler(proj) == ("7", False, False), "a warm process parsed the .pyx again"
    _edit(proj, "return x - 1", "return x - 2")
    assert _run_compiler(proj) == ("7", False, True), "an edit to a function f never reaches re-ran it"
    assert _run_compiler(proj) == ("7", False, False)
    _edit(proj, "return h(x) + 1", "return h(x) + 2")
    assert _run_compiler(proj)[:2] == ("8", True), "an edit to f was served the old result"
    assert _run_compiler(proj) == ("8", False, False)


def test_a_script_in_another_folder_uses_the_stored_analysis(cython_compiler, tmp_path):
    """``sys.path[0]`` is the script's folder: a driver started from another
    folder (or with another ``PYTHONPATH``) has another import path. Each
    cimport is looked up again under it rather than the module parsed
    again, and an edit is still seen from either folder."""
    proj = _project(tmp_path, files={"job.py": JOB_COMPILER})
    (proj / "job.py").write_text(JOB_COMPILER, encoding="utf-8")
    other = proj / "drivers"
    other.mkdir()
    (other / "job.py").write_text(JOB_COMPILER, encoding="utf-8")

    def elsewhere():
        p = run_python(
            "drivers/job.py", cwd=proj, env={"PYTHONPATH": f"{proj}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        )
        return p.stdout.strip(), "[RUN]" in p.stderr, "[COMPILER]" in p.stderr

    assert _run_compiler(proj)[:2] == ("7", True)
    assert elsewhere()[::2] == ("7", False), "another import path parsed the .pyx again"
    assert _run_compiler(proj)[::2] == ("7", False)
    assert elsewhere()[::2] == ("7", False)
    _edit(proj, "return h(x) + 1", "return h(x) + 2")
    assert elsewhere()[:2] == ("8", True), "an edit to f was served the old result"
    assert _run_compiler(proj) == ("8", False, False)


def test_a_pyx_edited_and_not_rebuilt_is_read_once(cython_compiler, tmp_path):
    """A ``.pyx`` newer than its binary keys the whole binary and the
    modules it cimports, which reads its cimports with Cython's scanner.
    That is stored too: the next process neither re-runs nor imports the
    compiler, and the rebuild is still seen."""
    proj = _project(tmp_path, files={"job.py": JOB_COMPILER})
    (proj / "job.py").write_text(JOB_COMPILER, encoding="utf-8")
    assert _run_compiler(proj)[:2] == ("7", True)
    pyx = proj / "fastops.pyx"
    pyx.write_text(PYX.replace("return x - 1", "return x - 2"), encoding="utf-8")
    assert _run_compiler(proj)[:2] == ("7", True)
    assert _run_compiler(proj) == ("7", False, False), "a newer .pyx was read with Cython in every process"
    pyx.write_text(PYX.replace("return h(x) + 1", "return h(x) + 2"), encoding="utf-8")
    _build(proj)
    assert _run_compiler(proj)[:2] == ("8", True), "the rebuilt f was served the old result"


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


# -- CRLF sources --------------------------------------------------------------


def _crlf(text):
    return text.replace("\n", "\r\n").encode("utf-8")


def _project_crlf(tmp_path, pyx=PYX, setup=SETUP):
    """`_project` with every source written with CRLF line endings, as a
    Windows checkout leaves them."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "fastops.pyx").write_bytes(_crlf(pyx))
    (proj / "setup.py").write_bytes(_crlf(setup))
    (proj / "job.py").write_bytes(_crlf(JOB_COMPILER))
    _build(proj)
    return proj


def _edit_crlf(proj, old, new, name="fastops.pyx"):
    path = proj / name
    data = path.read_bytes()
    assert _crlf(old) in data
    path.write_bytes(data.replace(_crlf(old), _crlf(new)))
    _build(proj)


def test_a_crlf_pyx_keeps_the_per_function_key(cython_compiler, tmp_path):
    """Cython's scanner refuses a ``\\r``, so a CRLF ``.pyx`` fell back to
    the whole binary in every process: an edit to another function re-ran
    ``f``. Line endings are made ``\\n`` first, as Cython's reader does."""
    proj = _project_crlf(tmp_path)
    assert _run_compiler(proj)[:2] == ("7", True)
    _run_compiler(proj)
    assert _run_compiler(proj) == ("7", False, False), "a warm process parsed the CRLF .pyx again"
    _edit_crlf(proj, "return x - 1", "return x - 2")
    assert _run_compiler(proj)[:2] == ("7", False), "an edit to a function f never reaches re-ran it"
    _edit_crlf(proj, "return h(x) + 1", "return h(x) + 2")
    assert _run_compiler(proj)[:2] == ("8", True), "an edit to f was served the old result"


def test_lf_and_crlf_of_the_same_source_key_alike(cython_compiler, tmp_path):
    """Only the line endings differ, so the key does not move: a CRLF
    checkout of a project hits what an LF one stored."""
    proj = _project(tmp_path)
    assert _run(proj) == ("7", True)
    for name in ("fastops.pyx", "setup.py"):
        path = proj / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    _build(proj)
    assert _run(proj) == ("7", False), "the CRLF form of the same source moved the key"


# -- the build configuration's metadata ------------------------------------------

SETUP_MACRO = """\
from setuptools import Extension, setup
from Cython.Build import cythonize

LEVEL = "1"
setup(
    name="fastops",
    version="0.8.40",
    description="fast ops",
    ext_modules=cythonize(
        [Extension("fastops", ["fastops.pyx"], define_macros=[("LEVEL", LEVEL)])],
        language_level=3,
        quiet=True,
    ),
)
"""


def test_a_version_bump_hits_and_a_macro_change_misses(cython_compiler, tmp_path):
    """The build configuration is keyed without its metadata: a version
    bump in ``setup()`` re-ran every Cython helper of the package. What can
    change the compiled result -- a macro here -- still misses."""
    proj = _project(tmp_path, setup=SETUP_MACRO)
    assert _run(proj) == ("7", True)
    assert _run(proj) == ("7", False)
    _edit(
        proj,
        'version="0.8.40",\n    description="fast ops"',
        'version="0.8.41",\n    description="faster ops"',
        "setup.py",
    )
    assert _run(proj) == ("7", False), "a version-only bump re-ran the Cython helper"
    _edit(proj, 'LEVEL = "1"', 'LEVEL = "2"', "setup.py")
    assert _run(proj) == ("7", True), "a define_macros change was served the old result"


def test_a_version_shared_with_a_macro_still_misses(cython_compiler, tmp_path):
    """Only the ``setup()`` keyword is dropped: a version also passed to
    ``define_macros`` is keyed there."""
    setup = SETUP_MACRO.replace('LEVEL = "1"', 'LEVEL = "0.8.40"').replace('version="0.8.40"', "version=LEVEL")
    proj = _project(tmp_path, setup=setup)
    assert _run(proj) == ("7", True)
    _edit(proj, 'LEVEL = "0.8.40"', 'LEVEL = "0.8.41"', "setup.py")
    assert _run(proj) == ("7", True), "a version that is also a macro was served the old result"


def _canonical(name, text):
    from cash.cython_digest import canonical_build_config

    return canonical_build_config(name, text.encode("utf-8"))


def test_the_canonical_pyproject_drops_only_metadata():
    base = '[project]\nname = "p"\nversion = "1"\ndescription = "d"\ndependencies = ["numpy"]\n'
    base += '[build-system]\nrequires = ["cython"]\n[tool.x]\nflag = 1\n'
    same = base.replace('version = "1"', 'version = "2"').replace('"d"', '"e"') + "# a comment\n"
    assert _canonical("pyproject.toml", base) == _canonical("pyproject.toml", same)
    for old, new in (('"numpy"', '"scipy"'), ('"cython"', '"cython<3"'), ("flag = 1", "flag = 2"), ('"p"', '"q"')):
        assert _canonical("pyproject.toml", base) != _canonical("pyproject.toml", base.replace(old, new)), new
    assert _canonical("pyproject.toml", "[[broken") == b"[[broken"


def test_the_canonical_setup_cfg_drops_only_metadata():
    base = "[metadata]\nname = p\nversion = 1\nlong_description = x\n[options]\nzip_safe = False\n"
    same = base.replace("version = 1", "version = 2").replace("= x", "= y")
    assert _canonical("setup.cfg", base) == _canonical("setup.cfg", same)
    assert _canonical("setup.cfg", base) != _canonical("setup.cfg", base.replace("zip_safe = False", "zip_safe = True"))
    assert _canonical("setup.cfg", base) != _canonical("setup.cfg", base.replace("name = p", "name = q"))


def test_the_canonical_setup_py_drops_only_setup_metadata_keywords():
    base = 'from setuptools import setup\nV = "1"\nsetup(name="p", version="1", ext_modules=m(V))\n'
    same = base.replace('version="1"', 'version="2"').replace("setup(", "# c\nsetup(\n    ")
    assert _canonical("setup.py", base) == _canonical("setup.py", same)
    assert _canonical("setup.py", base) != _canonical("setup.py", base.replace('V = "1"', 'V = "2"'))
    other = base.replace("setup(name", "other(name")
    assert _canonical("setup.py", other) != _canonical("setup.py", other.replace('version="1"', 'version="2"'))
    assert _canonical("setup.py", "def (") == b"def ("


def test_a_failed_analysis_is_stored_and_an_edit_invalidates_it(tmp_path, monkeypatch):
    """An analysis that fails (not a deliberate fallback) is an answer of
    the files it read, stored so each process does not pay for it again;
    an edit to one of them makes it stale."""
    from cash import cython_digest

    pyx = tmp_path / "m.pyx"
    pyx.write_bytes(b"def f():\n    pass\n")

    def failing(module, binary, co_filename, pyx_path=None):
        cython_digest._read(str(pyx))
        raise ValueError("parse failure")

    monkeypatch.setattr(cython_digest, "_analyse", failing)
    analysis, inputs = cython_digest._analyse_recorded(None, "", "", None)
    assert analysis is None
    stored = {"files": sorted(inputs.files.items()), "probes": []}
    assert cython_digest._inputs_unchanged(stored)
    pyx.write_bytes(b"def f():\n    return 1\n")
    assert not cython_digest._inputs_unchanged(stored)


def test_a_config_recorded_by_its_canonical_form_survives_a_version_bump(tmp_path):
    from cash import cython_digest

    setup = tmp_path / "setup.py"
    setup.write_text('from setuptools import setup\nsetup(name="p", version="1")\n', encoding="utf-8")
    inputs = cython_digest._Inputs()
    cython_digest._RECORDING.append(inputs)
    try:
        cython_digest._read_config(str(setup))
    finally:
        cython_digest._RECORDING.pop()
    stored = {"files": sorted(inputs.files.items()), "probes": []}
    setup.write_text('from setuptools import setup\nsetup(name="p", version="2")\n', encoding="utf-8")
    assert cython_digest._inputs_unchanged(stored)
    setup.write_text('from setuptools import setup\nsetup(name="q", version="2")\n', encoding="utf-8")
    assert not cython_digest._inputs_unchanged(stored)


def test_a_config_bumped_and_not_rebuilt_keeps_the_per_function_key(cython_compiler, tmp_path):
    """An incremental build does not rebuild a module whose sources did not
    change, so a version bump leaves ``setup.py`` newer than the binary.
    That fell back to the whole-binary key: every helper re-ran on the
    bump, and again when the next rebuild brought the per-function key
    back. The binary was seen with every file older than it, and the
    configuration's canonical form is what it was then: the key stays."""
    proj = _project(tmp_path, setup=SETUP_MACRO)
    assert _run(proj) == ("7", True)
    setup = proj / "setup.py"
    setup.write_text(SETUP_MACRO.replace('version="0.8.40"', 'version="0.8.41"'), encoding="utf-8")
    assert _run(proj) == ("7", False), "a version bump the build skipped re-ran the Cython helper"
    pyx = proj / "fastops.pyx"
    pyx.write_text(PYX, encoding="utf-8")
    assert _run(proj) == ("7", False), "a .pyx rewritten with the same text re-ran the Cython helper"
    _build(proj)
    assert _run(proj) == ("7", False), "the rebuild after a version bump re-ran the Cython helper"


def test_a_config_changed_and_not_rebuilt_still_falls_back(cython_compiler, tmp_path):
    """What the binary was not seen built from still keys the whole binary:
    a macro changed and not rebuilt, then the rebuild."""
    proj = _project(tmp_path, setup=SETUP_MACRO)
    assert _run(proj) == ("7", True)
    (proj / "setup.py").write_text(SETUP_MACRO.replace('LEVEL = "1"', 'LEVEL = "2"'), encoding="utf-8")
    assert _run(proj) == ("7", True), "a macro the binary was not built with keyed the per-function key"
    _build(proj)
    assert _run(proj) == ("7", True), "the rebuild with a new macro was served the old result"


def test_a_source_seen_with_another_binary_does_not_vouch_for_this_one(cython_compiler, tmp_path):
    """What a binary was seen built from is recorded by the binary's
    content: a ``.pyx`` put back to an older text, newer than the binary
    built since, matches the record of the OLD binary only. Keyed by that
    text, the rebuilt binary would be served the old binary's result."""
    proj = _project(tmp_path)
    assert _run(proj) == ("7", True)
    _edit(proj, "return h(x) + 1", "return h(x) + 2")
    assert _run(proj) == ("8", True)
    (proj / "fastops.pyx").write_text(PYX, encoding="utf-8")
    assert _run(proj) == ("8", True), "the rebuilt binary was served the result of the binary before it"


SETUP_DICT = """\
from setuptools import setup
V = "1"
configuration = {
    "name": "p",
    "version": "1",
    "description": "d",
    "ext_modules": m(V),
}
setup(**configuration)
"""


def test_the_canonical_setup_py_drops_metadata_of_a_dict_only_passed_to_setup():
    """hdbscan keeps its metadata in ``configuration = {...}`` and calls
    ``setup(**configuration)``: its version bump moved the key."""
    same = SETUP_DICT.replace('"version": "1"', '"version": "2"').replace('"d"', '"e"')
    assert _canonical("setup.py", SETUP_DICT) == _canonical("setup.py", same)
    assert _canonical("setup.py", SETUP_DICT) != _canonical("setup.py", SETUP_DICT.replace('V = "1"', 'V = "2"'))
    assert _canonical("setup.py", SETUP_DICT) != _canonical("setup.py", SETUP_DICT.replace('"p"', '"q"'))


@pytest.mark.parametrize(
    "extra",
    [
        'print(configuration["version"])\n',
        'configuration["define_macros"] = [("V", configuration["version"])]\n',
        "other(**configuration)\n",
        "configuration = dict(configuration)\n",
        "x = globals()\n",
        'getattr(mod, "configuration")\n',
        "def f(configuration):\n    pass\n",
        "configuration |= {}\n",
        "import __main__\n__main__.configuration\n",
    ],
)
def test_the_canonical_setup_py_keeps_a_dict_used_elsewhere(extra):
    """Any other use of the dict -- read, mutated, rebound, passed on,
    reachable by other means -- keeps every entry keyed."""
    base = SETUP_DICT + extra
    bumped = base.replace('"version": "1"', '"version": "2"')
    assert _canonical("setup.py", base) != _canonical("setup.py", bumped), extra


def test_the_canonical_setup_py_keeps_a_metadata_value_that_calls():
    """A metadata entry whose value calls something may do more than name
    the version: it stays keyed."""
    base = SETUP_DICT.replace('"version": "1"', '"version": version("1")')
    bumped = base.replace('version("1")', 'version("2")')
    assert _canonical("setup.py", base) != _canonical("setup.py", bumped)
