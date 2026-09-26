"""A helper from a compiled extension in the project is keyed by the built file.

``from fastops import scale`` with ``fastops`` built in place (``build_ext
--inplace``, an editable install of a C or Cython package) is a builtin
function: no source, no bytecode, so it was keyed by its name, and a rebuild
that changed what it computes was served the old result. The extension
file's content now stands for its code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import textwrap
import time

import pytest

pytestmark = [pytest.mark.timeout(300)]

C_SOURCE = r"""
#include <Python.h>
static PyObject* scale(PyObject* s, PyObject* a) {
    long x;
    if (!PyArg_ParseTuple(a, "l", &x)) return NULL;
    return PyLong_FromLong(x * FACTOR);
}
static PyMethodDef M[] = {{"scale", scale, METH_VARARGS, ""}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "fastops", NULL, -1, M};
PyMODINIT_FUNC PyInit_fastops(void) { return PyModule_Create(&mod); }
"""

JOB = """
import sys
import cash
from fastops import scale

@cash.cache
def f(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return scale(x)

print(f(2))
"""


def _build(proj, factor):
    """Build ``fastops`` in place in *proj*, as ``build_ext --inplace`` does:
    setuptools picks the compiler and the flags the interpreter needs (MSVC on
    Windows; on macOS a bundle that leaves the Python symbols to load time)."""
    script = (
        "from setuptools import Extension, setup\n"
        "setup(name='fastops', script_args=['-q', 'build_ext', '--inplace', '--force', '--build-temp', "
        f"{str(proj.parent / (proj.name + '-build'))!r}], ext_modules=[Extension('fastops', ['fastops.c'], "
        f"define_macros=[('FACTOR', '{factor}')])])\n"
    )
    p = subprocess.run([sys.executable, "-c", script], cwd=str(proj), capture_output=True, text=True, timeout=240)
    out = proj / f"fastops{sysconfig.get_config_var('EXT_SUFFIX')}"
    if p.returncode != 0 or not out.exists():
        raise BuildFailed((p.stdout + p.stderr)[-3000:])
    past = time.time() - 30
    os.utime(out, (past, past))


class BuildFailed(Exception):
    """The extension did not build; carries the compiler's output."""


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.strip(), "[RUN]" in p.stderr


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
    """Skip when no C compiler here can build a Python extension at all,
    found by building the module once; a build that fails after this is a
    failure, not a skip."""
    pytest.importorskip("setuptools")
    proj = tmp_path_factory.mktemp("probe") / "proj"
    proj.mkdir()
    (proj / "fastops.c").write_text(C_SOURCE, encoding="utf-8")
    try:
        _build(proj, 1)
    except BuildFailed as e:
        last = [line for line in str(e).splitlines() if line.strip()][-1:] or ["no output"]
        pytest.skip(f"no working C compiler builds a Python extension here: {last[0][:300]}")


def test_rebuilding_the_extension_recomputes(compiler, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "fastops.c").write_text(C_SOURCE, encoding="utf-8")
    (proj / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    _build(proj, 10)
    assert _run(proj) == ("20", True)
    assert _run(proj) == ("20", False), "the unedited run did not hit"
    _build(proj, 11)
    assert _run(proj) == ("22", True), "a rebuilt extension helper was served the old result"


def test_depending_on_it_is_not_inert(compiler, tmp_path, monkeypatch):
    """KEY-DEPENDS-ON-OPAQUE said the declaration did nothing; it keys the build now."""
    import importlib
    import warnings

    from cash import Cash

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "fastops.c").write_text(C_SOURCE, encoding="utf-8")
    _build(proj, 10)
    monkeypatch.syspath_prepend(str(proj))
    monkeypatch.delitem(sys.modules, "fastops", raising=False)
    fastops = importlib.import_module("fastops")

    def f(x):
        return x

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Cash(cache_dir=str(tmp_path / "cache")).cache(f, depends_on=[fastops.scale])
    assert not [w for w in rec if "KEY-DEPENDS-ON-OPAQUE" in str(w.message)]
