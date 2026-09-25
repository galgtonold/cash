"""A helper from a compiled extension in the project is keyed by the built file.

``from fastops import scale`` with ``fastops`` built in place (``build_ext
--inplace``, an editable install of a C or Cython package) is a builtin
function: no source, no bytecode, so it was keyed by its name, and a rebuild
that changed what it computes was served the old result. The extension
file's content now stands for its code.
"""

from __future__ import annotations

import os
import shutil
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
    include = sysconfig.get_paths()["include"]
    out = proj / f"fastops{sysconfig.get_config_var('EXT_SUFFIX')}"
    subprocess.run(
        ["gcc", "-shared", "-fPIC", f"-DFACTOR={factor}", "-I", include, str(proj / "fastops.c"), "-o", str(out)],
        check=True,
        capture_output=True,
    )
    past = time.time() - 30
    os.utime(out, (past, past))


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.strip(), "[RUN]" in p.stderr


needs_compiler = pytest.mark.skipif(
    shutil.which("gcc") is None or not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")),
    reason="needs gcc and the Python headers",
)


@needs_compiler
def test_rebuilding_the_extension_recomputes(tmp_path):
    (tmp_path / "fastops.c").write_text(C_SOURCE, encoding="utf-8")
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    _build(tmp_path, 10)
    assert _run(tmp_path) == ("20", True)
    assert _run(tmp_path) == ("20", False), "the unedited run did not hit"
    _build(tmp_path, 11)
    assert _run(tmp_path) == ("22", True), "a rebuilt extension helper was served the old result"


@needs_compiler
def test_depending_on_it_is_not_inert(tmp_path, monkeypatch):
    """KEY-DEPENDS-ON-OPAQUE said the declaration did nothing; it keys the build now."""
    import importlib
    import warnings

    from cash import Cash

    (tmp_path / "fastops.c").write_text(C_SOURCE, encoding="utf-8")
    _build(tmp_path, 10)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fastops", raising=False)
    fastops = importlib.import_module("fastops")

    def f(x):
        return x

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Cash(cache_dir=str(tmp_path / "cache")).cache(f, depends_on=[fastops.scale])
    assert not [w for w in rec if "KEY-DEPENDS-ON-OPAQUE" in str(w.message)]
