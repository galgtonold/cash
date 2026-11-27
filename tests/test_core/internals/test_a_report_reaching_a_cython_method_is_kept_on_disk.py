"""A cached function whose walk reaches a function Cython compiled (a
``classmethod`` of a ``cdef class``, hdbscan's ``DistanceMetric.get_metric``)
has its purity report stored, and a fresh process reads it instead of
walking again.

A compiled function is neither a ``FunctionType`` nor a class, so no name
was taken to lead back to it, the report was left out of the store, and
the function's helpers were walked in every process. Its module holds it
by its qualified name, as it holds a plain function. Read back, the report
checks its token, its text, its built file whole and its key digest: a
rebuild after an edit to it walks again and keys anew.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import textwrap
import time
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(600)]

PYX = """\
cdef class Metric:
    cdef double k

    @classmethod
    def get(cls, double x):
        return x * 3

def shift(double x):
    return x + 1
"""

SETUP = """\
from setuptools import setup
from Cython.Build import cythonize
setup(name="fastops", ext_modules=cythonize("fastops.pyx", language_level=3, quiet=True))
"""

APP = """\
import fastops


def top(x):
    return fastops.Metric.get(x) + fastops.shift(x)
"""

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash, app
    from cash.analysis import report_store
    top = cash.cache(app.top)
    value = top(2.0)
    key = top.explain(2.0).cache_key
    report_store.flush()
    print(json.dumps({"stats": report_store.status(), "key": key, "value": value}))
    """
)


class BuildFailed(Exception):
    """The extension did not build; carries the compiler's output."""


def _build(proj: Path) -> None:
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


def _project(root: Path, pyx: str = PYX) -> Path:
    proj = root / "proj"
    proj.mkdir()
    (proj / "fastops.pyx").write_text(pyx, encoding="utf-8")
    (proj / "setup.py").write_text(SETUP, encoding="utf-8")
    (proj / "app.py").write_text(APP, encoding="utf-8")
    _build(proj)
    return proj


def _run(proj: Path, *, store: bool = True, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_STORE": None if store else "0", "CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(proj), cwd=proj, cache_dir=proj / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


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


def test_a_report_through_a_cython_classmethod_is_read_in_a_new_process(cython_compiler, tmp_path):
    proj = _project(tmp_path)
    first = _run(proj)
    assert first["value"] == 9.0
    assert first["stats"]["stored"] >= 1 and not first["stats"]["refused"], first["stats"]
    second = _run(proj)
    assert second["stats"]["loaded"] >= 1 and not second["stats"]["missed"], second["stats"]
    assert (second["key"], second["value"]) == (first["key"], first["value"])
    assert _run(proj, store=False)["key"] == first["key"]
    assert _run(proj, verify=True)["stats"]["verified"] >= 1


def test_a_rebuild_after_an_edit_to_the_compiled_method_walks_again(cython_compiler, tmp_path):
    proj = _project(tmp_path)
    first = _run(proj)
    assert _run(proj)["stats"]["loaded"] >= 1
    pyx = proj / "fastops.pyx"
    pyx.write_text(PYX.replace("return x * 3", "return x * 4"), encoding="utf-8")
    _build(proj)
    again = _run(proj)
    assert again["value"] == 11.0
    assert again["key"] != first["key"]
    assert again["key"] == _run(proj, store=False)["key"]
    assert again["stats"]["loaded"] == 0, again["stats"]
