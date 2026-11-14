"""An edit to a numba helper in the user's own package invalidates, however
the cached function reaches it.

A decorated estimator method reached an ``@njit`` helper in another module of
its package through module functions and through methods of its arguments
and of ``self``; a cached result must never outlive an edit to that helper.
Each variant is a package of the user's (``mypkg``) beside a script that calls
one cached function: run, hit in a second process, edit the helper's body,
and the third process must run again and return the new value. Every variant
runs with the call trace on and with ``CASH_TRACE=0``.

The package is also installed editable into a venv of its own, both ways
setuptools does it: a ``.pth`` that puts its folder on the path, and an
``__editable__`` finder that a ``.pth`` installs as an import hook.
"""

from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
import sysconfig
import textwrap
import zipfile
from pathlib import Path

import pytest

from tests._scripts import child_env, run_python, run_tied

pytest.importorskip("numba")


HELPERS = """
import numba
import numpy as np


@numba.njit
def inner(x):
    return x * 2.0


@numba.njit
def nb_helper(x):
    return x * 2.0


@numba.njit
def nb_outer(x):
    return inner(x) + 0.0


@numba.njit(cache=True)
def nb_disk(x):
    return x * 2.0


@numba.vectorize(["float64(float64)"])
def nb_vec(x):
    return x * 2.0


@numba.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)")
def nb_guvec(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * 2.0


@numba.njit(parallel=True)
def nb_par(x):
    out = np.empty_like(x)
    for i in numba.prange(x.shape[0]):
        out[i] = x[i] * 2.0
    return out
"""

OTHER = """
from mypkg.helpers import nb_helper


def middle(x):
    return nb_helper(x) + 0.0


class Worker:
    def __init__(self, offset):
        self.offset = offset

    def compute(self, x):
        return nb_helper(x) + self.offset
"""

ESTIMATOR = """
import numpy as np
import cash

from mypkg import other


class Estimator:
    def __init__(self, offset):
        self.offset = offset

    @cash.cache
    def fit(self, n):
        print("RAN", flush=True)
        return float(other.middle(np.arange(n, dtype=np.float64)).sum()) + self.offset
"""

#: variant: (the extra parameter of the cached call, the argument given for
#: it, the call that reaches the helper, the function the edit changes)
VARIANTS = {
    "direct": ("", "", "helpers.nb_helper(x)", "nb_helper"),
    "module_function": ("", "", "other.middle(x)", "nb_helper"),
    "argument_method": ("worker, ", "other.Worker(0.0), ", "worker.compute(x)", "nb_helper"),
    "passed_as_argument": ("fn, ", "helpers.nb_helper, ", "fn(x)", "nb_helper"),
    "njit_callee": ("", "", "helpers.nb_outer(x)", "inner"),
    "on_disk_cache": ("", "", "helpers.nb_disk(x)", "nb_disk"),
    "vectorize": ("", "", "helpers.nb_vec(x)", "nb_vec"),
    "guvectorize": ("", "", "helpers.nb_guvec(x)", "nb_guvec"),
    "prange": ("", "", "helpers.nb_par(x)", "nb_par"),
}

MAIN = """
import numpy as np
import cash
from mypkg import helpers, other


@cash.cache
def run({param}n):
    print("RAN", flush=True)
    x = np.arange(n, dtype=np.float64)
    return float(np.asarray({body}).sum())


print("RESULT", run({arg}5), flush=True)
"""

BOUND_MAIN = """
from mypkg.estimator import Estimator

print("RESULT", Estimator(0.0).fit(5), flush=True)
"""


def _project(tmp_path: Path, main: str) -> Path:
    project = tmp_path / "proj"
    package = project / "mypkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text(textwrap.dedent(HELPERS).lstrip(), encoding="utf-8")
    (package / "other.py").write_text(textwrap.dedent(OTHER).lstrip(), encoding="utf-8")
    (package / "estimator.py").write_text(textwrap.dedent(ESTIMATOR).lstrip(), encoding="utf-8")
    (project / "main.py").write_text(textwrap.dedent(main).lstrip(), encoding="utf-8")
    return project


def _run(
    project: Path, trace: str, package_dir: Path | None = None, python: Path | None = None
) -> tuple[bool, str, str]:
    env = {"CASH_TRACE": trace}
    if package_dir is not None and python is None:
        env["PYTHONPATH"] = os.pathsep.join([child_env()["PYTHONPATH"], str(package_dir)])
    if python is None:
        out = run_python("main.py", cwd=project, env=env, timeout=240)
    else:
        out = subprocess.run(
            [str(python), "main.py"],
            cwd=project,
            env=child_env(project, env=env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
        )
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    output = out.stdout + out.stderr
    result = next((line.split()[1] for line in out.stdout.splitlines() if line.startswith("RESULT")), None)
    assert result is not None, output
    return "RAN" in out.stdout, result, output


def _edit(package_dir: Path, function: str) -> None:
    """Make *function*'s factor 2.0 a 3.0, nothing else."""
    path = package_dir / "mypkg" / "helpers.py"
    text = path.read_text(encoding="utf-8")
    head, sep, rest = text.partition(f"def {function}(")
    assert sep, function
    body, nxt, tail = rest.partition("\n\n\n")
    assert "2.0" in body, body
    path.write_text(head + sep + body.replace("2.0", "3.0") + nxt + tail, encoding="utf-8")


def _check(
    project: Path, function: str, trace: str, package_dir: Path | None = None, python: Path | None = None
) -> None:
    # sum(0..4) = 10, doubled; after the edit tripled
    ran, first, output = _run(project, trace, package_dir, python)
    assert ran and first == "20.0", output
    ran, again, output = _run(project, trace, package_dir, python)
    assert not ran, f"the second process ran again instead of hitting:\n{output}"
    assert again == "20.0", output
    _edit(package_dir or project, function)
    ran, after, output = _run(project, trace, package_dir, python)
    assert ran, f"an edit to {function} was served the old result:\n{output}"
    assert after == "30.0", output


@pytest.mark.timeout(600)
@pytest.mark.parametrize("trace", ["1", "0"], ids=["traced", "untraced"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_an_edited_numba_helper_invalidates(tmp_path, variant, trace):
    param, arg, body, function = VARIANTS[variant]
    project = _project(tmp_path, MAIN.format(param=param, arg=arg, body=body))
    _check(project, function, trace)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("trace", ["1", "0"], ids=["traced", "untraced"])
def test_an_edited_numba_helper_invalidates_a_bound_method(tmp_path, trace):
    """The cached function is a method of the user's class, called bound; it
    reaches the helper through a module function of another module."""
    project = _project(tmp_path, BOUND_MAIN)
    _check(project, "nb_helper", trace)


@pytest.mark.timeout(600)
def test_an_edited_numba_helper_invalidates_a_package_checked_out_beside_the_project(tmp_path):
    """The package is a checkout of its own on ``PYTHONPATH``, outside the
    folder the script runs in: still the user's code, so still keyed by its
    source."""
    project = _project(tmp_path, BOUND_MAIN)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.move(str(project / "mypkg"), str(checkout / "mypkg"))
    _check(project, "nb_helper", "1", package_dir=checkout)


PYPROJECT = """
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "mypkg"
version = "0.1"

[tool.setuptools]
packages = ["mypkg"]
"""

BUILD_EDITABLE = """
import sys
from setuptools import build_meta

config = {"editable_mode": sys.argv[2]} if sys.argv[2] else None
print(build_meta.build_editable(sys.argv[1], config))
"""

#: setuptools' editable modes: ``compat`` writes a ``.pth`` path entry, its
#: default (``lenient``) an ``__editable__`` finder installed by a ``.pth``.
EDITABLE_MODES = {"pth_path": "compat", "editable_finder": ""}


def _editable_install(tmp_path: Path, project: Path, mode: str) -> tuple[Path, Path]:
    """Move ``mypkg`` out of *project* into a repository of its own, install
    it editable into a new venv, and return (the venv's python, the
    repository). The venv sees this interpreter's packages (numba, numpy,
    cash's dependencies) through a ``.pth``; the wheel is built by setuptools
    directly and unpacked, as an installer would, so no network is needed."""
    pytest.importorskip("setuptools")
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.move(str(project / "mypkg"), str(repo / "mypkg"))
    (repo / "pyproject.toml").write_text(textwrap.dedent(PYPROJECT).lstrip(), encoding="utf-8")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    built = run_tied([sys.executable, "-c", BUILD_EDITABLE, str(wheels), EDITABLE_MODES[mode]], cwd=repo, timeout=240)
    assert built.returncode == 0, built.stdout + built.stderr
    venv = tmp_path / "venv"
    made = run_tied([sys.executable, "-m", "venv", "--without-pip", str(venv)], timeout=240)
    assert made.returncode == 0, made.stdout + made.stderr
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    ours = dict.fromkeys([*site.getsitepackages(), sysconfig.get_path("purelib")])
    (Path(purelib) / "zz_parent_packages.pth").write_text("\n".join(ours) + "\n", encoding="utf-8")
    (wheel,) = wheels.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(purelib)
    names = os.listdir(purelib)
    assert any(name.endswith("_finder.py") for name in names) == (mode == "editable_finder"), names
    return python, repo


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(sys.platform == "win32", reason="the venv layout is built for POSIX paths")
@pytest.mark.parametrize("mode", sorted(EDITABLE_MODES))
@pytest.mark.parametrize("trace", ["1", "0"], ids=["traced", "untraced"])
@pytest.mark.parametrize("variant", [*sorted(VARIANTS), "bound_method"])
def test_an_edited_numba_helper_in_an_editable_install_invalidates(tmp_path, variant, trace, mode):
    """The package is installed editable, as a commit walk installs the
    repository under test, and the script runs outside its folder."""
    if variant == "bound_method":
        project, function = _project(tmp_path, BOUND_MAIN), "nb_helper"
    else:
        param, arg, body, function = VARIANTS[variant]
        project = _project(tmp_path, MAIN.format(param=param, arg=arg, body=body))
    python, repo = _editable_install(tmp_path, project, mode)
    _check(project, function, trace, package_dir=repo, python=python)
