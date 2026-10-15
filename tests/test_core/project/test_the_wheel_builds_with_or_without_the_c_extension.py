"""The wheel carries the C extension where it can, and builds without it
everywhere else.

``hatch_build.py`` compiles ``cash._cwatch`` on CPython 3.12+. These tests
build real wheels from the checkout and look inside:

* a normal build there is a platform wheel holding the compiled module and
  not its C sources;
* ``CASH_PURE_PYTHON=1`` builds the universal ``py3-none-any`` wheel;
* a compile that fails (no working compiler) still builds a wheel, pure
  Python, instead of failing: that is what keeps ``pip install`` from the
  sdist working on a machine without a compiler.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import pytest

from tests._scripts import run_tied

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = [pytest.mark.slow, pytest.mark.timeout(240)]


def _build_wheel(out: Path, **env: str) -> tuple[Path, str]:
    if not (REPO_ROOT / "pyproject.toml").exists():
        pytest.skip("not running from the repo checkout")
    pytest.importorskip("build", reason="`build` is not installed")
    child = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    child.update(env)
    cp = run_tied([sys.executable, "-m", "build", "--wheel", "--outdir", out], timeout=200, cwd=REPO_ROOT, env=child)
    assert cp.returncode == 0, f"wheel build failed:\n{cp.stdout[-3000:]}\n{cp.stderr[-3000:]}"
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0], cp.stdout + cp.stderr


def _members(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def _extension_members(wheel: Path) -> list[str]:
    return [n for n in _members(wheel) if n.startswith("cash/_cwatch")]


def test_the_switch_builds_the_universal_pure_wheel(tmp_path):
    wheel, log = _build_wheel(tmp_path, CASH_PURE_PYTHON="1")
    assert wheel.name.endswith("-py3-none-any.whl"), wheel.name
    assert not _extension_members(wheel)
    assert "CASH_PURE_PYTHON is set" in log
    # Positive control: the package itself is there.
    assert "cash/__init__.py" in _members(wheel)


@pytest.mark.skipif(sys.platform == "win32", reason="MSVC is found by setuptools, not through CC")
def test_a_failed_compile_builds_a_pure_wheel_instead_of_failing(tmp_path):
    wheel, log = _build_wheel(tmp_path, CC="false")
    assert wheel.name.endswith("-py3-none-any.whl"), wheel.name
    assert not _extension_members(wheel)
    if sys.version_info >= (3, 12) and sys.implementation.name == "cpython":
        assert "could not compile the C extension" in log, log[-3000:]


def test_a_normal_build_carries_the_extension_where_python_has_watchers(tmp_path):
    wheel, _ = _build_wheel(tmp_path)
    members = _members(wheel)
    assert not [n for n in members if n.startswith("cash/_c/")], "the C sources belong in the sdist only"
    if sys.version_info < (3, 12) or sys.implementation.name != "cpython":
        assert wheel.name.endswith("-py3-none-any.whl"), wheel.name
        assert not _extension_members(wheel)
        return
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert f"-{tag}-" in wheel.name, wheel.name
    assert len(_extension_members(wheel)) == 1, members
