"""A numba ``@stencil`` and a library's ``@njit`` are keyed as code, cheaply.

A ``@stencil`` (numba's ``StencilFunc``) keeps no ``py_func``: read as a
global it was taken for a callable instance and its whole state -- the
kernel's IR, type caches, typing and target contexts -- was walked on every
key build (about 20 ms a hit, librosa's ``_localmax_sten`` and its
zero-crossing stencil among them), then warned KEY-UNHASHABLE-GLOBAL, so an
edit to the kernel never invalidated.

The kernel is the function its IR was built from. In a library it is keyed
as library code, by its name and options (the library's version is keyed
with the other installed libraries); in the user's own file by its source,
as any helper. Each run is a fresh interpreter over one cache folder.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from tests._scripts import child_env, run_python

pytest.importorskip("numba")


USER_HELPERS = """
import numba

@numba.stencil
def smooth(a):
    return a[-1] + a[0] + a[1]

@numba.njit
def scale(x):
    return x * 2.0
"""

USER_MAIN = """
import numpy as np
import cash
from helpers import scale, smooth

@cash.cache
def run(n):
    print("RAN", flush=True)
    return float(scale(smooth(np.arange(n, dtype=np.float64))).sum())

print("RESULT", run(5), flush=True)
"""

LIBRARY = """
import numba

@numba.stencil
def peak(a):
    return a[0] > a[-1] and a[0] >= a[1]

@numba.njit
def double(x):
    return x * 2.0
"""

LIBRARY_MAIN = """
import numpy as np
import cash
import cash.decorator.key_values as key_values
from fakelib.kernels import double, peak

# How many objects the key build rewrites for hashing: numba's internals
# walked as data are tens of thousands.
walked = [0]
_inner = key_values._stabilized

def counting(*args, **kwargs):
    walked[0] += 1
    return _inner(*args, **kwargs)

key_values._stabilized = counting

@cash.cache
def run(n):
    print("RAN", flush=True)
    x = np.arange(n, dtype=np.float64)
    return float(double(x).sum()) + float(peak(x).sum())

print("RESULT", run(5), flush=True)
walked[0] = 0
run(5)
print("WALKED", walked[0], flush=True)
"""


def _user_project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "helpers.py").write_text(textwrap.dedent(USER_HELPERS).lstrip(), encoding="utf-8")
    (project / "main.py").write_text(textwrap.dedent(USER_MAIN).lstrip(), encoding="utf-8")
    return project


def _run(project: Path, env: dict[str, str] | None = None) -> tuple[bool, str, str]:
    out = run_python("main.py", cwd=project, cache_dir=project / ".cash", env=env)
    result = next(line.split()[1] for line in out.stdout.splitlines() if line.startswith("RESULT"))
    return "RAN" in out.stdout, result, out.stdout + out.stderr


def _edit(project: Path, old: str, new: str) -> None:
    path = project / "helpers.py"
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def _assert_caches(project: Path, expected: str, env: dict[str, str] | None = None) -> str:
    ran, first, output = _run(project, env)
    assert ran and first == expected, output
    assert "KEY-UNHASHABLE" not in output, output
    ran, again, output = _run(project, env)
    assert not ran, f"the second process ran again instead of hitting:\n{output}"
    assert again == expected
    assert "KEY-UNHASHABLE" not in output, output
    return output


# smooth(0..4) with zero borders = [0, 3, 6, 9, 0]; doubled and summed
USER_RESULT = "36.0"


@pytest.mark.timeout(180)
def test_editing_a_user_stencil_invalidates(tmp_path):
    project = _user_project(tmp_path)
    _assert_caches(project, USER_RESULT)
    _edit(project, "return a[-1] + a[0] + a[1]", "return a[-1] + a[1]")
    ran, result, output = _run(project)
    assert ran, f"an edited stencil was served the old result:\n{output}"
    # [0, 2, 4, 6, 0] doubled
    assert result == "24.0"


@pytest.mark.timeout(180)
def test_editing_a_user_jitted_helper_invalidates(tmp_path):
    project = _user_project(tmp_path)
    _assert_caches(project, USER_RESULT)
    _edit(project, "return x * 2.0", "return x * 3.0")
    ran, result, output = _run(project)
    assert ran, f"an edited @njit helper was served the old result:\n{output}"
    assert result == "54.0"


@pytest.mark.timeout(180)
def test_a_library_stencil_and_jitted_function_hit_without_walking_numba(tmp_path):
    site = tmp_path / "site-packages"
    (site / "fakelib").mkdir(parents=True)
    (site / "fakelib" / "__init__.py").write_text("", encoding="utf-8")
    (site / "fakelib" / "kernels.py").write_text(textwrap.dedent(LIBRARY).lstrip(), encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text(textwrap.dedent(LIBRARY_MAIN).lstrip(), encoding="utf-8")
    # 0..4 doubled; the stencil's border is cval 0, so it finds no peak
    path = os.pathsep.join([child_env()["PYTHONPATH"], str(site)])
    output = _assert_caches(project, "20.0", env={"PYTHONPATH": path})
    walked = int(next(line.split()[1] for line in output.splitlines() if line.startswith("WALKED")))
    # Keyed by name, not by numba's state: about 8,000 objects a hit when the
    # stencil was walked as data.
    assert walked < 100, f"a hit walked {walked} objects"
