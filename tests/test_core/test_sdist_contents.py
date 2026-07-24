"""The sdist must contain the project, and nothing that merely sat in the repo.

Checking git is NOT a valid test for this. Tools like venv and coverage drop a
self-ignoring ``.gitignore`` (``*``) inside their own output directory, so
``git status`` shows a clean tree and ``git check-ignore`` calls them ignored --
but hatchling's sdist walk does not honour those nested files and sweeps the
directory in regardless.

That is how four stray local venvs reached a built sdist: 25 MB across 1908
files, including the entirety of pip's vendored site-packages, in a 36 MB
archive. Every signal short of building the archive looked clean.

So this test builds a real sdist and looks inside it.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Substrings that mean "this is somebody's environment or build output, not the
# project". Matched against every archive member path, case-insensitively.
JUNK_MARKERS = (
    "site-packages",
    "pyvenv.cfg",
    ".venv/",
    "coverage_html",
    "/dist/",
    ".pytest_cache",
    "__pycache__",
)


@pytest.mark.slow
def test_sdist_has_no_environment_or_build_junk(tmp_path):
    if not (REPO_ROOT / "pyproject.toml").exists():
        pytest.skip("not running from the repo checkout")
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("`build` not installed; cannot construct an sdist to inspect")

    cp = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert cp.returncode == 0, f"sdist build failed:\n{cp.stdout[-3000:]}\n{cp.stderr[-3000:]}"

    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1, f"expected exactly one sdist, got {archives}"
    with tarfile.open(archives[0]) as tar:
        names = tar.getnames()

    assert names, "sdist is empty"
    offenders = sorted({n for n in names if any(m in n.lower() for m in JUNK_MARKERS)})
    assert not offenders, (
        f"{len(offenders)} environment/build path(s) leaked into the sdist, e.g. "
        f"{offenders[:5]}. Add an exclude to [tool.hatch.build.targets.sdist] in "
        f"pyproject.toml -- a nested .gitignore will NOT keep them out."
    )

    # Positive control: the exclusion must not have taken the package with it.
    assert any("/src/cash/__init__.py" in n for n in names), (
        "sdist does not contain the package itself"
    )

    # The sdist is MINIMAL: the package plus only the files needed to build it
    # (pyproject + the README/LICENSE the metadata references). Tests, docs,
    # example notebooks and benchmarks are not required to build, install, or run
    # cash — they live on GitHub — so they must never bloat the release.
    rel = [n.split("/", 1)[1] for n in names if "/" in n]  # drop the cash_lib-X.Y.Z/ prefix
    tops = {r.split("/")[0] for r in rel if r}
    allowed_top = {"src", "pyproject.toml", "README.md", "LICENSE", "PKG-INFO", ".gitignore"}
    assert tops <= allowed_top, (
        f"sdist ships non-essential top-level entries {sorted(tops - allowed_top)}; "
        f"the sdist is a minimal include list — see [tool.hatch.build.targets.sdist]."
    )
    assert not any(r.endswith(".ipynb") for r in rel), "notebooks must not ship in the sdist"
