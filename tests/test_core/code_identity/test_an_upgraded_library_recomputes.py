"""An upgraded library recomputes: its installed version is in the key.

cash keys a call into a library by name and does not read its code, so after
``pip install -U`` the cached results the old version computed were still
served, with no warning (a plain-installed team package the same). The
installed version of each library the function reaches by a global name is
now folded into the key: a new version recomputes, the same version (also
code reinstalled under it) hits as before.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap

import numpy as np
import pandas as pd
import pytest

from cash.decorator.library_versions import library_tops, library_versions
from tests._scripts import CASH_SRC, run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

JOB = """
import sys
import cash
import fakelib
from fakelib import scale

@cash.cache
def by_module(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return fakelib.Model().fit(x)

@cash.cache
def by_name(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return scale(x)

def helper(x):
    return fakelib.Model().fit(x)

@cash.cache
def through_a_helper(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return helper(x)

print({"by_module": by_module, "by_name": by_name, "through_a_helper": through_a_helper}[sys.argv[1]](3))
"""


def _install(folder, version, factor):
    shutil.rmtree(folder, ignore_errors=True)
    package = folder / "fakelib"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            class Model:
                def fit(self, x):
                    return x * {factor}

            def scale(x):
                return x * {factor}
            """
        ),
        encoding="utf-8",
    )
    info = folder / f"fakelib-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: fakelib\nVersion: {version}\n", encoding="utf-8")
    (info / "top_level.txt").write_text("fakelib\n", encoding="utf-8")
    (info / "RECORD").write_text(
        f"fakelib/__init__.py,,\nfakelib-{version}.dist-info/RECORD,,\nfakelib-{version}.dist-info/METADATA,,\n",
        encoding="utf-8",
    )


def _run(proj, folder, which):
    p = run_python("job.py", which, cwd=proj, env={"PYTHONPATH": os.pathsep.join([CASH_SRC, str(folder)])})
    assert p.returncode == 0, p.stderr
    return p.stdout.strip(), "[RUN]" in p.stderr


@pytest.fixture
def proj(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text('[project]\nname = "proj"\n', encoding="utf-8")
    (proj / "job.py").write_text(JOB, encoding="utf-8")
    return proj


@pytest.mark.parametrize("which", ["by_module", "by_name", "through_a_helper"])
def test_an_upgrade_recomputes_and_the_same_version_hits(proj, tmp_path, which):
    lib = tmp_path / "site"
    _install(lib, "1.0", 2)
    assert _run(proj, lib, which) == ("6", True)
    assert _run(proj, lib, which) == ("6", False), "an unchanged library recomputed"
    _install(lib, "2.0", 5)
    assert _run(proj, lib, which) == ("15", True), "the upgrade served the old version's result"
    assert _run(proj, lib, which) == ("15", False)


def test_only_libraries_count():
    def f(x):
        return np.mean(x) + json.loads("1") + len([v for v in x if v])

    assert library_tops([f], "tests") == {"numpy"}
    assert ("numpy", np.__version__) in library_versions([f], "tests")


def test_what_a_library_requires_counts_too():
    def f(x):
        return pd.Series(x).sum()

    assert library_tops([f], "tests") == {"pandas"}
    assert ("numpy", np.__version__) in library_versions([f], "tests")


def test_a_function_that_reaches_no_library_folds_nothing():
    def f(x):
        return x + 1

    assert library_versions([f], "tests") == []
