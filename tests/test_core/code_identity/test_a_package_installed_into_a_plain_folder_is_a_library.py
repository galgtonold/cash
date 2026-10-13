"""A package installed into a plain folder on ``sys.path`` is a library.

``pip install --target DIR`` (or ``uv pip install --target``) puts packages in
a folder that is not called ``site-packages``; ``PYTHONPATH`` or a ``.pth``
file puts it on ``sys.path``. cash took every file there for the user's own
code: it walked LightGBM's and polars' internals as helpers, keyed them by
their source, reported their module-global writes as the user's, and a
cached ``train_lgbm`` was never stored ("a global ... changed while it ran").

Such a package is now a library, as it is in site-packages, when the folder
holds its installed metadata (a ``*.dist-info`` whose ``RECORD`` lists it)
and lies outside the project and the running script's folder. A source
folder with no metadata, an editable install and a package installed into
the project's own folder stay the user's code: an edit to them recomputes.
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from tests._scripts import CASH_SRC, run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

JOB = """
import sys
import cash
import fakelib

@cash.cache
def train(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return fakelib.Model().fit(x)

print(train(3))
"""


def _library(folder, factor, *, metadata=True, editable=False):
    package = folder / "fakelib"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            class Model:
                def fit(self, x):
                    return x * {factor}
            """
        ),
        encoding="utf-8",
    )
    if metadata:
        info = folder / "fakelib-1.0.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakelib\nVersion: 1.0\n", encoding="utf-8")
        (info / "RECORD").write_text("fakelib/__init__.py,,\nfakelib-1.0.dist-info/RECORD,,\n", encoding="utf-8")
        if editable:
            (info / "direct_url.json").write_text(
                json.dumps({"url": "file:///src/fakelib", "dir_info": {"editable": True}}), encoding="utf-8"
            )


def _project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text('[project]\nname = "proj"\n', encoding="utf-8")
    (proj / "job.py").write_text(JOB, encoding="utf-8")
    return proj


def _run(proj, folder):
    p = run_python("job.py", cwd=proj, env={"PYTHONPATH": os.pathsep.join([CASH_SRC, str(folder)])})
    return p.stdout.strip(), "[RUN]" in p.stderr


def _edit_between_runs(proj, folder, **kind):
    _library(folder, 2, **kind)
    first = _run(proj, folder)
    _library(folder, 5, **kind)
    return first, _run(proj, folder)


def test_a_target_folder_outside_the_project_is_keyed_as_a_library(tmp_path):
    proj = _project(tmp_path)
    first, second = _edit_between_runs(proj, tmp_path / "target")
    assert first == ("6", True)
    # A library is keyed by name, as one in site-packages is: the edit (an
    # upgrade in place under the same version) is not seen. Keyed as the
    # user's code, the edit recomputed.
    assert second == ("6", False), "the --target package was keyed as the user's code"


def test_a_source_folder_on_the_path_without_metadata_stays_user_code(tmp_path):
    proj = _project(tmp_path)
    first, second = _edit_between_runs(proj, tmp_path / "shared", metadata=False)
    assert first == ("6", True)
    assert second == ("15", True), "an edit to a source folder on PYTHONPATH was not seen"


def test_an_editable_install_outside_the_project_stays_user_code(tmp_path):
    proj = _project(tmp_path)
    first, second = _edit_between_runs(proj, tmp_path / "devsrc", editable=True)
    assert second == ("15", True), "an edit to an editable install was not seen"


def test_a_package_installed_into_the_project_folder_stays_user_code(tmp_path):
    proj = _project(tmp_path)
    first, second = _edit_between_runs(proj, proj / "vendor")
    assert second == ("15", True), "an edit to a package inside the project was not seen"


def test_the_verdict_names_the_installed_package_only(tmp_path, monkeypatch):
    from cash.install_paths import clear_caches, is_user_path

    target = tmp_path / "target"
    _library(target, 2)
    (target / "loose.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(target))
    clear_caches()
    try:
        assert not is_user_path(str(target / "fakelib" / "__init__.py"))
        # A file the metadata does not list is not the package's.
        assert is_user_path(str(target / "loose.py"))
    finally:
        monkeypatch.undo()
        clear_caches()
