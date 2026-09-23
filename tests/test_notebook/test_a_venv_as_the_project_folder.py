"""A virtualenv created as the project folder does not make the project "installed".

``python -m venv .`` in the project puts ``bin/``, ``lib/`` and ``pyvenv.cfg``
beside the user's own modules, so ``sys.prefix`` IS the project folder. The
file tracker never counted the prefix itself as installed, for exactly this
reason; the function tracker did, so ``import helpers`` from that project was
taken for a library: never tracked, and editing ``helpers.py`` did not
invalidate what used it. Every part of cash now asks
``cash.install_paths``, which counts the standard library, site-packages and
the scripts directory as installed -- never the prefix.
"""

from __future__ import annotations

import sys

import pytest

from cash import install_paths
from cash.tracking.function_tracker import FunctionTracker, is_local_module


@pytest.fixture
def venv_project(tmp_path, monkeypatch):
    """A project folder that is also the running interpreter's venv prefix."""
    project = tmp_path / "project"
    site_packages = project / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    site_packages.mkdir(parents=True)
    (project / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (project / "venvhelpers.py").write_text("def scale(x):\n    return x * 2\n", encoding="utf-8")
    (site_packages / "venvlib.py").write_text("def scale(x):\n    return x * 3\n", encoding="utf-8")
    for name in ("prefix", "exec_prefix"):
        monkeypatch.setattr(sys, name, str(project))
    monkeypatch.syspath_prepend(str(site_packages))
    monkeypatch.syspath_prepend(str(project))
    install_paths.clear_caches()
    yield project
    for name in ("venvhelpers", "venvlib"):
        sys.modules.pop(name, None)
    monkeypatch.undo()
    install_paths.clear_caches()


def test_a_module_in_the_project_is_local(venv_project):
    import venvhelpers

    assert is_local_module(venvhelpers)


def test_the_function_tracker_tracks_it(venv_project):
    import venvhelpers  # noqa: F401 - imported so the tracker finds it in sys.modules

    tracker = FunctionTracker()
    assert tracker.auto_track_local_imports("import venvhelpers") == {"venvhelpers"}


def test_a_package_installed_into_that_venv_is_still_installed(venv_project):
    """The control: the fix must not make the venv's own packages local."""
    import venvlib

    assert not is_local_module(venvlib)
    assert FunctionTracker().auto_track_local_imports("import venvlib") == set()
