"""A registered module first imported while a tracker is open keeps its real loader.

The import hook stands in for the loader only to patch the module after it
runs; ``importlib.resources``, ``pkgutil.get_data`` and the loader's own
methods must behave as they do for an import outside any tracker.
"""

import importlib.resources
import pkgutil
import sys

import pytest

from cash.tracking import reader_patches
from cash.tracking.file_tracker import FileAccessTracker
from cash.tracking.reader_patches import FileDependencyRegistry, _PatchingLoader


@pytest.fixture
def imported_under_tracker(tmp_path, monkeypatch):
    pkg = tmp_path / "loaderpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def read(path):\n    return path\n", encoding="utf-8")
    (pkg / "data.txt").write_text("hi\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = FileDependencyRegistry()
    registry.register("loaderpkg", "read", FileDependencyRegistry._create_path_arg_handler)
    monkeypatch.setattr(reader_patches, "_registry", registry)
    sys.modules.pop("loaderpkg", None)
    with FileAccessTracker() as tracker:
        import loaderpkg

        loaderpkg.read(str(pkg / "data.txt"))
    loaderpkg.tracked_during_import = tracker.get_accessed_files()
    yield loaderpkg
    sys.modules.pop("loaderpkg", None)


def test_the_module_keeps_its_real_loader(imported_under_tracker):
    mod = imported_under_tracker
    assert not isinstance(mod.__loader__, _PatchingLoader)
    assert not isinstance(mod.__spec__.loader, _PatchingLoader)
    assert mod.__loader__.is_package("loaderpkg")


def test_package_resources_are_readable(imported_under_tracker):
    assert importlib.resources.files("loaderpkg").joinpath("data.txt").read_text(encoding="utf-8") == "hi\n"
    assert pkgutil.get_data("loaderpkg", "data.txt") == b"hi\n"


def test_the_module_was_still_patched(imported_under_tracker):
    assert any(p.endswith("data.txt") for p in imported_under_tracker.tracked_during_import)
