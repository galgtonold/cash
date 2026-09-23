"""Reads the interpreter and libraries make for THEMSELVES are not data.

The tracker patches the process-wide ``open`` and ``os.listdir``, so it also
saw the import system list every ``sys.path`` directory (the notebook's own
folder included) and read package metadata, matplotlib load its styles on
import and its fonts on first draw, and Pillow reopen the PNG ``savefig`` had
just written with mode ``"w+b"``. All of it happens only the first time, so a
statement got a different lineage on its re-run; a new file next to the
notebook invalidated everything after an import; and a figure that looked
changed had its ``savefig`` replayed alone -- a blank chart.

Each drop has a control arm: the same kind of read that IS the user's data
stays tracked.
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap

import pytest

from cash.notebook import file_tracker
from cash.notebook.file_tracker import FileAccessTracker


def _norm(paths):
    return {os.path.normcase(str(p).replace("\\", "/")) for p in paths}


def _has(tracked, path):
    return os.path.normcase(str(path).replace("\\", "/")) in tracked


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    """An importable package that counts as INSTALLED (under a site root)."""
    site = tmp_path / "site"
    pkg = site / "fakelib"
    pkg.mkdir(parents=True)
    (pkg / "res.txt").write_text("resource", encoding="utf-8")
    (pkg / "__init__.py").write_text(
        textwrap.dedent("""
        import os
        HERE = os.path.dirname(__file__)

        def load_resource():
            with open(os.path.join(HERE, "res.txt"), encoding="utf-8") as fh:
                return fh.read()

        def read(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read()

        def probe(path):
            return os.path.exists(path)
    """),
        encoding="utf-8",
    )
    (site / "fakelib_importer.py").write_text(
        textwrap.dedent("""
        import os
        CONFIG = os.environ["FAKELIB_CONFIG"]
        with open(CONFIG, encoding="utf-8") as fh:
            SETTING = fh.read()
    """),
        encoding="utf-8",
    )
    root = os.path.normcase(str(site)).replace("\\", "/").rstrip("/") + "/"
    # raising=False: the fails-first control runs this against a tracker
    # that has neither function.
    monkeypatch.setattr(file_tracker, "_site_roots", lambda: (root,), raising=False)
    monkeypatch.setattr(file_tracker, "_installed_roots", lambda: (root,), raising=False)
    monkeypatch.syspath_prepend(str(site))
    for name in ("fakelib", "fakelib_importer"):
        sys.modules.pop(name, None)
    yield site
    for name in ("fakelib", "fakelib_importer"):
        sys.modules.pop(name, None)


@pytest.fixture
def user_file(tmp_path):
    p = tmp_path / "data" / "users.csv"
    p.parent.mkdir()
    p.write_text("id\n1\n", encoding="utf-8")
    return p


def _run(fn, user_ns=None):
    tracker = FileAccessTracker(user_ns={} if user_ns is None else user_ns)
    with tracker:
        fn()
    return tracker


# --- package metadata ---------------------------------------------------------


def test_a_package_metadata_lookup_is_not_tracked(tmp_path, monkeypatch):
    """``importlib.metadata`` lists the sys.path entry and reads METADATA."""
    root = tmp_path / "meta_root"
    info = root / "zzfake_dist-1.2.3.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: zzfake-dist\nVersion: 1.2.3\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    import importlib.metadata as md

    tracker = _run(lambda: md.version("zzfake-dist"))
    tracked = _norm(tracker.get_accessed_files())
    assert not [t for t in tracked if "meta_root" in t], tracked


# --- a library reading its own files -----------------------------------------


def test_a_library_reading_its_own_resource_is_not_tracked(fake_site):
    import fakelib

    tracker = _run(fakelib.load_resource)
    assert not _has(_norm(tracker.get_accessed_files()), fake_site / "fakelib" / "res.txt")


def test_a_library_reading_the_users_file_is_tracked(fake_site, user_file):
    """Control: `PIL.Image.open(p)` / `pd.read_csv(p)` read the user's data."""
    import fakelib

    tracker = _run(lambda: fakelib.read(str(user_file)))
    assert _has(_norm(tracker.get_accessed_files()), user_file)


def test_an_installed_tool_reading_its_own_data_is_tracked_for_its_own_functions(fake_site):
    """Control: for the tool's OWN cached functions its data files are inputs."""
    import fakelib

    tracker = _run(fakelib.load_resource, user_ns={"__name__": "fakelib.pipeline"})
    assert _has(_norm(tracker.get_accessed_files()), fake_site / "fakelib" / "res.txt")


def test_another_packages_data_read_by_the_standard_library_is_not_tracked(fake_site):
    """``zoneinfo`` loads ``tzdata/zoneinfo/UTC`` once per
    process. The reader is not the package that owns the file."""
    res = fake_site / "fakelib" / "res.txt"
    tracker = _run(lambda: open(res, encoding="utf-8").read())
    assert not _has(_norm(tracker.get_accessed_files()), res)
    # ... but for the package's own cached functions it is still an input
    tracker = _run(lambda: open(res, encoding="utf-8").read(), user_ns={"__name__": "fakelib.pipeline"})
    assert _has(_norm(tracker.get_accessed_files()), res)


def test_a_notebook_namespace_has_no_own_package():
    assert FileAccessTracker(user_ns={"__name__": "__main__"})._own_package is None
    assert FileAccessTracker(user_ns={"__name__": "tool.sub"})._own_package == "tool"


def test_a_library_probing_for_an_optional_file_of_its_own_is_not_an_absent_dependency(fake_site):
    import fakelib

    missing = fake_site / "fakelib" / "optional.cfg"
    tracker = _run(lambda: fakelib.probe(str(missing)))
    assert not tracker.absent_files, tracker.absent_files


def test_a_library_probing_for_the_users_file_is_an_absent_dependency(fake_site, tmp_path):
    """Control: `fakelib.probe(p)` asks about the user's path."""
    import fakelib

    tracker = _run(lambda: fakelib.probe(str(tmp_path / "later.csv")))
    assert tracker.absent_files


def test_the_user_probing_for_a_file_is_still_an_absent_dependency(tmp_path):
    """Control."""
    missing = tmp_path / "not_yet.csv"
    tracker = _run(lambda: os.path.exists(str(missing)))
    assert tracker.absent_files


# --- a library being imported -------------------------------------------------


def test_a_file_read_while_an_installed_library_imports_is_not_tracked(fake_site, tmp_path, monkeypatch):
    cfg = tmp_path / "fakelibrc"
    cfg.write_text("x", encoding="utf-8")
    monkeypatch.setenv("FAKELIB_CONFIG", str(cfg))
    tracker = _run(lambda: importlib.import_module("fakelib_importer"))
    assert not _has(_norm(tracker.get_accessed_files()), cfg)


def test_a_user_module_reading_its_config_at_import_is_tracked(tmp_path, monkeypatch):
    """Control: the user's own module is not installed; its config is data."""
    proj = tmp_path / "proj"
    proj.mkdir()
    cfg = proj / "settings.toml"
    cfg.write_text("x = 1", encoding="utf-8")
    (proj / "zz_user_settings.py").write_text(
        f"with open({str(cfg)!r}, encoding='utf-8') as fh:\n    VALUE = fh.read()\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(proj))
    sys.modules.pop("zz_user_settings", None)
    try:
        tracker = _run(lambda: importlib.import_module("zz_user_settings"))
    finally:
        sys.modules.pop("zz_user_settings", None)
    assert _has(_norm(tracker.get_accessed_files()), cfg)


# --- the interpreter's own files -----------------------------------------------


def test_reading_a_standard_library_file_is_not_tracked():
    src = os.path.join(os.path.dirname(os.__file__), "json", "__init__.py")

    def read():
        with open(src, encoding="utf-8") as fh:
            fh.read()

    tracker = _run(read)
    assert not _has(_norm(tracker.get_accessed_files()), os.path.realpath(src))


# --- a write that reads back ----------------------------------------------------


@pytest.mark.parametrize("mode", ["w+b", "w+", "x+b"])
def test_opening_for_write_plus_is_not_a_read(tmp_path, mode):
    """Pillow saves every image with "w+b" -- the chart is output, not input."""
    out = tmp_path / "chart.png"

    def write():
        with open(out, mode) as fh:
            fh.write(b"png" if "b" in mode else "png")

    tracker = _run(write)
    assert not _has(_norm(tracker.get_accessed_files()), out)


@pytest.mark.parametrize("mode", ["r+", "a+", "rb"])
def test_modes_that_read_existing_content_are_reads(tmp_path, mode):
    """Control: `r+` and `a+` see what was there before."""
    f = tmp_path / "log.txt"
    f.write_text("old", encoding="utf-8")

    def touch():
        with open(f, mode) as fh:
            fh.read()

    tracker = _run(touch)
    assert _has(_norm(tracker.get_accessed_files()), f)
