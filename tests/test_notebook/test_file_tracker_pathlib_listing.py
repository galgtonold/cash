"""Every way of listing a directory records that directory, on every Python.

pathlib does not always reach the patched ``os.scandir``: on 3.10 it lists
through ``_NormalAccessor``, and on 3.13 ``Path.glob`` goes through
``glob._Globber``, both holding the original function captured at import. A
notebook globbing a ledger folder with ``Path.glob`` on 3.13 never
saw the new month's file.
"""

import glob
import os
import pathlib

import pytest

from cash.tracking.file_tracker import FileAccessTracker

LISTINGS = {
    "Path.glob": lambda d: sorted(pathlib.Path(d).glob("*.csv")),
    "Path.rglob": lambda d: list(pathlib.Path(d).rglob("*.csv")),
    "Path.iterdir": lambda d: list(pathlib.Path(d).iterdir()),
    "glob.glob": lambda d: glob.glob(os.path.join(d, "*.csv")),
    "os.listdir": lambda d: os.listdir(d),
    "os.scandir": lambda d: list(os.scandir(d)),
}


@pytest.mark.parametrize("listing", LISTINGS)
def test_listing_records_the_directory(tmp_path, listing):
    (tmp_path / "a.csv").write_text("x")
    with FileAccessTracker({}) as tracker:
        LISTINGS[listing](str(tmp_path))
    listed = {os.path.normcase(os.path.abspath(p)).rstrip("\\/") for p in tracker.get_accessed_files()}
    assert os.path.normcase(str(tmp_path)) in listed, sorted(listed)
