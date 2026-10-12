"""A folder that gains a file while a cached call lists it is not stored.

``names_of(d)`` listed an inbox while another process dropped a file into
it. The entry was stored against the folder as it was at store time --
with the new file -- next to a listing without it, and every later call,
in this process and the next, returned the short listing. A listed folder
whose stat moved during the call now refuses the store, as a file read does
(STORE-INPUT-CHANGED), unless the call changed an entry of the folder
itself.
"""

from __future__ import annotations

import os
import threading
import warnings

import pytest

pytestmark = pytest.mark.core


class _Dropper:
    """Writes *name* into the folder once the body has listed it."""

    def __init__(self, folder, name):
        self.folder, self.name = folder, name
        self.listed = threading.Event()
        self.dropped = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def __cash_key__(self):
        return (str(self.folder), self.name)

    def _run(self):
        self.listed.wait(10)
        (self.folder / self.name).write_text("9", encoding="utf-8")
        self.dropped.set()


def _inbox(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i in range(3):
        (inbox / f"f{i}.csv").write_text(str(i), encoding="utf-8")
    return inbox


def test_a_file_landing_mid_call_is_listed_by_the_next_call(disk_cash, tmp_path):
    inbox = _inbox(tmp_path)

    @disk_cash.cache
    def names_of(d, dropper):
        names = sorted(os.listdir(d))
        if dropper is not None:
            dropper.listed.set()
            dropper.dropped.wait(10)
        return names

    dropper = _Dropper(inbox, "late.csv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert names_of(str(inbox), dropper) == ["f0.csv", "f1.csv", "f2.csv"]
    assert any("STORE-INPUT-CHANGED" in str(w.message) for w in caught)
    assert names_of(str(inbox), dropper) == ["f0.csv", "f1.csv", "f2.csv", "late.csv"]


def test_a_glob_is_a_listing_too(disk_cash, tmp_path):
    inbox = _inbox(tmp_path)

    @disk_cash.cache
    def csvs(d, dropper):
        import glob

        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.csv")))
        dropper.listed.set()
        dropper.dropped.wait(10)
        return names

    dropper = _Dropper(inbox, "late.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csvs(str(inbox), dropper)
    assert "late.csv" in csvs(str(inbox), dropper)


@pytest.mark.parametrize("how", ["write", "temp-and-replace", "remove", "mkdir"])
def test_a_call_changing_the_folder_itself_is_stored(disk_cash, tmp_path, how):
    """Control: its own entry moves the folder as an outside one does, and a
    function listing a folder and writing its output into it caches."""
    inbox = _inbox(tmp_path)
    runs = []

    @disk_cash.cache
    def index(d):
        runs.append(1)
        names = sorted(os.listdir(d))
        if how == "write":
            with open(os.path.join(d, "index.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(names))
        elif how == "temp-and-replace":
            with open(os.path.join(d, "index.tmp"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(names))
            os.replace(os.path.join(d, "index.tmp"), os.path.join(d, "index.txt"))
        elif how == "remove":
            os.remove(os.path.join(d, "f0.csv"))
        else:
            os.mkdir(os.path.join(d, f"sub{len(runs)}"))
        return names

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index(str(inbox))
    assert not any("STORE-INPUT-CHANGED" in str(w.message) for w in caught)
