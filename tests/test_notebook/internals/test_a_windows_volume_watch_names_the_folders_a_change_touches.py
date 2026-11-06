"""A Windows volume's watch names the watched folders a reported change touches.

A watch per folder held a handle on it, and Windows then refused to rename
or move that folder or any folder above it: renaming a project folder in
Explorer failed while a kernel ran. Each volume is now watched once from its
root, over the whole tree, and the names it reports are mapped to the
watched folders they touch (`file_watch._FolderIndex`). A name that may be a
short (8.3) alias, an overflow, or a report that cannot be read counts as a
change of every folder it could name. These tests stand in for kernel32, so
they run on every OS.
"""

from __future__ import annotations

import struct
import threading
import time

import pytest

from cash.tracking import file_watch
from cash.tracking.file_watch import _FolderIndex, _Volume, parse_notify_information


def _notify(*names: str, action: int = 3) -> bytes:
    """A buffer of FILE_NOTIFY_INFORMATION records naming *names*, each with
    *action* (3: modified, 2: removed, 4: renamed from)."""
    records = []
    for name in names:
        encoded = name.encode("utf-16-le")
        record = struct.pack("<III", 0, action, len(encoded)) + encoded
        records.append(record + b"\0" * (-len(record) % 4))
    out = b""
    for i, record in enumerate(records):
        following = len(record) if i < len(records) - 1 else 0
        out += struct.pack("<I", following) + record[4:]
    return out


@pytest.fixture
def index() -> _FolderIndex:
    folders = _FolderIndex()
    for path in ["Data\\Photos\\Anna", "Data\\Photos\\Bob", "Data\\Other", "Work"]:
        folders.add(path, path.lower())
    return folders


def test_a_file_changed_in_a_watched_folder_touches_that_folder_only(index):
    assert index.touched("data\\photos\\anna\\img001.jpg") == ["data\\photos\\anna"]
    assert index.touched("DATA\\PHOTOS\\ANNA\\IMG001.JPG") == ["data\\photos\\anna"]  # case folded


def test_a_change_elsewhere_touches_nothing(index):
    assert index.touched("Users\\me\\AppData\\Local\\Temp\\x.tmp") == []
    assert index.touched("Data\\Photos\\Anna\\deeper\\img.jpg") == []
    assert index.touched("Data\\Photos\\readme.txt") == []


def test_a_renamed_folder_above_watched_ones_touches_all_below_it(index):
    assert sorted(index.touched("Data\\Photos")) == ["data\\photos\\anna", "data\\photos\\bob"]
    assert sorted(index.touched("Data")) == ["data\\other", "data\\photos\\anna", "data\\photos\\bob"]


def test_a_watched_folder_itself_changed_touches_it_and_its_holder(index):
    index.add("Data\\Photos", "data\\photos")
    assert sorted(index.touched("Data\\Photos\\Anna")) == ["data\\photos", "data\\photos\\anna"]


def test_a_name_that_may_be_a_short_alias_touches_everything_below_its_folder(index):
    # Anything below Data: the alias may name any folder there.
    assert sorted(index.touched("Data\\PHOTOS~1\\Anna\\img001.jpg")) == [
        "data\\other",
        "data\\photos\\anna",
        "data\\photos\\bob",
    ]
    assert sorted(index.touched("DATA~1\\x")) == sorted(index.keys.values())
    assert index.touched("Data\\Photos\\Anna\\~$report.docx") == ["data\\photos\\anna"]  # no digit: not an alias


def test_a_removed_folder_is_no_longer_named(index):
    index.remove("Data\\Photos\\Bob")
    assert index.touched("Data\\Photos") == ["data\\photos\\anna"]
    index.remove("Data\\Photos\\Anna")
    assert index.touched("Data\\Photos") == []
    assert "data\\photos" not in index.above


def test_notify_records_are_read_whole_or_not_at_all():
    data = _notify("Data\\Photos\\Anna\\a.jpg", "Work\\b.txt")
    assert parse_notify_information(data) == [(3, "Data\\Photos\\Anna\\a.jpg"), (3, "Work\\b.txt")]
    assert parse_notify_information(data[:10]) is None
    assert parse_notify_information(data[:-4]) is None


class _Kernel:
    """kernel32 as a volume's watch calls it. `report` makes the pending read
    end with *data* (or *error*); a wait never ends by itself."""

    def __init__(self) -> None:
        self.ended: tuple[int, int] | None = None
        self.payload = b""
        self.issued = 0
        self.can_issue = True
        self.cancelled = 0
        self.closed: list = []
        self.never_ends = False
        self.waited_in: set = set()

    def report(self, data: bytes = b"", error: int = 0) -> None:
        self.payload = data
        self.ended = (len(data), error)

    def new_event(self):
        return "event"

    def new_read(self):
        return object(), object()

    def issue(self, handle, overlapped, buffer, event):
        self.issued += 1
        return self.can_issue

    def result(self, handle, overlapped):
        if self.ended is None:
            return False, 0, file_watch._ERROR_IO_INCOMPLETE
        n, error = self.ended
        self.ended = None
        return True, n, error

    def data(self, buffer, n):
        return self.payload[:n]

    def wait(self, event, ms):
        self.waited_in.add(threading.current_thread())
        time.sleep(0.005)
        return False

    def cancel(self, handle, overlapped):
        self.cancelled += 1
        if not self.never_ends:
            self.ended = (0, 995)  # ERROR_OPERATION_ABORTED

    def close(self, handle):
        self.closed.append(handle)


@pytest.fixture
def volume():
    kernel = _Kernel()
    v = _Volume(kernel, "C:\\", "root-handle")
    for path in ["Data\\Photos\\Anna", "Work"]:
        v.index.add(path, path.lower())
    assert v.start()
    yield v, kernel
    v.stop()


def test_a_poll_takes_a_report_the_thread_has_not(volume):
    v, kernel = volume
    assert v.poll() == ([], [], False)
    kernel.report(_notify("Data\\Photos\\Anna\\a.jpg"))
    assert v.poll() == (["data\\photos\\anna"], [], False)
    assert kernel.issued == 2  # the next read is issued at once
    assert v.poll() == ([], [], False)


def test_an_overflow_touches_every_folder_of_the_volume(volume):
    v, kernel = volume
    kernel.report(b"")  # ended with nothing read: the buffer overflowed
    keys, _gone, dead = v.poll()
    assert sorted(keys) == ["data\\photos\\anna", "work"] and not dead


def test_an_unreadable_report_touches_every_folder_of_the_volume(volume):
    v, kernel = volume
    kernel.report(b"\x01\x02\x03")
    assert sorted(v.poll()[0]) == ["data\\photos\\anna", "work"]


def test_a_read_that_fails_ends_the_volumes_watch(volume):
    v, kernel = volume
    kernel.report(error=5)  # access denied: the volume went
    keys, _gone, dead = v.poll()
    assert dead and sorted(keys) == ["data\\photos\\anna", "work"]


def test_a_read_that_cannot_be_issued_again_ends_the_volumes_watch(volume):
    v, kernel = volume
    kernel.can_issue = False
    kernel.report(_notify("Work\\a.txt"))
    _keys, _gone, dead = v.poll()
    assert dead


def test_stopping_a_cancelled_read_that_never_ends_waits_a_while_not_for_good(monkeypatch):
    monkeypatch.setattr(file_watch, "_CANCEL_WAIT_SECONDS", 0.05)
    kernel = _Kernel()
    kernel.never_ends = True
    v = _Volume(kernel, "C:\\", "root-handle")
    assert v.start()
    started = time.monotonic()
    v.stop()
    assert time.monotonic() - started < 2
    assert kernel.cancelled == 1
    assert "root-handle" in kernel.closed
    assert len(v.kept) == 1  # its buffer is never freed under the system


def test_stopping_a_read_that_ends_frees_its_buffer():
    kernel = _Kernel()
    v = _Volume(kernel, "C:\\", "root-handle")
    assert v.start()
    v.stop()
    assert kernel.cancelled == 1
    assert v.kept == []
    assert "root-handle" in kernel.closed


# -- placing a folder on its volume ------------------------------------------------


class _Disk(_Kernel):
    """A volume ``C:\\`` for `_VolumeChanges`: real folders with their true
    case, short aliases, and junctions, counting each lookup it answers."""

    def __init__(self) -> None:
        super().__init__()
        self.folders: dict[str, str] = {}  # case folded -> true case
        self.aliases = {"progra~1": "Program Files"}
        self.junctions: dict[str, str] = {}  # case folded -> target
        self.calls = {"real_path": 0, "long_path": 0, "volume_root": 0, "local": 0, "attributes": 0}

    def make(self, *paths: str) -> None:
        for path in paths:
            parts = path.split("\\")
            for i in range(1, len(parts) + 1):
                self.folders["\\".join(parts[:i]).lower()] = "\\".join(parts[:i])

    def _follow(self, path: str) -> str:
        parts = path.replace("/", "\\").split("\\")
        for i in range(len(parts), 0, -1):
            target = self.junctions.get("\\".join(parts[:i]).lower())
            if target is not None:
                return self._follow("\\".join([target, *parts[i:]]))
        return path

    def real_path(self, path):
        self.calls["real_path"] += 1
        return self._follow(path)

    def long_path(self, path):
        self.calls["long_path"] += 1
        return self._long(path)

    def _long(self, path):
        parts = [self.aliases.get(p.lower(), p) for p in path.replace("/", "\\").rstrip("\\").split("\\")]
        if parts == ["C:"]:
            return "C:\\"
        return self.folders.get("\\".join(parts).lower())

    def volume_root(self, path):
        self.calls["volume_root"] += 1
        return "C:\\"

    def local(self, root):
        self.calls["local"] += 1
        return True

    def attributes(self, path):
        self.calls["attributes"] += 1
        if path.replace("/", "\\").lower() in self.junctions:
            return 0x10 | 0x400
        return 0x10 if self._long(self._follow(path)) else None

    def open_root(self, root):
        return "root-handle"

    def lookups(self) -> int:
        """The costly lookups: a real path, a long path, a volume."""
        return self.calls["real_path"] + self.calls["long_path"] + self.calls["volume_root"]


@pytest.fixture
def disk():
    kernel = _Disk()
    changes = file_watch._VolumeChanges(kernel)
    yield kernel, changes
    changes.close()


def _packages(n: int) -> list[str]:
    return [f"C:\\Python\\Lib\\site-packages\\pkg{i // 10}\\sub{i % 10}" for i in range(n)]


def test_a_folder_below_a_placed_one_costs_one_attribute_lookup(disk):
    kernel, changes = disk
    folders = _packages(200)
    kernel.make(*folders)
    for folder in sorted({f.rsplit("\\", 1)[0] for f in folders}) + folders:
        assert changes.add(folder) is not None
    # 220 folders: the costly lookups were 2 x 3 per folder, now the root once or twice.
    assert kernel.lookups() <= 3 * 2
    assert kernel.calls["attributes"] <= 220 + 10  # and once for each folder above them


def test_a_folder_whose_parent_was_never_placed_is_placed_once(disk):
    kernel, changes = disk
    folders = _packages(200)
    kernel.make(*folders)
    for folder in folders:
        assert changes.add(folder) is not None
    assert kernel.lookups() <= 3 * 2  # the volume root, the first time twice


def test_a_derived_place_is_the_place_found_by_lookups(disk):
    kernel, changes = disk
    kernel.make("C:\\Users\\Anna\\Data\\Photos")
    changes.add("C:\\Users\\Anna")
    changes.add("C:\\Users\\Anna\\x")  # never made: not watched
    derived = changes.add("c:\\USERS\\anna\\DATA")
    assert derived == ("c:\\", "users\\anna\\data")
    assert changes.add("C:/Users/Anna/Data/Photos") == ("c:\\", "users\\anna\\data\\photos")


def test_a_junction_is_placed_where_it_leads(disk):
    kernel, changes = disk
    kernel.make("C:\\Work\\Real\\Data", "C:\\Users\\Anna")
    kernel.junctions["c:\\users\\anna\\link"] = "C:\\Work\\Real"
    changes.add("C:\\Users\\Anna")
    changes.add("C:\\Users\\Anna\\Other")
    assert changes.add("C:\\Users\\Anna\\Link") == ("c:\\", "work\\real")
    # Below a junction: its parent is placed where the junction leads.
    assert changes.add("C:\\Users\\Anna\\Link\\Data") == ("c:\\", "work\\real\\data")


def test_a_short_alias_is_placed_by_its_long_name(disk):
    kernel, changes = disk
    kernel.make("C:\\Program Files\\App")
    changes.add("C:\\Program Files")
    assert changes.add("C:\\PROGRA~1\\App") == ("c:\\", "program files\\app")


def test_a_parent_reported_renamed_is_placed_again(disk):
    kernel, changes = disk
    kernel.make("C:\\Proj\\Data\\A", "C:\\Proj\\Data\\B", "C:\\Proj\\Data\\C")
    changes.add("C:\\Proj")
    a = changes.add("C:\\Proj\\Data\\A")
    changes.add("C:\\Proj\\Data\\B")
    before = kernel.lookups()
    assert changes.add("C:\\Proj\\Data\\C") is not None
    assert kernel.lookups() == before  # derived from the kept parent

    # Moved away, and a junction left where it was.
    kernel.make("C:\\Moved\\Data\\A", "C:\\Moved\\Data\\D")
    kernel.junctions["c:\\proj"] = "C:\\Moved"
    kernel.report(_notify("Proj", action=4))
    reported = changes.poll()
    assert (a, True) in reported  # a watched folder below it is no longer where its key says
    assert changes.add("C:\\Proj\\Data\\D") == ("c:\\", "moved\\data\\d")
    assert kernel.lookups() > before


def test_a_file_removed_in_a_kept_folder_keeps_it(disk):
    kernel, changes = disk
    kernel.make("C:\\Proj\\Data\\A", "C:\\Proj\\Data\\B")
    changes.add("C:\\Proj\\Data")
    changes.add("C:\\Proj\\Data\\A")
    kernel.report(_notify("Proj\\Data\\notes.tmp", action=2))
    assert changes.poll() == [(("c:\\", "proj\\data"), False)]
    before = kernel.lookups()
    changes.add("C:\\Proj\\Data\\B")
    assert kernel.lookups() == before


def test_an_overflow_forgets_every_kept_place(disk):
    kernel, changes = disk
    kernel.make("C:\\Proj\\Data\\A", "C:\\Proj\\Data\\B")
    changes.add("C:\\Proj\\Data")
    changes.add("C:\\Proj\\Data\\A")
    kernel.report(b"")
    changes.poll()
    before = kernel.lookups()
    changes.add("C:\\Proj\\Data\\B")
    assert kernel.lookups() > before


def test_a_parent_reported_moved_while_its_child_is_placed_places_the_child_afresh(disk):
    kernel, changes = disk
    kernel.make("C:\\Proj\\Data\\A", "C:\\Proj\\Data\\B")
    changes.add("C:\\Proj\\Data")
    changes.add("C:\\Proj\\Data\\A")
    looked = kernel.attributes

    def moved_meanwhile(path):
        if path.endswith("B"):
            kernel.make("C:\\Moved\\B")
            kernel.junctions["c:\\proj\\data"] = "C:\\Moved"
            kernel.report(_notify("Proj\\Data", action=4))
            changes._volumes["c:\\"].poll()  # the thread takes it
        return looked(path)

    kernel.attributes = moved_meanwhile
    assert changes.add("C:\\Proj\\Data\\B") == ("c:\\", "moved\\b")


def test_a_poll_takes_reports_without_waiting_for_the_volumes_thread(disk):
    # The cell's thread takes what ended itself, under the volume's lock: it
    # never waits on the report event, which only the volume's thread does.
    kernel, changes = disk
    kernel.make("C:\\Proj\\Data")
    changes.add("C:\\Proj\\Data")
    for _ in range(20):
        kernel.report(_notify("Proj\\Data\\a.txt"))
        assert changes.poll() == [(("c:\\", "proj\\data"), False)]
    time.sleep(0.05)
    assert kernel.waited_in  # the volume's thread waits
    assert threading.current_thread() not in kernel.waited_in


# -- a name reported through a short alias -----------------------------------------
#
# A change is reported by the name it was made through. A program writing
# through a short path (``%TEMP%`` is often ``C:\Users\PHILIP~1\AppData\...``)
# names its files so, and each such name counted as a change of every
# watched folder below ``C:\Users``: every temporary file any program wrote
# dropped every stat kept for the notebook, its packages and its data.


@pytest.fixture
def home() -> _FolderIndex:
    folders = _FolderIndex()
    for path in ["Users\\Philipp Gergen\\proj\\.venv\\lib\\numpy", "Users\\Philipp Gergen\\proj\\data"]:
        folders.add(path, path.lower())
    return folders


_LONG = {"users\\philip~1": "Users\\Philipp Gergen", "users\\philipp gergen\\proj\\data\\photos~1": None}


def _long(below: str, last: bool) -> str | None:
    return _LONG.get(below)


def test_a_short_alias_put_in_its_long_form_touches_the_folder_it_names_only(home):
    temp = "Users\\PHILIP~1\\AppData\\Local\\Temp\\tmp123.json"
    assert home.touched(temp, _long) == []
    data = "Users\\PHILIP~1\\proj\\data\\a.csv"
    assert home.touched(data, _long) == ["users\\philipp gergen\\proj\\data"]
    assert home.moved("Users\\PHILIP~1\\proj\\data", _long) == ["users\\philipp gergen\\proj\\data"]


def test_a_short_alias_with_no_long_form_still_touches_everything_below_its_folder(home):
    # No longer there to be looked up: the alias may have named any folder.
    assert sorted(home.touched("Users\\XYZ~1\\a.txt", _long)) == sorted(home.keys.values())
    assert home.touched("Users\\PHILIP~1\\proj\\data\\PHOTOS~1\\a.jpg", _long) == ["users\\philipp gergen\\proj\\data"]
    assert sorted(home.touched("Users\\PHILIP~1\\a.txt")) == sorted(home.keys.values())  # none asked


def test_a_volumes_reports_through_a_short_alias_are_looked_up_once():
    kernel = _Disk()
    kernel.make("C:\\Users\\Philipp Gergen\\AppData\\Local\\Temp", "C:\\Users\\Philipp Gergen\\proj\\data")
    kernel.aliases["philip~1"] = "Philipp Gergen"
    v = _Volume(kernel, "C:\\", "root-handle")
    v.index.add("Users\\Philipp Gergen\\proj\\data", "data")
    assert v.start()
    try:
        for i in range(20):
            kernel.report(_notify(f"Users\\PHILIP~1\\AppData\\Local\\Temp\\tmp{i}.json"))
            assert v.poll() == ([], [], False)
        assert kernel.calls["long_path"] == 1
        kernel.report(_notify("Users\\PHILIP~1\\proj\\data\\a.csv"))
        assert v.poll() == (["data"], [], False)
        # A file gone frees no folder's alias; the folder gone does: looked up afresh.
        kernel.report(_notify("Users\\PHILIP~1\\AppData\\Local\\Temp\\tmp0.json", action=2))
        v.poll()
        assert kernel.calls["long_path"] == 1
        kernel.report(_notify("Users\\Philipp Gergen", action=4))
        v.poll()
        kernel.report(_notify("Users\\PHILIP~1\\AppData\\Local\\Temp\\tmp1.json"))
        v.poll()
        assert kernel.calls["long_path"] == 2
    finally:
        v.stop()


# -- a report the volume's thread was told of but has not run ---------------------


class _Apc(_Kernel):
    """kernel32 as it ends a read: the report is made by the writer, but
    the read shows as ended only once the thread that issued it (the
    volume's) runs in the kernel again, as an APC queued to it runs then:
    here, when it enters or leaves a wait."""

    def __init__(self) -> None:
        super().__init__()
        self.queued: tuple[bytes, int] | None = None
        self.poked = threading.Event()

    def report(self, data: bytes = b"", error: int = 0) -> None:
        self.queued = (data, error)

    def _run_apcs(self) -> bool:
        if self.queued is None:
            return False
        data, error = self.queued
        self.queued = None
        super().report(data, error)
        return True

    def new_poke(self):
        return "poke"

    def poke(self, event):
        self.poked.set()

    def wait_either(self, event, poke, ms):
        self.waited_in.add(threading.current_thread())
        if self._run_apcs():
            return 0
        poked = self.poked.wait(min(ms, 50) / 1000)
        self.poked.clear()
        if self._run_apcs():
            return 0
        return 1 if poked else None


def test_a_change_reported_before_a_poll_is_in_it_though_the_thread_had_not_run():
    kernel = _Apc()
    v = _Volume(kernel, "C:\\", "root-handle")
    v.index.add("Data\\Photos\\Anna", "anna")
    assert v.start()
    try:
        time.sleep(0.02)  # the thread is in its wait
        for _ in range(20):
            kernel.report(_notify("Data\\Photos\\Anna\\a.jpg"))  # the cell wrote a.jpg
            assert v.poll(sync=True) == (["anna"], [], False)
        assert v.counts.get("polls the thread did not answer", 0) == 0
    finally:
        v.stop()


def test_a_poll_the_thread_does_not_answer_counts_every_folder_changed(monkeypatch):
    monkeypatch.setattr(file_watch, "_SYNC_SECONDS", 0.05)
    kernel = _Apc()
    kernel.poke = lambda event: None  # the thread is never woken early
    kernel.wait_either = lambda event, poke, ms: time.sleep(0.5)
    v = _Volume(kernel, "C:\\", "root-handle")
    v.index.add("Data\\Photos\\Anna", "anna")
    v.index.add("Work", "work")
    assert v.start()
    try:
        time.sleep(0.02)
        assert v.poll() == ([], [], False)  # a poll that need not hold every change before it
        assert sorted(v.poll(sync=True)[0]) == ["anna", "work"]
    finally:
        v.stop()


class _Timed(_Apc):
    """kernel32 waiting as long as it is told: a poke ends a wait early, a
    wait that is not poked lasts its whole timeout."""

    def wait_either(self, event, poke, ms):
        self.waited_in.add(threading.current_thread())
        if self._run_apcs():
            return 0
        poked = self.poked.wait(ms / 1000) if ms else self.poked.is_set()
        self.poked.clear()
        if self._run_apcs():
            return 0
        return 1 if poked else None


def test_a_poll_is_answered_by_the_wait_after_the_one_its_poke_ended(monkeypatch):
    # The poke ends the wait that began before the poll asked, which can
    # answer only older polls; the next wait must not sit out its timeout.
    monkeypatch.setattr(file_watch, "_THREAD_WAIT_MS", 5000)
    kernel = _Timed()
    v = _Volume(kernel, "C:\\", "root-handle")
    v.index.add("Data\\Photos\\Anna", "anna")
    assert v.start()
    try:
        time.sleep(0.02)  # the thread is in its wait
        for _ in range(5):
            kernel.report(_notify("Data\\Photos\\Anna\\a.jpg"))
            assert v.poll(sync=True) == (["anna"], [], False)
        assert v.counts.get("polls the thread did not answer", 0) == 0
    finally:
        v.stop()
