"""Reads from one folder by the same frames are classified and credited once.

A notebook statement reading 24,000 small files asked for each one whether
it was a library's (`incidental_read`) and which user code to credit it to
(`credit_read_to_stack`), walking the frames above the read every time:
~25 us a file. The answer depends on the file's folder and on those frames,
so a read from the folder of the last one, by the same code below the same
anchor frame and with ``sys.path`` as it was, takes the last answer. Each
case here is answered as the full walk answers it: a package installed into
a folder put on ``sys.path`` in the middle of the loop, two functions taking
turns reading one folder, and a folder that is itself on ``sys.path``,
where a module file is a package of its own.
"""

from __future__ import annotations

import os
import sys
import sysconfig

import pytest

from cash._paths import normalize_path
from cash.install_paths import sys_path_dir
from cash.tracking import file_tracker, read_credit
from cash.tracking.file_tracker import FileAccessTracker


def _code(name, source, filename="<string>", **names):
    """A function *source* defines, run as module *name* (``cash...`` for
    cash's own frame above the reading code, ``__main__`` for a statement)
    from *filename*."""
    namespace = {"__name__": name, **names}
    exec(compile(source, filename, "exec"), namespace)
    return namespace


def _run(steps, where):
    """Run *steps* -- ``(reader, path)`` or a callable -- under a tracker,
    from below a cash frame as a notebook statement does."""
    statement = _code(
        "__main__",
        "def first(path):\n    with open(path, encoding='utf-8') as fh:\n        fh.read()\n"
        "def second(path):\n    with open(path, encoding='utf-8') as fh:\n        fh.read()\n"
        "def statement(steps):\n"
        "    for step in steps:\n"
        "        if callable(step):\n"
        "            step()\n"
        "        else:\n"
        "            globals()[step[0]](step[1])\n",
        filename=str(where / "statement.py"),
    )
    runner = _code(
        "cash.notebook.fake_runner",
        "def runner(steps):\n    with tracker:\n        statement(steps)\n",
        statement=statement["statement"],
        tracker=FileAccessTracker(),
    )
    runner["runner"](steps)
    return runner["tracker"], statement


def _both_ways(monkeypatch, steps):
    """The tracker's reads and credits, the same with every read walked in
    full, and the reads that took the last read's answer."""
    where = steps.where
    again = []
    with monkeypatch.context() as m:
        real = file_tracker.credit_read_again
        m.setattr(file_tracker, "credit_read_again", lambda *a: again.append(a[0]) or real(*a))
        fast = _run(steps(), where)
    with monkeypatch.context() as m:
        m.setattr(FileAccessTracker, "_reader_of", lambda self, abs_path: None)
        full = _run(steps(), where)
    return fast, full, again


def _credited(statement, name, files):
    reads = read_credit.credited_reads(statement[name].__code__)
    return {f for f in files if normalize_path(os.path.realpath(f)) in reads}


def _files(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_text(name, encoding="utf-8")
        paths.append(str(path))
    return paths


def _resolved(paths):
    return {normalize_path(os.path.realpath(p)) for p in paths}


@pytest.fixture
def own_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


def test_a_package_installed_mid_loop_is_a_library_from_then_on(tmp_path, monkeypatch, own_sys_path):
    def steps():
        target = tmp_path / f"target{len(list(tmp_path.iterdir()))}"
        before = _files(target / "fakepkg", ["a.txt", "b.txt", "c.txt"])
        after = _files(target / "fakepkg", ["d.txt", "e.txt", "f.txt"])

        def install():
            info = target / "fakepkg-1.0.dist-info"
            info.mkdir()
            (info / "RECORD").write_text("fakepkg/__init__.py,,\nfakepkg-1.0.dist-info/RECORD,,\n", encoding="utf-8")
            sys.path.insert(0, str(target))

        steps.files = before, after
        return [*(("first", p) for p in before), install, *(("first", p) for p in after)]

    steps.where = tmp_path
    (fast, _), (full, _), again = _both_ways(monkeypatch, steps)
    before, after = steps.files
    # c.txt took b.txt's answer (a.txt had no anchor yet); e.txt and f.txt
    # took d.txt's, a library's file credited to no one.
    assert [os.path.basename(p) for p in again] == ["c.txt"]
    assert _resolved(before) <= full.accessed_files
    assert not _resolved(after) & full.accessed_files
    assert {p for p in fast.accessed_files if "target0" in p} == {
        p.replace("target1", "target0") for p in full.accessed_files if "target1" in p
    }


def test_two_readers_taking_turns_are_each_credited_their_own_files(tmp_path, monkeypatch):
    files = _files(tmp_path / "data", [f"part{i}.txt" for i in range(8)])

    def steps():
        # first, first, second, second, first, ...
        return [("first" if i % 4 < 2 else "second", p) for i, p in enumerate(files)]

    steps.where = tmp_path

    (fast, fast_code), (full, full_code), again = _both_ways(monkeypatch, steps)
    assert len(again) == 3  # the second read of each pair after the first
    assert fast.accessed_files == full.accessed_files == _resolved(files)
    for statement in (fast_code, full_code):
        assert _credited(statement, "first", files) == {f for i, f in enumerate(files) if i % 4 < 2}
        assert _credited(statement, "second", files) == {f for i, f in enumerate(files) if i % 4 >= 2}


def test_a_folder_on_sys_path_is_classified_file_by_file(tmp_path, monkeypatch, own_sys_path):
    folder = tmp_path / "plain_target"
    files = _files(folder, ["mod_a.txt", "mod_b.txt", "mod_a.cfg", "mod_c.txt"])
    info = folder / "mod_a-1.0.dist-info"
    info.mkdir()
    (info / "RECORD").write_text("mod_a.py,,\nmod_a-1.0.dist-info/RECORD,,\n", encoding="utf-8")
    sys.path.insert(0, str(folder))

    def steps():
        return [("first", p) for p in files]

    steps.where = tmp_path
    (fast, _), (full, _), again = _both_ways(monkeypatch, steps)
    assert not again
    # ``mod_a`` is installed there; the files named for no package are the user's.
    assert fast.accessed_files == full.accessed_files == _resolved([files[1], files[3]])


def test_a_relative_sys_path_entry_names_one_folder_after_a_chdir(tmp_path, monkeypatch, own_sys_path):
    # "." on sys.path while the working folder is a ``pip install --target``
    # folder; the loop moves to another folder, and sys.path then grows.
    target, elsewhere = tmp_path / "relA", tmp_path / "relB"
    files = _files(target, ["mod_r.txt", "other.txt", "mod_r.cfg", "other2.txt"])
    _files(elsewhere, ["x.txt"])
    info = target / "mod_r-1.0.dist-info"
    info.mkdir()
    (info / "RECORD").write_text("mod_r.py,,\n", encoding="utf-8")
    monkeypatch.chdir(target)
    sys.path.insert(0, ".")
    # Each run below starts from the target folder with "." looked at there.
    assert sys_path_dir(".")

    def steps():
        def move():
            os.chdir(elsewhere)
            sys.path.append(str(tmp_path / "zzz"))

        def back():
            os.chdir(target)
            sys.path.remove(str(tmp_path / "zzz"))

        return [("first", str(elsewhere / "x.txt")), move, *(("first", p) for p in files + files), back]

    steps.where = tmp_path
    (fast, _), (full, _), _again = _both_ways(monkeypatch, steps)
    assert fast.accessed_files == full.accessed_files
    assert _resolved([files[1], files[3]]) <= full.accessed_files


def test_a_library_function_equal_to_a_user_helper_is_not_taken_for_it(tmp_path):
    files = _files(tmp_path / "eq", [f"e{i}.txt" for i in range(6)])
    source = "\n\ndef load(p):\n    with open(p, encoding='utf-8') as fh:\n        return fh.read()\n"
    helper = _code("helper", source, filename=str(tmp_path / "helper.py"))["load"]
    library = _code("eqlib", source, filename=os.path.join(sysconfig.get_paths()["purelib"], "eqlib", "__init__.py"))[
        "load"
    ]
    assert helper.__code__ == library.__code__ and helper.__code__ is not library.__code__

    _run([lambda fn=fn, p=p: fn(p) for p in files for fn in (helper, library)], tmp_path)
    assert {
        f for f in files if normalize_path(os.path.realpath(f)) in read_credit.credited_reads(helper.__code__)
    } == set(files)
