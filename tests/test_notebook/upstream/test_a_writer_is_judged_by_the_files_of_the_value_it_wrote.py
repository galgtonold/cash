"""A writer is judged by the files of the value it wrote, not of a later one.

A write of what was read from the same file (``n = int(open(P).read())``,
then ``open(P, 'w').write(str(n + 1))``) is never re-fired: it moves the
file on each time it runs. The scheduler tells it by the files the
writer's inputs were read from (``executed_file_deps``), which hold each
name's last binding. ``df.to_csv(P)`` followed by ``df = pd.read_csv(P)``
gave ``df`` the very file the writer writes, so the writer looked like such
a counter and was never re-fired: an edit of a review file the frame was
read from never reached the CSV, nor the ``df`` read back from it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream.file_writers import FileWriterScheduler

WRITE = "df.to_csv('movie_data.csv', index=False)"


def _scheduler(tmp_path, monkeypatch, deps: dict[str, set[str]], at_write: dict[str, set[str]]) -> FileWriterScheduler:
    """A scheduler whose runtime holds *deps* (each name's last binding),
    and *at_write*: what each writer's inputs were read from as it ran."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "movie_data.csv").write_text("review\n", encoding="utf-8")
    state = TrackingState()
    state.executed_file_deps.update(deps)
    state.executed_write_input_files.update({code: frozenset(files) for code, files in at_write.items()})
    shell = MagicMock()
    shell.user_ns = {}
    return FileWriterScheduler(shell, state, MagicMock())


def _reviews(tmp_path) -> set[str]:
    return {str(tmp_path / "aclImdb" / f"{i}_7.txt") for i in range(3)}


def test_a_name_read_back_from_the_written_file_below_is_not_what_the_writer_read(tmp_path, monkeypatch):
    csv = {"movie_data.csv", str(tmp_path / "movie_data.csv")}
    scheduler = _scheduler(tmp_path, monkeypatch, {"df": csv}, {WRITE: _reviews(tmp_path)})

    assert not scheduler._rewrites_what_it_read(WRITE, {"df"}, [])


def test_a_counter_read_again_below_still_writes_back_what_it_read(tmp_path, monkeypatch):
    """Each step reads the counter afresh: the name is bound again below
    each writer, and still read from the file it writes."""
    write = "open('movie_data.csv', 'w').write(str(n + 1))"
    scheduler = _scheduler(tmp_path, monkeypatch, {"n": {"movie_data.csv"}}, {write: {"movie_data.csv"}})

    assert scheduler._rewrites_what_it_read(write, {"n"}, [])


def test_a_writer_not_seen_running_is_judged_by_the_files_held_now(tmp_path, monkeypatch):
    """After a restart nothing was seen writing: the runtime's record is
    all there is, as before."""
    write = "open('movie_data.csv', 'w').write(str(n + 1))"
    scheduler = _scheduler(tmp_path, monkeypatch, {"n": {"movie_data.csv"}}, {})

    assert scheduler._rewrites_what_it_read(write, {"n"}, [])
