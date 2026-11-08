"""``cash info`` says whether a notebook watches folders for changes to the
files its cells read (``CASH_FILE_WATCH``, on by default), and if not, why:
the switch set to 0, or no watch on this platform."""

from __future__ import annotations

import sys

import pytest

from cash import __main__ as cli
from cash.tracking import file_watch
from tests._cli_args import cli_args
from tests._scripts import run_python


@pytest.fixture(autouse=True)
def _own_watch(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    file_watch.reset()
    yield
    file_watch.reset()


def _line(capsys) -> str:
    cli.cmd_info(cli_args("info"))
    out = capsys.readouterr().out
    return next(line for line in out.splitlines() if line.startswith("  File watch: "))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inotify")
def test_on_linux_it_is_on(capsys):
    assert _line(capsys) == "  File watch: on (inotify)"


def test_switched_off_it_names_the_switch(monkeypatch, capsys):
    monkeypatch.setenv("CASH_FILE_WATCH", "0")
    assert _line(capsys) == "  File watch: off (CASH_FILE_WATCH=0; unset it to watch folders for file changes)"


def test_where_no_folder_can_be_watched_it_says_so(monkeypatch, capsys):
    with monkeypatch.context() as m:
        m.setattr(file_watch.sys, "platform", "darwin")
        assert file_watch._platform_name() == "macOS"
    monkeypatch.setattr(file_watch, "_backend", lambda: None)
    monkeypatch.setattr(file_watch, "_platform_name", lambda: "macOS")
    assert _line(capsys) == (
        "  File watch: off: not supported on macOS; every file a cached statement read is checked before each cell"
    )


def test_a_fresh_interpreter_with_the_switch_off_says_so(tmp_path):
    done = run_python("-m", "cash", "info", cwd=tmp_path, env={"CASH_FILE_WATCH": "off"})
    assert "  File watch: off (CASH_FILE_WATCH=off; unset it to watch folders for file changes)" in (
        done.stdout.splitlines()
    ), done.stdout
