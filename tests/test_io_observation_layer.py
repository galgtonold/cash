"""cash watches I/O through one layer, and only while something needs it.

Python-level opens, directory listings, ``socket.connect`` and
``subprocess.Popen`` arrive as audit events; only readers no event reports are
wrapped, and only while a tracker or observer is open (or ``%cash_on`` is on).
Once the last scope closes the originals are back and the audit hook is inert.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap

import pytest

from cash.effect_observer import EffectObserver
from cash.tracking import io_watch
from cash.tracking.file_tracker import FileAccessTracker

# Taken at import, before any tracker opened: a library that did
# `from io import open` long before cash was imported holds this.
_CAPTURED_OPEN = io.open


def _tracked(tracker) -> set[str]:
    return {p.replace("\\", "/") for p in tracker.get_accessed_files()}


def _run_fresh(script: str) -> None:
    import cash

    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.abspath(cash.__file__)))  # the cash under test
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_patches_exist_only_while_a_scope_is_open():
    _run_fresh(
        """
        import builtins, concurrent.futures, json, os, pathlib, socket, sqlite3, subprocess, sys
        import pandas as pd

        def current():
            return {
                "read_csv": pd.read_csv,
                "connect": sqlite3.connect,
                "exists": os.path.exists,
                "stat": pathlib.Path.stat,
                "submit": concurrent.futures.ThreadPoolExecutor.submit,
            }

        def never_wrapped():
            return (builtins.open, json.load, os.listdir, os.scandir, socket.socket.connect, subprocess.Popen.__init__)

        before, plain = current(), never_wrapped()

        import cash
        from cash.effect_observer import EffectObserver
        from cash.tracking import io_watch
        from cash.tracking.file_tracker import FileAccessTracker, _shared_import_hook

        with FileAccessTracker():
            with EffectObserver():
                pass  # an inner scope closing must not uninstall anything
            during = current()
            assert all(during[k] is not before[k] for k in before), during
            assert _shared_import_hook in sys.meta_path
            assert never_wrapped() == plain

        assert current() == before
        assert never_wrapped() == plain
        assert _shared_import_hook not in sys.meta_path
        assert io_watch._active == {}, io_watch._active  # the audit hook is inert

        @cash.cache
        def f():
            return 1

        # A decorated function watches opens outside calls (a memo filled
        # before the first call), and nothing else.
        assert set(io_watch._active) == {"open"}, io_watch._active
        assert current() == before
        """
    )


def test_the_next_scope_wraps_what_is_there_now():
    """Scopes reinstall the same wrappers cheaply, but never over a change."""
    _run_fresh(
        """
        import sqlite3, sys, tempfile, types
        import cash
        from cash.tracking.file_tracker import FileAccessTracker

        with FileAccessTracker():
            pass

        def replacement(*args, **kwargs):
            return "fake"

        sqlite3.connect = replacement  # a test's monkeypatch, between two calls
        with FileAccessTracker():
            assert sqlite3.connect._original_func is replacement
        assert sqlite3.connect is replacement

        # A reader module imported between two scopes is wrapped by the next.
        late = types.ModuleType("late_reader")
        late.read_blob = lambda path: path
        cash.Cash(register_magic=False).register_file_handler(
            "late_reader", "read_blob",
            lambda original, track: (lambda path: (track(path), original(path))[1]),
        )
        sys.modules["late_reader"] = late
        blob = tempfile.NamedTemporaryFile(suffix=".blob", delete=False)
        blob.close()
        with FileAccessTracker() as tracker:
            assert getattr(late.read_blob, "_is_file_tracker_patch", False)
            late.read_blob(blob.name)
        assert not getattr(late.read_blob, "_is_file_tracker_patch", False)
        assert tracker.get_accessed_files(), "the late reader's read was not recorded"
        """
    )


def test_a_read_through_a_reference_taken_before_cash_is_tracked(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("x")
    tracker = FileAccessTracker()
    with tracker, _CAPTURED_OPEN(data) as fh:
        fh.read()
    assert any(p.endswith("/data.txt") for p in _tracked(tracker))


def test_importing_a_user_module_records_neither_its_source_nor_its_folder(tmp_path, monkeypatch):
    (tmp_path / "freshly_written_module.py").write_text("VALUE = 7\n")
    (tmp_path / "rows.txt").write_text("1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    tracker = FileAccessTracker()
    try:
        with tracker:
            import freshly_written_module  # noqa: F401

            imported = _tracked(tracker)
            with open(tmp_path / "rows.txt") as fh:  # control: a read here IS tracked
                fh.read()
    finally:
        sys.modules.pop("freshly_written_module", None)
    root = str(tmp_path).replace("\\", "/")
    assert not [p for p in imported if p.startswith(root) or root.startswith(p)], imported
    assert any(p.endswith("/rows.txt") for p in _tracked(tracker))


def test_a_missing_file_opened_for_reading_is_an_absence_but_not_for_append(tmp_path):
    tracker = FileAccessTracker()
    with tracker:
        with pytest.raises(FileNotFoundError):
            open(tmp_path / "gone.txt")
        with open(tmp_path / "log.txt", "a+") as fh:  # creates it: no absence
            fh.write("x")
    absent = {p.replace("\\", "/") for p in tracker.get_absent_files()}
    assert any(p.endswith("/gone.txt") for p in absent), absent
    assert not any(p.endswith("/log.txt") for p in absent), absent


@pytest.mark.parametrize("reader", ["read_code_file", "file_content_hash", "has_main_guard"])
def test_cash_reading_a_file_for_itself_is_nobodys_dependency(tmp_path, reader):
    """The audit event reports ``io.FileIO`` too, which these reads once used
    because it was not patched: a module cash read to key the code that runs
    became a raw-bytes input, and a comment added to it re-ran the work."""
    from cash import core, source_norm
    from cash.tracking import file_dep_snapshot

    module = tmp_path / "helpers.py"
    module.write_text("def scale(x):\n    return x * 2\n")
    read = {
        "read_code_file": source_norm.read_code_file,
        "file_content_hash": file_dep_snapshot.file_content_hash,
        "has_main_guard": core._has_main_guard,
    }[reader]
    tracker = FileAccessTracker()
    with tracker:
        read(str(module))
    assert not any(p.endswith("/helpers.py") for p in _tracked(tracker)), _tracked(tracker)


def test_os_open_is_not_a_dependency(tmp_path):
    """Tests count executions with os.open/os.write precisely because the
    tracker does not turn them into dependencies."""
    counter = tmp_path / "count.txt"
    counter.write_text("0")
    tracker = FileAccessTracker()
    with tracker:
        os.close(os.open(counter, os.O_RDONLY))
    assert not any(p.endswith("/count.txt") for p in _tracked(tracker))


def test_a_spawned_process_is_observed():
    observer = EffectObserver()
    with observer:
        subprocess.run([sys.executable, "-c", "pass"], check=True)
    assert any(kind == "subprocess" and detail.startswith("spawned ") for kind, detail in observer.effects), (
        observer.effects
    )


@pytest.mark.skipif(os.name == "nt", reason="the POSIX shell form")
def test_a_shell_command_is_described_by_its_command_not_the_shell():
    observer = EffectObserver()
    with observer:
        subprocess.run("exit 0", shell=True, check=True)
    details = [detail for kind, detail in observer.effects if kind == "subprocess"]
    assert details and details[0].startswith("spawned exit 0"), details


def test_cash_on_holds_the_watch_until_cash_off(cash_magics):
    held = io_watch._holds
    cash_magics.cash_on("")
    try:
        assert io_watch._holds == held + 1
        cash_magics.cash_on("")  # a second %cash_on holds nothing more
        assert io_watch._holds == held + 1
    finally:
        cash_magics.cash_off("")
    assert io_watch._holds == held


def test_the_shells_own_open_is_not_a_notebook_input(mock_shell):
    """cash no longer wraps ``user_ns['open']``, so what a ``def`` that writes a
    log line reads there is IPython's ``open``, which cannot be pickled: hashed
    as a value it gave the statement a per-kernel key, and nothing restored
    after a restart."""
    import IPython.core.interactiveshell as shell_module

    from cash.notebook.cache_key import is_cash_instrumentation

    assert is_cash_instrumentation(shell_module._modified_open)

    def users_own_open(path):
        return path

    assert not is_cash_instrumentation(users_own_open)
    assert not is_cash_instrumentation(open)
