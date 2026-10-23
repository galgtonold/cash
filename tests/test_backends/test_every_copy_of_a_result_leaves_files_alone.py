"""No copy cash makes of a result changes a file, and checking for one
neither hangs nor changes what the rest of the process does.

- The store's copy was checked, but not a hit's copy, nor the ``deepcopy``
  a value pickle refuses falls back to (a class defined in a function): a
  ``__setstate__`` reopening its sink with ``"w"`` emptied the file on every
  call.
- With ``tempfile.tempdir`` unset, a refused ``mkdir`` or exclusive create
  asked ``tempfile.gettempdir()`` from inside the check, while that function
  held its own lock to probe the folder: the process hung.
- ``sys.dont_write_bytecode`` was set for the whole process while a value
  was rebuilt: another thread's import lost its ``.pyc``, and a ``True``
  the program set meanwhile was put back to ``False``.
- A module whose first import writes its own cache (matplotlib's font list)
  was imported inside the check, refused, and a disk hit became a miss.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading

import pytest

import cash
from cash.backends.memory_backend import InMemoryBackend

SRC = os.path.join(os.path.dirname(cash.__file__), os.pardir)


@pytest.fixture
def sink(tmp_path):
    path = tmp_path / "sink.txt"
    path.write_text("keep me", encoding="utf-8")
    return path


def _local_sink_class():
    class Sink:  # in a function: pickle cannot find it by name
        def __init__(self, path):
            self.path = path

        def __getstate__(self):
            return {"path": self.path}

        def __setstate__(self, state):
            self.path = state["path"]
            with open(self.path, "w", encoding="utf-8"):
                pass

    return Sink


def test_a_value_copied_by_deepcopy_leaves_the_file_alone(tmp_path, sink):
    Sink = _local_sink_class()
    c = cash.Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)

    @c.cache
    def build(n):
        return {"sink": Sink(str(sink)), "n": n}

    seen = []
    for _ in range(3):
        sink.write_text("keep me", encoding="utf-8")
        build(1)
        seen.append(sink.read_text(encoding="utf-8"))
    assert seen == ["keep me"] * 3


class _WritesLater:
    """Rebuilt without a write when stored; with one once ``armed``."""

    armed = False

    def __init__(self, path):
        self.path = path

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        if _WritesLater.armed:
            with open(self.path, "w", encoding="utf-8"):
                pass


@pytest.fixture
def writes_later():
    yield _WritesLater
    _WritesLater.armed = False


def test_a_hit_whose_copy_would_write_is_a_miss(sink, writes_later):
    backend = InMemoryBackend()
    backend.set("k", {"v": writes_later(str(sink))}, metadata={"copy_required": True})
    writes_later.armed = True
    assert backend.get("k") == (None, None)
    assert sink.read_text(encoding="utf-8") == "keep me"
    writes_later.armed = False
    assert backend.get("k") == (None, None)  # dropped, not copied the same way again


def _run(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": os.path.abspath(SRC), **(env or {})},
    )


@pytest.mark.parametrize("make", ["tempfile.mkdtemp()", "os.mkdir(os.path.join(os.getcwd(), 'made'))"])
def test_a_refused_or_scratch_change_does_not_hang_without_a_temp_folder_set(tmp_path, make):
    done = _run(
        f"""
        import os, tempfile
        assert tempfile.tempdir is None
        from cash.backends.memory_backend import InMemoryBackend
        class M:
            def __getstate__(self):
                return {{}}
            def __setstate__(self, state):
                {make}
        try:
            InMemoryBackend().set("k", {{"v": M()}}, metadata={{"copy_required": True}})
            print("stored")
        except Exception as exc:
            print("refused", type(exc).__name__)
        """,
        env={"TMPDIR": str(tmp_path)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[0] in ("stored", "refused")


class _Waits:
    """Rebuilt while another thread works."""

    entered = threading.Event()
    go_on = threading.Event()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        _Waits.entered.set()
        _Waits.go_on.wait(30)


def test_a_rebuild_leaves_other_threads_and_the_programs_setting_alone(tmp_path, monkeypatch):
    (tmp_path / "fresh_on_another_thread.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fresh_on_another_thread", raising=False)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    _Waits.entered.clear()
    _Waits.go_on.clear()
    backend = InMemoryBackend()
    storing = threading.Thread(target=backend.set, args=("k", {"v": _Waits()}), kwargs={"metadata": {}})
    storing.start()
    try:
        assert _Waits.entered.wait(30)
        import fresh_on_another_thread  # noqa: F401 - imported while the other thread rebuilds

        sys.dont_write_bytecode = True  # the program's own, set meanwhile
    finally:
        _Waits.go_on.set()
        storing.join(30)
    assert sys.dont_write_bytecode is True
    assert any(name.startswith("fresh_on_another_thread.") for name in os.listdir(tmp_path / "__pycache__"))


def test_a_module_whose_first_import_writes_its_cache_is_served_from_disk(tmp_path):
    """Imported as the program would import it, then rebuilt again."""
    library = tmp_path / "lib" / "site-packages"  # a library's, not the user's: the key does not import it
    library.mkdir(parents=True)
    (library / "caches_on_import.py").write_text(
        textwrap.dedent(
            """
            import os
            fonts = os.path.join(os.environ["CACHES_AT"], "fonts.json")
            if not os.path.exists(fonts):
                with open(fonts, "w") as f:
                    f.write("[]")
            class Figure:
                def __init__(self, n):
                    self.n = n
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "maker.py").write_text(
        "def make(n):\n    import caches_on_import\n    return caches_on_import.Figure(n)\n", encoding="utf-8"
    )
    script = f"""
        import sys, time
        sys.path[:0] = [{str(tmp_path)!r}, {str(library)!r}]
        import cash, maker
        c = cash.Cash(cache_dir={str(tmp_path / "cache")!r}, register_magic=False)
        ran = []
        @c.cache
        def figure(n):
            ran.append(n)
            time.sleep(1.0)
            return maker.make(n)
        imported = "caches_on_import" in sys.modules
        n = figure(3).n
        print("ran", len(ran), n, imported)
        """
    caches = tmp_path / "caches"
    caches.mkdir()
    env = {**os.environ, "PYTHONPATH": os.path.abspath(SRC), "CACHES_AT": str(caches)}
    for expected in (["ran", "1", "3", "False"], ["ran", "0", "3", "False"]):
        (caches / "fonts.json").unlink(missing_ok=True)  # as a fresh MPLCONFIGDIR
        done = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, timeout=120, env=env
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == expected
        assert (caches / "fonts.json").exists()
