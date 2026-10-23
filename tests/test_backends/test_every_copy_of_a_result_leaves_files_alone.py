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
- Allowed for ``importlib``'s ``.pyc``, a remove or a rename onto any
  ``__pycache__/*.pyc`` went through from the value's own code too; and
  imported outside the check for that retry, a module of the user's wrote
  what its import writes, where the program never imported it.
- The ``.pyc`` steps were let through for any function of their name in a
  file whose path held "importlib": the value's own call of
  ``importlib._bootstrap_external._write_atomic``, or of a ``_write_atomic``
  of its own, overwrote a ``.pyc`` that was there. Only an import's write,
  below ``get_code`` caching the bytecode, goes through now.
- A user's portion of a namespace package was judged by the package's first
  folder: with an installed portion earlier on ``sys.path``, the user's
  ``acmex.mine`` counted as a library and was imported for the retry.
"""

from __future__ import annotations

import os
import sys
import textwrap
import threading

import pytest

import cash
from cash.backends.memory_backend import InMemoryBackend
from tests._scripts import run_python


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


def _run(code: str, cwd, env: dict | None = None):
    return run_python("-c", textwrap.dedent(code), cwd=cwd, env=env, timeout=60, check=False)


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
        cwd=tmp_path,
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
    for expected in (["ran", "1", "3", "False"], ["ran", "0", "3", "False"]):
        (caches / "fonts.json").unlink(missing_ok=True)  # as a fresh MPLCONFIGDIR
        done = run_python("-c", textwrap.dedent(script), cwd=tmp_path, env={"CACHES_AT": str(caches)})
        assert done.stdout.split() == expected
        assert (caches / "fonts.json").exists()


class _Clobbers:
    """Rebuilt with a change to a ``.pyc`` that was there before, made the
    way ``importlib`` makes its own."""

    how = ""
    #: A ``_write_atomic`` of the user's, in a folder named like importlib's.
    fake = None

    def __init__(self, path):
        self.path = path

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        if _Clobbers.how == "remove":
            os.remove(self.path)
        elif _Clobbers.how == "replace":
            temporary = f"{self.path}.{id(self)}"
            with open(temporary, "x", encoding="utf-8") as f:
                f.write("clobbered")
            os.replace(temporary, self.path)
        elif _Clobbers.how == "write_atomic":
            import importlib._bootstrap_external as bootstrap

            bootstrap._write_atomic(self.path, b"clobbered")
        elif _Clobbers.how == "a_function_of_that_name":
            _Clobbers.fake(self.path, b"clobbered")


@pytest.mark.parametrize("how", ["remove", "replace", "write_atomic", "a_function_of_that_name"])
def test_a_pyc_that_was_there_is_left_alone(tmp_path, how, monkeypatch):
    tools = tmp_path / "myimportlib_tools"
    tools.mkdir()
    (tools / "fake_atomic.py").write_text(
        textwrap.dedent(
            """
            import os
            def _write_atomic(path, data):
                temporary = f"{path}.{id(path)}"
                with open(temporary, "xb") as f:
                    f.write(data)
                os.replace(temporary, path)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tools))
    import fake_atomic

    monkeypatch.setattr(_Clobbers, "fake", staticmethod(fake_atomic._write_atomic))
    folder = tmp_path / "pkg" / "__pycache__"
    folder.mkdir(parents=True)
    kept = folder / "keep.cpython-312.pyc"
    kept.write_text("user data", encoding="utf-8")
    backend = InMemoryBackend()
    _Clobbers.how = how
    try:
        backend.set("k", {"v": _Clobbers(str(kept))}, metadata={"copy_required": True})
    except Exception:
        pass
    finally:
        _Clobbers.how = ""
    assert kept.read_text(encoding="utf-8") == "user data"
    assert backend.get("k") == (None, None)


def test_a_users_module_whose_import_writes_is_not_imported_for_a_retry(tmp_path):
    """The first process imports the module itself and stores the value;
    the second, which never imports it, rebuilds the value from disk."""
    side = tmp_path / "side.txt"
    (tmp_path / "writes_on_import.py").write_text(f"open({str(side)!r}, 'w').write('side ran')\n", encoding="utf-8")
    (tmp_path / "sink.py").write_text(
        textwrap.dedent(
            """
            class Sink:
                def __init__(self, module):
                    self.module = module
                def __getstate__(self):
                    return {"module": self.module}
                def __setstate__(self, state):
                    self.module = state["module"]
                    __import__(self.module)
            """
        ),
        encoding="utf-8",
    )
    script = f"""
        import sys, time
        sys.path.insert(0, {str(tmp_path)!r})
        if sys.argv[1] == "first":
            import writes_on_import
            open({str(side)!r}, "w").write("orig")
        import cash, sink
        c = cash.Cash(cache_dir={str(tmp_path / "cache")!r}, register_magic=False)
        ran = []
        @c.cache
        def build(n):
            ran.append(n)  # @cash:assume-safe
            time.sleep(1.0)
            return {{"s": sink.Sink("writes_on_import"), "n": n}}
        build(1)
        print(len(ran), "writes_on_import" in sys.modules, open({str(side)!r}).read())
        """
    first = run_python("-c", textwrap.dedent(script), "first", cwd=tmp_path)
    assert first.stdout.split() == ["1", "True", "orig"]
    second = run_python("-c", textwrap.dedent(script), "second", cwd=tmp_path)
    assert second.stdout.split() == ["1", "False", "orig"]


def test_a_users_portion_of_a_namespace_package_is_not_imported_for_a_retry(tmp_path):
    """``acmex``'s installed portion comes first on ``sys.path``; the user's
    ``src/acmex/mine``, appended as notebooks often do, writes on import."""
    side = tmp_path / "side.txt"
    installed = tmp_path / "venvlike" / "site-packages"
    (installed / "acmex" / "core").mkdir(parents=True)
    (installed / "acmex" / "core" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "acmex" / "mine").mkdir(parents=True)
    (tmp_path / "src" / "acmex" / "mine" / "__init__.py").write_text(
        f"open({str(side)!r}, 'w').write('side ran')\n", encoding="utf-8"
    )
    (tmp_path / "sink.py").write_text(
        textwrap.dedent(
            """
            class Sink:
                def __init__(self, module):
                    self.module = module
                def __getstate__(self):
                    return {"module": self.module}
                def __setstate__(self, state):
                    self.module = state["module"]
                    __import__(self.module)
            """
        ),
        encoding="utf-8",
    )
    script = f"""
        import site, sys, time
        sys.dont_write_bytecode = True
        sys.path.insert(0, {str(tmp_path)!r})
        sys.path.insert(1, {str(installed)!r})
        sys.path.append({str(tmp_path / "src")!r})
        if sys.argv[1] == "first":
            import acmex.mine
        open({str(side)!r}, "w").write("orig")
        import cash, sink
        c = cash.Cash(cache_dir={str(tmp_path / "cache")!r}, register_magic=False)
        ran = []
        @c.cache
        def build(n):
            ran.append(n)  # @cash:assume-safe
            time.sleep(1.0)
            return {{"s": sink.Sink("acmex.mine"), "n": n}}
        build(1)
        print(len(ran), "acmex.mine" in sys.modules, open({str(side)!r}).read())
        """
    first = run_python("-c", textwrap.dedent(script), "first", cwd=tmp_path)
    assert first.stdout.split() == ["1", "True", "orig"], first.stderr[-2000:]
    second = run_python("-c", textwrap.dedent(script), "second", cwd=tmp_path)
    assert second.stdout.split() == ["1", "False", "orig"], second.stderr[-2000:]
