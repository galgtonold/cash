"""A method edited at the same size, moments after a run or with its mtime put
back, is the new method: a cached call that runs it is not served the old one.

``linecache`` (where ``inspect`` reads source) keeps a file's lines while its
mtime and size are unchanged, and cash's memos of what it read from a file
were keyed the same way. A same-size edit (``x * 1`` -> ``x * 2``) that kept
both -- ``os.utime``, ``cp -p``, ``tar`` and ``rsync -t`` put the mtime back --
left a reloaded module running the new code while cash keyed it by the old
text: the old code's result was served. What plain Python computes decides.
"""

from __future__ import annotations

import importlib
import inspect
import linecache
import os
import sys
import textwrap
import time
import types

import pytest

from cash import code_digest, loaded_code, source_reading
from cash.tracking import module_symbols
from tests._scripts import run_python

pytestmark = pytest.mark.core

SHAPES = textwrap.dedent("""
    import cash

    class Base:
        def scale(self, x):
            return x * 1 + self.k

    class Model(Base):
        def __init__(self, k):
            self.k = k

    @cash.cache
    def in_module(x):
        return Model(5).scale(x)
""").lstrip()

# One process: run, edit shapes.py at the same size, reload it, run again --
# the cached function defined in the module, and one defined beside it.
IN_PROCESS = textwrap.dedent("""
    import importlib, os, sys, time
    import cash
    import shapes

    @cash.cache
    def beside(x):
        return shapes.Model(5).scale(x)

    print(shapes.in_module(10), beside(10))
    st = os.stat("shapes.py")
    with open("shapes.py", encoding="utf-8") as fh:
        text = fh.read()
    with open("shapes.py", "w", encoding="utf-8") as fh:
        fh.write(text.replace("x * 1", "x * 2"))
    if PUT_BACK:
        os.utime("shapes.py", ns=(st.st_atime_ns, st.st_mtime_ns))
    importlib.reload(shapes)
    print(shapes.in_module(10), beside(10))
""")

ACROSS_PROCESSES = textwrap.dedent("""
    import cash
    from shapes import Model

    @cash.cache
    def run(x):
        return Model(5).scale(x)

    print(run(10))
""")


def _project(tmp_path, main: str, settled: bool):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "shapes.py").write_text(SHAPES, encoding="utf-8")
    (proj / "main.py").write_text(main, encoding="utf-8")
    if settled:
        old = time.time() - 100
        os.utime(proj / "shapes.py", (old, old))
        # The change time cannot be put back: wait until it has settled
        # too, so the first run memoises what it read from the file.
        time.sleep(source_reading.SETTLED_SECONDS + 0.2)
    return proj


def _out(proj, cache, disable=False):
    env = {"CASH_DISABLE": "1"} if disable else None
    return run_python("main.py", cwd=proj, cache_dir=cache, env=env, timeout=120).stdout.split()


@pytest.mark.parametrize("settled", [False, True], ids=["just-saved", "settled"])
@pytest.mark.parametrize("put_back", [False, True], ids=["new-mtime", "mtime-put-back"])
def test_a_reloaded_module_s_same_size_edit_runs_the_new_method(tmp_path, put_back, settled):
    main = f"PUT_BACK = {put_back}\n" + IN_PROCESS
    proj = _project(tmp_path, main, settled)
    cached = _out(proj, tmp_path / "cache")
    (proj / "shapes.py").write_text(SHAPES, encoding="utf-8")
    plain = _out(proj, tmp_path / "cache", disable=True)
    assert plain == ["15", "15", "25", "25"]
    assert cached == plain, "the reloaded module's edited method was served the old method's result"


@pytest.mark.parametrize("settled", [False, True], ids=["just-saved", "settled"])
@pytest.mark.parametrize("put_back", [False, True], ids=["new-mtime", "mtime-put-back"])
def test_a_same_size_edit_between_runs_runs_the_new_method(tmp_path, put_back, settled):
    proj = _project(tmp_path, ACROSS_PROCESSES, settled)
    path = proj / "shapes.py"
    assert _out(proj, tmp_path / "cache") == ["15"]
    st = os.stat(path)
    path.write_text(SHAPES.replace("x * 1", "x * 2"), encoding="utf-8")
    if put_back:
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    plain = _out(proj, tmp_path / "cache", disable=True)
    assert plain == ["25"]
    assert _out(proj, tmp_path / "cache") == plain


@pytest.fixture
def settled_module(tmp_path, monkeypatch):
    """``(module, path)``: a module whose file was saved a minute ago, its
    mtime put back and the memos' clock (`stat_has_settled`) run forward."""
    monkeypatch.syspath_prepend(str(tmp_path))
    name = f"_same_size_{tmp_path.name}"
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    path = tmp_path / f"{name}.py"
    path.write_text(SHAPES, encoding="utf-8")
    old = time.time() - 60
    os.utime(path, (old, old))
    monkeypatch.setattr(source_reading, "time", types.SimpleNamespace(time=lambda: time.time() + 60))
    return importlib.import_module(name), path


def _rewrite_putting_the_mtime_back(path):
    st = os.stat(path)
    time.sleep(0.05)  # past the change time's tick (coarse on Linux)
    path.write_text(SHAPES.replace("x * 1", "x * 2"), encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(path).st_size == st.st_size


def test_source_read_after_the_edit_is_the_new_text(settled_module):
    """``inspect`` alone answers the old text here; cash's reads do not."""
    module, path = settled_module
    assert "x * 1" in source_reading.getsource(module.Base.scale)
    assert "x * 1" in source_reading.getsource(module.Base)

    _rewrite_putting_the_mtime_back(path)
    module = importlib.reload(module)

    assert "x * 2" in source_reading.getsource(module.Base.scale)
    assert "x * 2" in source_reading.getsource(module.Base)
    assert "x * 2" in inspect.getsource(module.Base.scale)


@pytest.mark.parametrize(
    "read",
    [
        code_digest.module_identity,
        lambda path: loaded_code._compiled_module_and_lines(path)[1],
        lambda path: module_symbols.closure_digest(path, ["Base"]),
    ],
    ids=["module-identity", "compiled-module", "closure-digest"],
)
def test_a_memo_of_a_file_is_not_kept_across_the_edit(settled_module, read):
    _module, path = settled_module
    before = read(str(path))
    assert read(str(path)) == before
    _rewrite_putting_the_mtime_back(path)
    assert read(str(path)) != before


def test_a_settled_file_s_linecache_entry_is_kept(settled_module):
    """The fast path: a file left alone is not read again for each question."""
    module, path = settled_module

    source_reading.getsource(module.Base.scale)
    entry = linecache.cache.get(str(path))
    assert entry is not None
    for _ in range(3):
        source_reading.getsource(module.Base.scale)
        source_reading.getsource(module.Model)
        assert linecache.cache.get(str(path)) is entry
