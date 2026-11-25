"""A callable's source text is joined once per file version, and read again
whenever the file, or what ``linecache`` holds for it, changes.

`source_reading.getsource` answers from the memo of the lines
(`source_reading._SOURCE_LINES`), but every read copied those lines out of
it, joined them and checked again that the block defines what it says: the
key and the analysis of a fresh process read the same helper's text a dozen
times (its key digest, its report-store guard, the names its signature and
decorators read, the globals its body may change), 3,500 reads over the
first calls of nilearn's cached functions in one process, of which 300 ask
for a block no read before them had asked for.

The text is kept under the same key as the lines, so what is pinned here is
that nothing is kept a change could outlive: an edited file is read again
(the key holds the file's version), and a file ``linecache`` holds in
memory -- a notebook cell, ``exec``ed source -- keeps nothing at all,
because its entry is replaced without the file's version moving.
"""

from __future__ import annotations

import inspect
import linecache
import textwrap

import pytest

from cash import source_reading

pytestmark = pytest.mark.core

_MODULE = """
TABLE = [1, 2, 3]


def helper(k):
    # a comment inside the block
    return TABLE[k]


class Box:
    def size(self):
        return len(TABLE)


def top(k):
    return helper(k) + Box().size()
"""


@pytest.fixture
def reads(monkeypatch):
    """How many reads went the whole way: `block_defines` runs on each one
    that cut, copied and joined the block, and on no served text."""
    counted: list = []
    real = source_reading.block_defines
    monkeypatch.setattr(source_reading, "block_defines", lambda lines, code: counted.append(code) or real(lines, code))
    monkeypatch.setattr(source_reading, "SETTLED_SECONDS", -1.0)  # every file has settled
    return counted


@pytest.fixture
def module(tmp_path, monkeypatch):
    """A module on disk, imported fresh, whose file the test may rewrite."""
    import importlib
    import sys

    name = "textmemo_" + tmp_path.name.replace("-", "_")
    path = tmp_path / f"{name}.py"
    path.write_text(_MODULE, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(name), path
    sys.modules.pop(name, None)


def test_reading_one_text_many_times_cuts_it_once(module, reads):
    mod, _path = module
    first = source_reading.getsource(mod.helper)
    assert first == inspect.getsource(mod.helper)
    for _ in range(20):
        assert source_reading.getsource(mod.helper) == first
    assert len(reads) == 1, f"{len(reads)} reads went the whole way, not one"


def test_every_shape_of_callable_reads_what_inspect_reads(module, reads):
    mod, _path = module
    for obj in (mod.helper, mod.top, mod.Box, mod.Box.size, mod.top.__code__):
        for _ in range(3):
            assert source_reading.getsource(obj) == inspect.getsource(obj)


def test_an_edited_file_is_read_again(module, reads):
    mod, path = module
    before = source_reading.getsource(mod.helper)
    grown = _MODULE.replace("    return TABLE[k]\n", "    k = k + 1\n    return TABLE[k] * 2\n")
    path.write_text(grown, encoding="utf-8", newline="\n")
    linecache.checkcache(str(path))
    after = source_reading.getsource(mod.helper)
    assert after != before
    assert after.endswith("    return TABLE[k] * 2\n")
    assert after == inspect.getsource(mod.helper)
    assert len(reads) > 1, "the edited file was served from the memo"


def test_text_held_in_linecache_is_read_again_when_it_is_replaced(reads):
    """A cell's source is replaced in ``linecache`` under the same name,
    with no file to take a version from: nothing may be kept for it."""
    name = "<cash-text-memo-cell>"
    first = "def cell(x):\n    return x + 1\n"
    second = "def cell(x):\n    return x + 2\n"
    scope: dict = {}
    exec(compile(first, name, "exec"), scope)  # a cell, as IPython runs one
    linecache.cache[name] = (len(first), None, first.splitlines(keepends=True), name)
    try:
        assert source_reading.getsource(scope["cell"]).strip().endswith("return x + 1")
        exec(compile(second, name, "exec"), scope)
        linecache.cache[name] = (len(second), None, second.splitlines(keepends=True), name)
        assert source_reading.getsource(scope["cell"]).strip().endswith("return x + 2")
    finally:
        linecache.cache.pop(name, None)


def test_a_wrapper_keeps_its_own_text(module, reads):
    """``own_source`` of a ``functools.wraps`` wrapper is the wrapper's own
    block, however often it is read (the wrapped function's text is its
    own)."""
    import functools

    mod, _path = module

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            return fn(*a, **k) + 100

        return wrapper

    wrapped = decorate(mod.helper)
    for _ in range(3):
        own = source_reading.own_source(wrapped)
        assert "return fn(*a, **k) + 100" in own
        assert source_reading.own_source(mod.helper) == inspect.getsource(mod.helper)


def test_code_compiled_from_other_text_than_its_file_is_not_kept(reads, tmp_path, monkeypatch):
    """Code whose file holds something else at its line (``%autoreload``
    compiles an edited function alone) is found where it is defined, or not
    at all: either way nothing is kept under the file's version, and the
    answer follows the file."""
    path = tmp_path / "relocated.py"
    path.write_text("A = 1\nB = 2\n", encoding="utf-8", newline="\n")
    scope: dict = {}
    exec(compile("def gone(x):\n    return x\n", str(path), "exec"), scope)
    for _ in range(3):
        with pytest.raises(OSError):
            source_reading.getsource(scope["gone"])


def test_a_dedented_block_is_what_the_text_store_digests(module, reads):
    """The text served is the exact block, so every digest built from it
    (`cash.code_digest.callable_identity`) is the one a fresh read gives."""
    from cash.code_digest import callable_identity

    mod, path = module
    before = callable_identity(mod.helper)
    assert before == callable_identity(mod.helper)
    path.write_text(_MODULE.replace("TABLE[k]", "TABLE[k] * 3"), encoding="utf-8", newline="\n")
    linecache.checkcache(str(path))
    assert callable_identity(mod.helper) != before


def test_the_text_is_not_kept_while_the_file_is_still_being_written(module, monkeypatch):
    """A file whose last write is too recent to have settled keeps nothing:
    `settled_source_version` gives no version, and the text is read again."""
    mod, _path = module
    counted: list = []
    real = source_reading.block_defines
    monkeypatch.setattr(source_reading, "block_defines", lambda lines, code: counted.append(code) or real(lines, code))
    monkeypatch.setattr(source_reading, "SETTLED_SECONDS", 1e9)  # nothing has settled
    for _ in range(3):
        assert source_reading.getsource(mod.helper) == inspect.getsource(mod.helper)
    assert len(counted) == 3, "an unsettled file's text was served from the memo"


def test_the_memo_holds_one_entry_per_object_and_version(module, reads):
    """What is kept is keyed by the file's version and the object's place in
    it: an edit leaves the old entry unread rather than serving it."""
    mod, path = module
    source_reading.getsource(mod.helper)
    keys_before = {k for k in source_reading._SOURCE_TEXT.keys() if str(path) in str(k)}
    assert keys_before, "nothing was kept for the module's file"
    path.write_text(textwrap.dedent(_MODULE).replace("TABLE[k]", "TABLE[k] + 7"), encoding="utf-8", newline="\n")
    linecache.checkcache(str(path))
    assert "TABLE[k] + 7" in source_reading.getsource(mod.helper)
