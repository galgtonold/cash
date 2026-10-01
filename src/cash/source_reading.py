"""Reading the code that runs: source files, and a function's or class's
source lines, read where no file tracker sees them and memoised per file
version.

`getsource` and `getsourcelines` are ``inspect``'s, read once per object and
file version; `own_source` does not follow a wrapper's ``__wrapped__``.
`read_code_file` reads a module's bytes for cash's own use, untracked.
Every memo here is keyed on a file's ``(mtime, size)`` and only once the file
has settled (`stat_has_settled`): a same-size edit inside one mtime tick
keeps that stat.
"""

from __future__ import annotations

import ast
import inspect
import io
import linecache
import os
import sys
import time
import types

from ._memo import CODE_OBJECTS, LruMemo
from .tracking.tracker_context import untracked

__all__ = [
    "SETTLED_SECONDS",
    "getsource",
    "getsourcelines",
    "own_source",
    "read_code_file",
    "read_code_text",
    "settled_source_version",
    "source_version_unchanged",
    "stat_has_settled",
]

#: How long a file must have been left alone before something read from it is
#: memoised on its ``(mtime, size)``. See `stat_has_settled`.
SETTLED_SECONDS = 2.0


def stat_has_settled(st: object) -> bool:
    """True when the file *st* describes may be memoised on its stat.

    A memo keyed on ``(mtime, size)`` cannot see an edit that keeps both, and
    a same-size edit moments after the last one can: the mtime moves in
    ticks -- ~15.6 ms on Windows, whole seconds on HFS+ and ext3, two on FAT
    -- so two saves inside one tick share it. The module digests served the
    first save's code for the second that way, and five tests that rewrite a
    file straight after reading it failed intermittently on Windows CI.

    Git's "racy git" rule, applied when the entry is made: a file whose mtime
    is within a tick of now may still be written again without the stat
    moving, so it is read every time; one untouched for longer than the
    coarsest tick cannot be, so what was read from it holds until the stat
    moves. Ask BEFORE reading the file, so an edit the read missed cannot be
    one that kept the stat. Costs a re-read for a couple of seconds after
    each save. (``file_dep_snapshot`` holds its input digests to the same
    rule, over a longer window.)
    """
    return time.time() - st.st_mtime > SETTLED_SECONDS


def read_code_file(path: str) -> bytes:
    """The bytes of the code file at *path*, read where no file tracker sees it.

    Cash reads a module to key or check the code that runs, not as data that
    code reads. Through ``open`` the read was recorded by whatever statement or
    cached call was running (the tracker patches ``open``), so the module
    became a raw-bytes input and a comment added to it re-ran the work. The
    memos in front of these reads hid it once a file had settled; for two
    seconds after a save (`stat_has_settled`) every key re-read the file.
    The tracker sees every Python-level open, ``io.FileIO`` included, so the
    read runs :class:`untracked`.
    """
    with untracked(), io.FileIO(path, "rb") as fh:
        return fh.readall()


def read_code_text(path: str) -> str:
    """:func:`read_code_file` as UTF-8 text, newlines translated as text-mode ``open`` does."""
    return read_code_file(path).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


#: `getsourcelines` answers: (path, mtime_ns, size, what) -> (lines, first line).
_SOURCE_LINES: LruMemo[tuple, tuple[tuple[str, ...], int]] = LruMemo(CODE_OBJECTS)
#: Where each class in a file starts, from one parse per file version:
#: (path, mtime_ns, size) -> {qualname: 0-based line}. Few entries: one is
#: needed while a file's classes are read, and each holds a whole file's map.
_CLASS_STARTS: LruMemo[tuple[str, int, int], dict[str, int]] = LruMemo(64)
#: ``inspect.findsource`` finds a class by parsing its whole file (before
#: 3.13, where it reads ``__firstlineno__``): a 3000-line module took ~50 ms
#: per call, and the analysis asks about a class several times.
_CLASS_SOURCE_PARSES_FILE = sys.version_info < (3, 13) and hasattr(inspect, "_ClassFinder")


def _source_target(obj: object) -> tuple | None:
    """What decides *obj*'s source within its file, or None to not memoise:
    the first line for code, the qualified name for a class."""
    if inspect.ismethod(obj):
        obj = obj.__func__
    if inspect.isfunction(obj):
        obj = obj.__code__
    if inspect.iscode(obj):
        return ("code", obj.co_firstlineno)
    if inspect.isclass(obj):
        return ("class", obj.__qualname__, vars(obj).get("__firstlineno__"))
    return None


def _disk_backed(path: str) -> bool:
    """Is what ``linecache`` gives for *path* the file on disk? Not for an
    entry put there by hand (a notebook cell, ``exec`` source: no mtime) or
    by a module loader: those can change while the file does not."""
    entry = linecache.cache.get(path)
    return entry is None or (len(entry) == 4 and entry[1] is not None)


def getsourcelines(obj: object) -> tuple[list[str], int]:
    """``inspect.getsourcelines``, read once per object and file version.

    The analysis and the key read the same functions' source many times in a
    cold process, and each read re-checked the file and re-tokenised the
    block. Memoised on the file's ``(mtime, size)`` once the file has settled
    (`stat_has_settled`), for files ``linecache`` reads from disk only; any
    other case asks ``inspect`` every time. Raises what it raises.
    """
    obj = inspect.unwrap(obj)
    target = _source_target(obj)
    version = settled_source_version(obj) if target is not None else None
    if version is None:
        return inspect.getsourcelines(obj)  # type: ignore[arg-type]
    key = (*version, target)
    hit = _SOURCE_LINES.get(key)
    if hit is not None:
        return list(hit[0]), hit[1]
    if target[0] == "class" and _CLASS_SOURCE_PARSES_FILE:
        lines, first = _class_source_lines(obj, version)
    else:
        lines, first = inspect.getsourcelines(obj)  # type: ignore[arg-type]
    if source_version_unchanged(version):
        _SOURCE_LINES[key] = (tuple(lines), first)
    return lines, first


def settled_source_version(obj: object) -> tuple[str, int, int] | None:
    """``(path, mtime_ns, size)`` of the file *obj*'s source is read from, when
    what is read from it may be memoised on that: the file is on disk, has
    settled (`stat_has_settled`), and ``linecache`` reads it from disk. None
    otherwise. Check `source_version_unchanged` after reading, before storing.
    """
    try:
        path = inspect.getsourcefile(obj)  # type: ignore[arg-type]
    except TypeError:
        return None
    if path is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat_has_settled(st) or not _disk_backed(path):
        return None
    return (path, st.st_mtime_ns, st.st_size)


def source_version_unchanged(version: tuple[str, int, int]) -> bool:
    """Is the file still at *version*? A memo stores what it read only then."""
    path = version[0]
    try:
        st = os.stat(path)
    except OSError:
        return False
    return (path, st.st_mtime_ns, st.st_size) == version and _disk_backed(path)


def getsource(obj: object) -> str:
    """``inspect.getsource`` through `getsourcelines`'s memo."""
    return "".join(getsourcelines(obj)[0])


def _class_source_lines(cls: object, version: tuple[str, int, int]) -> tuple[list[str], int]:
    """``inspect.getsourcelines`` for a class, with the file parsed once per
    version for all its classes rather than once per question."""
    path = version[0]
    linecache.checkcache(path)
    module = inspect.getmodule(cls, path)
    lines = linecache.getlines(path, module.__dict__) if module else linecache.getlines(path)
    if not lines:
        raise OSError("could not get source code")
    starts = _CLASS_STARTS.get(version)
    if starts is None:
        starts = _class_starts(ast.parse("".join(lines)))
        if source_version_unchanged(version):
            _CLASS_STARTS[version] = starts
    first = starts.get(cls.__qualname__)  # type: ignore[attr-defined]
    if first is None:
        raise OSError("could not find class definition")
    return inspect.getblock(lines[first:]), first + 1


def _class_starts(tree: ast.AST) -> dict[str, int]:
    """Each class's qualified name -> the 0-based line its source starts on
    (its first decorator), the first definition in walk order winning: what
    ``inspect``'s ``_ClassFinder`` answers, for every class at once."""
    starts: dict[str, int] = {}
    stack: list[str] = []

    def visit_children(node: ast.AST) -> None:
        # A class or function is a statement: no expression holds one.
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.expr):
                visit(child)

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend((node.name, "<locals>"))
            visit_children(node)
            del stack[-2:]
        elif isinstance(node, ast.ClassDef):
            stack.append(node.name)
            line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            starts.setdefault(".".join(stack), line - 1)
            visit_children(node)
            stack.pop()
        else:
            visit_children(node)

    visit(tree)
    return starts


def own_source(fn: object) -> str:
    """``inspect.getsource``, without following ``__wrapped__`` for a function.

    ``getsource`` unwraps, so for a ``functools.wraps`` wrapper it returned
    the WRAPPED function's text. Keyed by that, every function one decorator
    wraps shares the wrapper's code object and was keyed by whichever wrapped
    body was read first; analysed by it, the wrapper's own body was never
    read. Reading the code object gives each half its own
    text; whoever needs the wrapped half reaches it through ``__wrapped__``.
    Raises what ``inspect.getsource`` raises (`SOURCE_RETRIEVAL_ERRORS`).
    """
    if isinstance(fn, types.FunctionType) and hasattr(fn, "__wrapped__"):
        return getsource(fn.__code__)
    return getsource(fn)
