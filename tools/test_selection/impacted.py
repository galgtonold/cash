"""Pick the tests a diff can affect, from a per-test line-coverage map.

Usage (from the repo root)::

    python tools/test_selection/impacted.py                  # print node ids, one per line
    python tools/test_selection/impacted.py --run            # ... and run them with pytest
    python tools/test_selection/impacted.py --run -n 4 -- -x # workers, then extra pytest args
    python tools/test_selection/impacted.py -o ids.txt       # write them to a file
    python tools/test_selection/impacted.py main..HEAD       # another range
    python tools/test_selection/impacted.py --file-level     # no map: tests that import what changed
    python tools/test_selection/impacted.py build            # build the map (the whole suite, slow)

The range is ``A...B`` (from the merge base of A and B), ``A..B`` or a single
ref ``A`` (to HEAD). The default is ``origin/main...HEAD``; when the range ends
at HEAD, uncommitted and untracked files count as changed too
(``--no-worktree`` leaves them out). Node ids go to stdout, one per line, and
nothing else does, so ``mapfile -t ids < <(... impacted.py)`` is safe; what was
picked and why goes to stderr.

How a changed file is mapped to tests:

* ``src/**.py``: the tests whose recorded coverage runs a changed line. Lines
  that run at import (``def`` lines, module constants) run once per process,
  for whichever test came first, so a change there counts as a change to the
  function it heads, or, at module level, to the whole file. Changed lines no
  test runs (new code, an untested branch) widen to the tests of the enclosing
  function or class, then of the file. A file the map has no coverage for falls
  back to the tests that import it, then to tests named after it.
* ``src/cash/_c/*`` and ``hatch_build.py``: the C extension. Its code is not
  traced, so this picks the tests that run the Python lines calling into it
  (names bound from ``cash._native``), the tests that import ``cash._native``
  or ``cash._cwatch``, and ``tests/test_core/watchers/``.
* Test files that changed run whole; a changed ``conftest.py`` runs its
  directory; a changed test helper runs the tests that import it.
* ``docs/`` and top-level ``*.md`` select ``tests/docs``; ``tools/`` selects
  ``tests/test_tooling``. Anything else selects the tests that mention its name.

The map is a SQLite file outside the repo: ``--map``, else ``$CASH_TEST_MAP``,
else ``~/.cache/cash-test-map/map.sqlite``. ``build`` makes it by running the
suite once under ``run_baseline.py`` with per-test coverage contexts; see
``docs/how-it-works/testing.md`` for the command and how long it takes. When the
map was built at another commit, changed lines are carried to the map's line
numbers through ``git diff``; lines that moved too much to place widen to their
file, and the report says how many.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

#: Bumped whenever the map's tables change meaning; an older map is rejected.
FORMAT_VERSION = 1
DEFAULT_RANGE = "origin/main...HEAD"
MAP_ENV = "CASH_TEST_MAP"

#: What ``build`` runs: every test CI runs, but not the wheel gate or the
#: wall-clock ``perf`` tests, which coverage would only make fail.
BUILD_PATHS = ["tests"]
BUILD_PYTEST_ARGS = ["--ignore=tests/test_wheel_gate", "-m", "not perf"]

C_SOURCES = ("src/cash/_c/", "hatch_build.py")
C_MODULE_RE = re.compile(r"\b_native\b|\b_cwatch\b")
C_TEST_DIR = "tests/test_core/watchers"

#: Basenames too common for a name search to mean anything.
GENERIC_NAMES = {"__init__.py", "__main__.py", "README.md", "conftest.py", "setup.py", "index.md"}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE tests (id INTEGER PRIMARY KEY, nodeid TEXT NOT NULL UNIQUE, outcome TEXT, duration REAL);
CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE);
-- lines: coverage.py's numbits encoding (bit n of the little-endian bytes = line n)
CREATE TABLE cover (file_id INTEGER NOT NULL, test_id INTEGER NOT NULL, lines BLOB NOT NULL,
                    PRIMARY KEY (file_id, test_id)) WITHOUT ROWID;
"""


class UsageError(Exception):
    """A problem the user fixes (no map, a bad range); printed without a traceback."""


def default_map_path() -> Path:
    env = os.environ.get(MAP_ENV)
    return Path(env).expanduser() if env else Path.home() / ".cache" / "cash-test-map" / "map.sqlite"


# ---------------------------------------------------------------------------
# git


def git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise UsageError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def resolve(root: Path, rev: str) -> str:
    return git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").strip()


@dataclass(frozen=True)
class DiffRange:
    base: str  # a commit
    head: str | None  # a commit, or None for the working tree
    label: str


def parse_range(root: Path, spec: str, worktree: bool = True) -> DiffRange:
    if "..." in spec:
        left, right = spec.split("...", 1)
        right = right or "HEAD"
        base = git(root, "merge-base", left or "HEAD", right, check=False).strip()
        if not base:
            raise UsageError(f"no merge base for {spec!r}: does {left!r} exist? (git fetch origin)")
    elif ".." in spec:
        left, right = spec.split("..", 1)
        right = right or "HEAD"
        base = resolve(root, left or "HEAD")
    else:
        base, right = resolve(root, spec), "HEAD"
    if not base:
        raise UsageError(f"unknown revision in {spec!r}")
    head = resolve(root, right)
    at_head = head == resolve(root, "HEAD")
    if worktree and at_head:
        return DiffRange(base, None, f"{spec} plus uncommitted changes")
    return DiffRange(base, head, spec)


# ---------------------------------------------------------------------------
# diffs


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int

    def old_lines(self) -> range:
        return range(self.old_start, self.old_start + self.old_count)

    def new_lines(self) -> range:
        return range(self.new_start, self.new_start + self.new_count)


@dataclass
class FileDiff:
    old_path: str | None  # None: added
    new_path: str | None  # None: deleted
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def unquote_path(raw: str) -> str:
    """A path as ``git diff`` prints it: maybe C-quoted, maybe with a trailing tab."""
    raw = raw.rstrip("\t\n")
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
        raw = raw.encode("latin-1").decode("utf-8", "replace")
    return raw


def parse_diff(text: str) -> list[FileDiff]:
    """Parse ``git diff -U0`` output into files and hunks."""
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            cur = FileDiff(None, None)
            files.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("rename from "):
            cur.old_path = unquote_path(line[len("rename from ") :])
        elif line.startswith("rename to "):
            cur.new_path = unquote_path(line[len("rename to ") :])
        elif line.startswith("--- "):
            name = unquote_path(line[4:])
            cur.old_path = None if name == "/dev/null" else name[2:]
        elif line.startswith("+++ "):
            name = unquote_path(line[4:])
            cur.new_path = None if name == "/dev/null" else name[2:]
        elif line.startswith("@@"):
            m = HUNK_RE.match(line)
            if m:
                os_, oc, ns, nc = m.groups()
                cur.hunks.append(Hunk(int(os_), int(oc or 1), int(ns), int(nc or 1)))
    # a pure rename or mode change has no ---/+++ lines; one "diff --git" header gives both paths
    for fd in files:
        if fd.old_path is None and fd.new_path is None:
            fd.old_path = fd.new_path = ""
    return [fd for fd in files if fd.path]


def diff_files(root: Path, old: str, new: str | None) -> list[FileDiff]:
    """Files changed from commit ``old`` to commit ``new`` (None: the working tree, untracked included)."""
    args = ["diff", "-U0", "-M", "--no-color", "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/"]
    files = parse_diff(git(root, *args, old, *([new] if new else [])))
    if new is None:
        for name in git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0"):
            if not name:
                continue
            try:
                count = len((root / name).read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                count = 0
            files.append(FileDiff(None, name, [Hunk(0, 0, 1, count)] if count else []))
    return files


def map_back(hunks: list[Hunk], line: int) -> int | None:
    """The old-side line of new-side ``line``, or None if the diff added it."""
    offset = 0
    for h in sorted(hunks, key=lambda h: h.new_start):
        start = h.new_start if h.new_count else h.new_start + 1
        if line < start:
            break
        if line < start + h.new_count:
            return None
        offset += h.new_count - h.old_count
    return line - offset


class LineMapper:
    """Carries line numbers of one file at some revision back to the map's commit."""

    def __init__(self, hunks: list[Hunk] | None, present: bool = True):
        self.hunks = hunks or []
        self.present = present

    def __call__(self, line: int) -> int | None:
        if not self.present:
            return None
        return map_back(self.hunks, line)


# ---------------------------------------------------------------------------
# sources and scopes


class Sources:
    """File contents at a commit (or, for None, in the working tree), memoised."""

    def __init__(self, root: Path):
        self.root = root
        self._memo: dict[tuple[str | None, str], str | None] = {}

    def get(self, rev: str | None, path: str) -> str | None:
        key = (rev, path)
        if key not in self._memo:
            if rev is None:
                p = self.root / path
                self._memo[key] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
            else:
                proc = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=self.root, capture_output=True, check=False)
                self._memo[key] = proc.stdout.decode("utf-8", "replace") if proc.returncode == 0 else None
        return self._memo[key]


@dataclass(frozen=True)
class Scope:
    qualname: str
    kind: str  # "def" or "class"
    start: int  # first decorator or the def/class line
    body_start: int
    end: int

    def contains(self, line: int) -> bool:
        return self.start <= line <= self.end


@dataclass
class FileScopes:
    scopes: list[Scope]
    body: set[int]  # lines inside function bodies: the ones a test (not an import) runs

    @classmethod
    def parse(cls, source: str | None) -> FileScopes:
        if source is None:
            return cls([], set())
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return cls([], set())
        out: list[Scope] = []

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = f"{prefix}{child.name}"
                    start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                    kind = "class" if isinstance(child, ast.ClassDef) else "def"
                    out.append(Scope(name, kind, start, child.body[0].lineno, child.end_lineno or child.lineno))
                    visit(child, name + ".")
                else:
                    visit(child, prefix)

        visit(tree, "")
        body: set[int] = set()
        for s in out:
            if s.kind == "def":
                body.update(range(s.body_start, s.end + 1))
        return cls(out, body)

    def chain(self, line: int) -> list[Scope]:
        """The scopes holding ``line``, innermost first."""
        return sorted((s for s in self.scopes if s.contains(line)), key=lambda s: s.end - s.start)

    def lines_of(self, scope: Scope) -> set[int]:
        lo = scope.body_start if scope.kind == "def" else scope.start
        return {n for n in self.body if lo <= n <= scope.end}

    def by_name(self, qualname: str) -> list[Scope]:
        return [s for s in self.scopes if s.qualname == qualname]


# ---------------------------------------------------------------------------
# the map


def lines_to_numbits(lines: Iterable[int]) -> bytes:
    lines = sorted(set(lines))
    if not lines:
        return b""
    buf = bytearray(lines[-1] // 8 + 1)
    for n in lines:
        buf[n // 8] |= 1 << (n % 8)
    return bytes(buf)


def numbits_to_lines(blob: bytes) -> set[int]:
    mask = int.from_bytes(blob, "little")
    return {i for i in range(mask.bit_length()) if mask >> i & 1}


def mask_of(lines: Iterable[int]) -> int:
    m = 0
    for n in lines:
        m |= 1 << n
    return m


class CoverageMap:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            self.meta = dict(self.db.execute("SELECT key, value FROM meta"))
        except sqlite3.Error as exc:
            raise UsageError(f"{path} is not a test map ({exc}); rebuild it with `impacted.py build`") from exc
        version = int(self.meta.get("format_version", "0"))
        if version != FORMAT_VERSION:
            raise UsageError(
                f"{path} has map format {version}, this tool reads {FORMAT_VERSION}; rebuild it with `impacted.py build`"
            )
        self.commit = self.meta.get("commit", "")
        self._files = dict(self.db.execute("SELECT path, id FROM files"))
        self._memo: dict[str, list[tuple[str, int]]] = {}
        self.durations = {n: d or 0.0 for n, d in self.db.execute("SELECT nodeid, duration FROM tests")}

    def has_file(self, path: str) -> bool:
        return path in self._files

    def files(self) -> list[str]:
        return sorted(self._files)

    def all_tests(self, path: str) -> set[str]:
        return {nid for nid, bits in self._rows(path) if bits}

    def _rows(self, path: str) -> list[tuple[str, int]]:
        if path not in self._memo:
            fid = self._files.get(path)
            rows = (
                self.db.execute(
                    "SELECT t.nodeid, c.lines FROM cover c JOIN tests t ON t.id = c.test_id WHERE c.file_id = ?",
                    (fid,),
                )
                if fid is not None
                else []
            )
            self._memo[path] = [(nid, int.from_bytes(blob, "little")) for nid, blob in rows]
        return self._memo[path]

    def tests_for(self, path: str, lines: Iterable[int]) -> set[str]:
        m = mask_of(lines)
        if not m:
            return set()
        return {nid for nid, bits in self._rows(path) if bits & m}


def write_map(
    path: Path,
    meta: dict[str, str],
    tests: dict[str, tuple[str | None, float | None]],
    cover: Iterable[tuple[str, str, bytes]],
) -> None:
    """Write a map atomically. ``cover`` yields (node id, repo-relative file, numbits)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO meta VALUES (?, ?)", sorted({**meta, "format_version": str(FORMAT_VERSION)}.items()))
    test_ids: dict[str, int] = {}
    file_ids: dict[str, int] = {}

    def tid(nodeid: str) -> int:
        if nodeid not in test_ids:
            outcome, duration = tests.get(nodeid, (None, None))
            test_ids[nodeid] = db.execute(
                "INSERT INTO tests (nodeid, outcome, duration) VALUES (?, ?, ?)", (nodeid, outcome, duration)
            ).lastrowid
        return test_ids[nodeid]

    def fid(rel: str) -> int:
        if rel not in file_ids:
            file_ids[rel] = db.execute("INSERT INTO files (path) VALUES (?)", (rel,)).lastrowid
        return file_ids[rel]

    merged: dict[tuple[int, int], int] = {}
    for nodeid, rel, blob in cover:
        key = (fid(rel), tid(nodeid))
        merged[key] = merged.get(key, 0) | int.from_bytes(blob, "little")
    for nodeid in tests:
        tid(nodeid)
    db.executemany(
        "INSERT INTO cover VALUES (?, ?, ?)",
        ((f, t, m.to_bytes((m.bit_length() + 7) // 8, "little")) for (f, t), m in merged.items() if m),
    )
    db.commit()
    db.close()
    os.replace(tmp, path)


def repo_relative(raw: str, root: Path) -> str | None:
    """A coverage.py file path as a repo-relative posix path (None: not ours)."""
    norm = raw.replace("\\", "/")
    root_s = root.resolve().as_posix().rstrip("/") + "/"
    if norm.lower().startswith(root_s.lower()):
        return norm[len(root_s) :]
    i = norm.rfind("/src/cash/")
    return norm[i + 1 :] if i >= 0 else None


def convert_baseline(out_dir: Path, map_path: Path, root: Path) -> dict[str, str]:
    """Turn a ``run_baseline.py`` output directory into a map."""
    data = out_dir / "cov" / ".coverage"
    if not data.is_file():
        raise UsageError(f"no combined coverage data at {data}")
    info = json.loads((out_dir / "run_info.json").read_text(encoding="utf-8"))
    results_file = out_dir / "results.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.is_file() else {}
    tests = {nid: (r.get("outcome"), r.get("duration")) for nid, r in results.items()}

    src = sqlite3.connect(f"file:{data.as_posix()}?mode=ro", uri=True)
    tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not {"file", "context", "line_bits"} <= tables:
        raise UsageError(f"{data} is not coverage.py line data with contexts (tables: {sorted(tables)})")
    files = {fid: repo_relative(p, root) for fid, p in src.execute("SELECT id, path FROM file")}
    contexts = dict(src.execute("SELECT id, context FROM context"))

    def rows() -> Iterable[tuple[str, str, bytes]]:
        for file_id, ctx_id, blob in src.execute("SELECT file_id, context_id, numbits FROM line_bits"):
            nodeid, rel = contexts.get(ctx_id, ""), files.get(file_id)
            if nodeid and "::" in nodeid and rel:
                yield nodeid, rel, blob

    dirty = git(root, "status", "--porcelain", "--untracked-files=no", "--", "src", check=False).strip()
    meta = {
        "commit": info.get("commit", ""),
        "built_at": info.get("started", ""),
        "elapsed_seconds": str(info.get("elapsed_seconds", "")),
        "pytest_exit_code": str(info.get("pytest_exit_code", "")),
        "paths": " ".join(info.get("paths", [])),
        "python": info.get("python", ""),
        "platform": info.get("platform", ""),
        "src_dirty_when_converted": "yes" if dirty else "no",
    }
    write_map(map_path, meta, tests, rows())
    src.close()
    return meta


# ---------------------------------------------------------------------------
# a static index of tests/


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith("tests/") and path.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))
    )


def module_name(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    parts = path[:-3].split("/")
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or None


class TestIndex:
    """Every .py file under tests/: what it imports and what it says."""

    def __init__(self, root: Path):
        self.root = root
        self.text: dict[str, str] = {}
        self.imports: dict[str, set[str]] = {}
        tests_dir = root / "tests"
        for p in sorted(tests_dir.rglob("*.py")) if tests_dir.is_dir() else []:
            rel = p.relative_to(root).as_posix()
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            self.text[rel] = text
            self.imports[rel] = self._imports(rel, text)

    @staticmethod
    def _imports(rel: str, text: str) -> set[str]:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return set()
        package = (module_name(rel) or "").split(".")
        if not rel.endswith("__init__.py"):
            package = package[:-1]
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - node.level + 1] if node.level > 1 else package
                    mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                found.add(mod)
                found.update(f"{mod}.{a.name}" for a in node.names)
        return found

    def mentioning(self, pattern: re.Pattern[str]) -> set[str]:
        return {rel for rel, text in self.text.items() if pattern.search(text)}

    def importers(self, module: str) -> set[str]:
        pat = re.compile(rf"\b{re.escape(module)}(?![\w])")
        return {rel for rel, mods in self.imports.items() if module in mods} | self.mentioning(pat)

    def selectable(self, hits: set[str]) -> set[str]:
        """Test files and conftest directories reached from ``hits``, through helper modules."""
        out: set[str] = set()
        seen: set[str] = set()
        queue = list(hits)
        while queue:
            rel = queue.pop()
            if rel in seen:
                continue
            seen.add(rel)
            if is_test_file(rel):
                out.add(rel)
            elif rel.endswith("/conftest.py"):
                out.add(rel.rsplit("/", 1)[0])
            else:
                mod = module_name(rel)
                if mod:
                    queue.extend(self.importers(mod) - seen)
        return out


# ---------------------------------------------------------------------------
# selection


@dataclass
class Selection:
    reasons: dict[str, set[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, ids: Iterable[str], reason: str) -> int:
        n = 0
        for i in ids:
            self.reasons.setdefault(i, set()).add(reason)
            n += 1
        return n

    def ids(self) -> list[str]:
        """Sorted, with ids inside a selected file or directory dropped."""
        paths = {i for i in self.reasons if "::" not in i}
        out = []
        for i in sorted(self.reasons):
            file_part = i.split("::", 1)[0]
            if "::" in i and file_part in paths:
                continue
            if any(file_part.startswith(p + "/") for p in paths if p != file_part):
                continue
            out.append(i)
        return out


class Selector:
    def __init__(self, root: Path, rng: DiffRange, cmap: CoverageMap | None):
        self.root = root
        self.rng = rng
        self.cmap = cmap
        self.sources = Sources(root)
        self.index = TestIndex(root)
        self.sel = Selection()
        self.unmapped_lines = 0
        self.changes = diff_files(root, rng.base, rng.head)
        self.map_commit: str | None = None
        self._to_base: dict[str, FileDiff] = {}
        self._to_new: dict[str, FileDiff] = {}
        self._map_paths_base: dict[str, str | None] = {}
        self._map_paths_new: dict[str, str | None] = {}
        self.changed_since_map: set[str] = set()
        if cmap is not None:
            self._prepare_map()

    # -- the map's commit vs. this diff --------------------------------------

    def _prepare_map(self) -> None:
        assert self.cmap is not None
        commit = self.cmap.commit
        if not commit or not resolve_quiet(self.root, commit):
            self.sel.notes.append(
                f"the map was built at {commit[:10] or 'an unknown commit'}, which is not in this clone "
                "(git fetch, or rebuild the map): every changed file widens to file level"
            )
            return
        self.map_commit = commit
        for target, into, paths in (
            (self.rng.base, self._to_base, self._map_paths_base),
            (self.rng.head, self._to_new, self._map_paths_new),
        ):
            if target == commit:
                continue
            for fd in diff_files(self.root, commit, target):
                if fd.new_path:
                    into[fd.new_path] = fd
                    paths[fd.new_path] = fd.old_path
                if fd.old_path and fd.old_path != fd.new_path:
                    self.changed_since_map.add(fd.old_path)
                if fd.new_path:
                    self.changed_since_map.add(fd.new_path)
        if self.map_commit != self.rng.base:
            ahead = git(self.root, "rev-list", "--count", f"{self.map_commit}..{self.rng.base}", check=False).strip()
            behind = git(self.root, "rev-list", "--count", f"{self.rng.base}..{self.map_commit}", check=False).strip()
            self.sel.notes.append(
                f"the map is at {self.map_commit[:10]}, the diff base at {self.rng.base[:10]} "
                f"({ahead or '?'} commits after the map, map {behind or '?'} commits after the base): "
                "line numbers are carried over through git diff"
            )

    def _mapper(self, side: str, path: str | None) -> tuple[LineMapper, str | None]:
        """A line mapper from ``side`` ("base"/"new") of ``path`` to the map commit, and the map-side path."""
        if path is None:
            return LineMapper(None, present=False), None
        diffs, paths = (self._to_base, self._map_paths_base) if side == "base" else (self._to_new, self._map_paths_new)
        if path in diffs:
            fd = diffs[path]
            return LineMapper(fd.hunks, present=fd.old_path is not None), paths.get(path)
        return LineMapper(None), path

    # -- rules ----------------------------------------------------------------

    def run(self) -> Selection:
        for fd in self.changes:
            self._one(fd)
        self._validate()
        if self.unmapped_lines:
            self.sel.notes.append(
                f"{self.unmapped_lines} changed line(s) could not be placed in the map's version of their file "
                "(they moved since the map was built); those hunks widened to file level"
            )
        return self.sel

    def _one(self, fd: FileDiff) -> None:
        path = fd.path
        if path.startswith("tests/"):
            self._test_side(fd)
        elif path.startswith(C_SOURCES):
            self._c_extension(path)
        elif path.startswith("src/") and path.endswith(".py"):
            if self.cmap is None:
                self._static_module(path, "file-level")
            else:
                self._covered_module(fd)
        elif path.startswith("docs/") or ("/" not in path and path.endswith(".md")):
            self._note(path, self.sel.add(["tests/docs"], f"{path}: docs change -> tests/docs"), "tests/docs (rule)")
        elif path.startswith("tools/"):
            n = self.sel.add(["tests/test_tooling"], f"{path}: tooling change -> tests/test_tooling")
            n += self._mentions(path)
            self._note(path, n, "tests/test_tooling (rule) and tests naming it")
        else:
            n = self._mentions(path)
            self._note(path, n, "tests naming it" if n else "no rule maps it to tests; nothing selected")

    def _note(self, path: str, n: int, how: str) -> None:
        self.sel.notes.append(f"{path}: {how}" + (f" ({n} selected)" if n else ""))

    def _mentions(self, path: str) -> int:
        name = path.rsplit("/", 1)[-1]
        if name in GENERIC_NAMES or len(name) < 6:
            return 0
        stem = name.rsplit(".", 1)[0] if name.endswith(".py") else name
        hits = self.index.mentioning(re.compile(rf"(?<![\w]){re.escape(stem)}(?![\w])"))
        hits = {h for h in hits if h != path}
        return self.sel.add(self.index.selectable(hits), f"{path}: name search")

    def _test_side(self, fd: FileDiff) -> None:
        path = fd.path
        if fd.new_path is None:
            self.sel.notes.append(f"{path}: deleted")
            return
        if is_test_file(path):
            self.sel.add([path], f"{path}: the test file changed")
            self.sel.notes.append(f"{path}: test file changed, runs whole")
        elif path.endswith("/conftest.py") or path == "tests/conftest.py":
            d = path.rsplit("/", 1)[0]
            self.sel.add([d], f"{path}: conftest changed")
            self.sel.notes.append(f"{path}: conftest changed, runs {d}/")
        elif path.endswith(".py"):
            mod = module_name(path) or ""
            hits = self.index.selectable(self.index.importers(mod) - {path})
            self._note(path, self.sel.add(hits, f"{path}: test helper imported"), "tests importing this helper")
        else:
            self._note(path, self._mentions(path), "tests naming this file")

    def _c_extension(self, path: str) -> None:
        n_static = self.sel.add(
            self.index.selectable(self.index.mentioning(C_MODULE_RE)),
            f"{path}: C extension -> tests importing cash._native / cash._cwatch",
        )
        n_dir = self.sel.add([C_TEST_DIR] if (self.root / C_TEST_DIR).is_dir() else [], f"{path}: C extension")
        n_cov = 0
        if self.cmap is not None and self.map_commit:
            sites = c_call_sites(self.cmap, self.sources, self.map_commit)
            tests: set[str] = set()
            for rel, lines in sites.items():
                tests |= self.cmap.tests_for(rel, lines)
            n_cov = self.sel.add(tests, f"{path}: C extension -> tests running the Python lines that call into it")
            where = f"{sum(len(v) for v in sites.values())} call-site lines in {len(sites)} modules"
        else:
            where = "no map, so no call-site coverage"
        self.sel.notes.append(
            f"{path}: C source, not traced: {n_cov} tests run its Python call sites ({where}); "
            f"{n_static} test files import cash._native/_cwatch" + (f"; plus {C_TEST_DIR}/" if n_dir else "")
        )

    def _static_module(self, path: str, why: str) -> int:
        mod = module_name(path) or ""
        hits = self.index.selectable(self.index.importers(mod))
        n = self.sel.add(hits, f"{path}: {why}: tests importing {mod}")
        if n:
            self.sel.notes.append(f"{path}: {why}: {n} test files/dirs import {mod}")
            return n
        n = self._name_heuristic(path)
        return n

    def _name_heuristic(self, path: str) -> int:
        parts = path[:-3].split("/")[2:] if path.startswith("src/cash/") else path[:-3].split("/")
        stem = parts[-1] if parts and parts[-1] != "__init__" else (parts[-2] if len(parts) > 1 else "")
        files = {
            rel for rel in self.index.text if is_test_file(rel) and len(stem) >= 4 and stem in rel.rsplit("/", 1)[-1]
        }
        hits: set[str] = set(files)
        if not hits:
            dirs = {p.relative_to(self.root).as_posix() for p in (self.root / "tests").rglob("*") if p.is_dir()}
            for comp in reversed(parts[:-1] or parts):
                hits = {d for d in dirs if d.rsplit("/", 1)[-1].removeprefix("test_") == comp}
                if hits:
                    break
        n = self.sel.add(hits, f"{path}: name heuristic")
        self.sel.notes.append(
            f"{path}: nothing covers or imports it; name heuristic picked {n}: {', '.join(sorted(hits)) or 'nothing'}"
        )
        return n

    def _covered_module(self, fd: FileDiff) -> None:
        assert self.cmap is not None
        base_map, mpath_b = self._mapper("base", fd.old_path)
        new_map, mpath_n = self._mapper("new", fd.new_path)
        mpath = mpath_b or mpath_n
        path = fd.path
        if self.map_commit is None or mpath is None or not self.cmap.has_file(mpath):
            why = "not in the map" if self.map_commit else "map commit unavailable"
            if self.map_commit is None and mpath and self.cmap.has_file(mpath):
                n = self.sel.add(self.cmap.all_tests(mpath), f"{path}: file level")
                self.sel.notes.append(f"{path}: {why}: file level, {n} tests")
                return
            self._static_module(path, why)
            return
        scopes_m = FileScopes.parse(self.sources.get(self.map_commit, mpath))
        if fd.new_path is None:
            n = self.sel.add(self._file_level(mpath, scopes_m), f"{path}: deleted -> its tests")
            self.sel.notes.append(f"{path}: deleted; {n} tests ran it")
            return
        scopes_base = FileScopes.parse(self.sources.get(self.rng.base, fd.old_path)) if fd.old_path else None
        scopes_new = FileScopes.parse(self.sources.get(self.rng.head, fd.new_path))
        counts = {"line": 0, "function": 0, "file": 0}
        picked: set[str] = set()
        names: set[str] = set()
        for h in fd.hunks:
            old_mapped = [base_map(n) for n in h.old_lines()]
            new_mapped = [new_map(n) for n in h.new_lines()]
            precise = {n for n in old_mapped + new_mapped if n is not None}
            if precise:
                tests, how = self._precise(mpath, scopes_m, precise)
                if tests:
                    picked |= tests
                    self.sel.add(tests, f"{path}: {how}")
                    if how == "changed lines":
                        counts["line"] += 1
                    elif how.startswith("module-level"):
                        counts["file"] += 1
                    else:
                        counts["function"] += 1
                        names.add(how.removeprefix("signature of "))
                    continue
            if h.old_count and all(n is None for n in old_mapped) and not precise and self.map_commit != self.rng.base:
                self.unmapped_lines += h.old_count
                tests = self._file_level(mpath, scopes_m)
                picked |= tests
                self.sel.add(tests, f"{path}: lines moved since the map, file level")
                counts["file"] += 1
                continue
            # no test runs these lines (new code, or a branch nothing reaches): widen
            qualnames: list[str] = []
            if precise:
                qualnames = [s.qualname for s in scopes_m.chain(min(precise))]
            elif h.new_count:
                qualnames = [s.qualname for s in scopes_new.chain(h.new_start)]
            elif h.old_count and scopes_base is not None:
                qualnames = [s.qualname for s in scopes_base.chain(h.old_start)]
            tests, name = self._by_scope(mpath, scopes_m, qualnames)
            if not tests:
                tests = self._file_level(mpath, scopes_m)
                picked |= tests
                self.sel.add(tests, f"{path}: no test runs the changed lines; file level")
                counts["file"] += 1
            else:
                picked |= tests
                self.sel.add(tests, f"{path}: no test runs the changed lines; tests of {name}")
                names.add(name)
                counts["function"] += 1
        if fd.hunks and not picked:
            self._static_module(path, "no recorded test runs it")
            return
        parts = [f"{counts['line']} hunk(s) by changed line ({len(picked)} tests)"] if counts["line"] else []
        if counts["function"]:
            parts.append(f"{counts['function']} widened to the enclosing function/class ({', '.join(sorted(names))})")
        if counts["file"]:
            parts.append(f"{counts['file']} widened to the whole file")
        self.sel.notes.append(f"{path}: " + ("; ".join(parts) or "no hunks"))

    def _precise(self, mpath: str, scopes: FileScopes, lines: set[int]) -> tuple[set[str], str]:
        """Tests running ``lines`` (map numbering), import-time lines read as the code they define."""
        query = {n for n in lines if n in scopes.body}
        how = "changed lines"
        for n in lines - query:
            chain = scopes.chain(n)
            if chain and chain[0].kind == "def":
                query |= scopes.lines_of(chain[0])
                how = f"signature of {chain[0].qualname}"
            else:
                return self._file_level(mpath, scopes), "module-level code, file level"
        return self.cmap.tests_for(mpath, query), how

    def _by_scope(self, mpath: str, scopes: FileScopes, qualnames: list[str]) -> tuple[set[str], str]:
        for q in qualnames:
            # an enclosing def may be new too; its own parents still exist
            for s in scopes.by_name(q):
                tests = self.cmap.tests_for(mpath, scopes.lines_of(s))
                if tests:
                    return tests, q
        return set(), ""

    def _file_level(self, mpath: str, scopes: FileScopes) -> set[str]:
        # function bodies only: import-time lines run for whichever test came first
        return self.cmap.tests_for(mpath, scopes.body) if scopes.body else self.cmap.all_tests(mpath)

    # -- check the ids still exist -------------------------------------------

    def _validate(self) -> None:
        dropped = 0
        for nid in list(self.sel.reasons):
            file_part = nid.split("::", 1)[0]
            if not (self.root / file_part).exists():
                del self.sel.reasons[nid]
                dropped += 1
                continue
            if "::" in nid and file_part in self.changed_since_map:
                fixed = existing_id(self.root, nid)
                reasons = self.sel.reasons.pop(nid)
                if fixed is None:
                    dropped += 1
                else:
                    self.sel.reasons.setdefault(fixed, set()).update(reasons)
        if dropped:
            self.sel.notes.append(f"{dropped} recorded test(s) no longer exist and were left out")


def resolve_quiet(root: Path, rev: str) -> str:
    return git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False).strip()


def existing_id(root: Path, nodeid: str) -> str | None:
    """``nodeid`` without its parameters if its test still exists in the file, else None.

    The file changed since the map was built, so a parameter id may be stale;
    the bare function id runs every parameter there is now.
    """
    file_part, _, rest = nodeid.partition("::")
    names = rest.split("[", 1)[0].split("::")
    try:
        tree = ast.parse((root / file_part).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None
    scope: list[ast.stmt] = tree.body
    for name in names:
        found = next(
            (
                n
                for n in scope
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
            ),
            None,
        )
        if found is None:
            return None
        scope = found.body if isinstance(found, ast.ClassDef) else []
    return f"{file_part}::{'::'.join(names)}"


def c_call_sites(cmap: CoverageMap, sources: Sources, commit: str) -> dict[str, set[int]]:
    """Function-body lines, per mapped module, that use a name bound from ``cash._native``."""
    out: dict[str, set[int]] = {}
    for rel in cmap.files():
        if not rel.endswith(".py") or rel.endswith("_native.py"):
            continue
        text = sources.get(commit, rel)
        if not text or "_native" not in text:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        bound = {"_native"}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("_native"):
                bound.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
                v = node.value
                if isinstance(v.value, ast.Name) and v.value.id == "_native":
                    bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        body = FileScopes.parse(text).body
        lines = {n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in bound and n.lineno in body}
        # a function-local `from .._native import cwatch` binds it in that body only
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("_native") and node.lineno in body:
                lines.add(node.lineno)
        if lines:
            out[rel] = lines
    return out


# ---------------------------------------------------------------------------
# CLI


def open_map(path: Path) -> CoverageMap:
    if not path.is_file():
        raise UsageError(
            f"no test map at {path}.\n"
            "Build one (the whole suite once with per-test coverage; about half an hour on 16+ cores):\n"
            "    python tools/test_selection/impacted.py build\n"
            f"or point --map / ${MAP_ENV} at one, or use --file-level (tests that import what changed; no map)."
        )
    return CoverageMap(path)


def pytest_reads_argfiles() -> bool:
    try:
        from importlib.metadata import version

        major, minor = (int(x) for x in version("pytest").split(".")[:2])
    except (ImportError, ValueError):
        return False
    return (major, minor) >= (8, 2)


def run_pytest(root: Path, ids: list[str], ids_file: Path | None, workers: str | None, extra: list[str]) -> int:
    own_file = None
    try:
        if pytest_reads_argfiles():
            if ids_file is None:
                fd, name = tempfile.mkstemp(prefix="cash-impacted-", suffix=".txt")
                os.close(fd)
                own_file = Path(name)
                own_file.write_text("".join(i + "\n" for i in ids), encoding="utf-8")
                ids_file = own_file
            targets = [f"@{ids_file}"]
        else:
            targets = ids  # no shell is involved, so spaces and brackets pass as they are
        cmd = [sys.executable, "-m", "pytest", *targets, *(["-n", workers] if workers else []), *extra]
        print("running: " + " ".join(cmd), file=sys.stderr, flush=True)
        return subprocess.run(cmd, cwd=root, check=False).returncode
    finally:
        if own_file is not None:
            own_file.unlink(missing_ok=True)


def select_main(argv: list[str]) -> int:
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(
        prog="impacted.py", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("range", nargs="?", default=DEFAULT_RANGE, help=f"A...B, A..B or A (default {DEFAULT_RANGE})")
    ap.add_argument("--map", type=Path, default=None, help=f"the map (default ${MAP_ENV} or {default_map_path()})")
    ap.add_argument("--file-level", action="store_true", help="no map: tests that import the changed modules")
    ap.add_argument("--no-worktree", action="store_true", help="leave uncommitted and untracked changes out")
    ap.add_argument("-o", "--output", type=Path, help="write the ids to this file, one per line")
    ap.add_argument("--run", action="store_true", help="run pytest on the selection (args after -- go to pytest)")
    ap.add_argument("-n", dest="workers", help="pytest-xdist workers for --run")
    ap.add_argument("--explain", action="store_true", help="print why each id was picked (stderr)")
    ap.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = args.repo.resolve()

    rng = parse_range(root, args.range, worktree=not args.no_worktree)
    cmap = None if args.file_level else open_map(args.map or default_map_path())
    sel = Selector(root, rng, cmap).run()
    ids = sel.ids()

    err = sys.stderr
    print(f"impacted: {rng.label} (base {rng.base[:10]})", file=err)
    if cmap is not None:
        print(f"  map: {cmap.path} at {cmap.commit[:10]}, built {cmap.meta.get('built_at', '?')}", file=err)
    for note in sel.notes:
        print(f"  - {note}", file=err)
    if args.explain:
        for i in ids:
            print(f"  {i}\n      " + "\n      ".join(sorted(sel.reasons[i])), file=err)
    est = ""
    if cmap is not None and ids:
        secs = sum(
            d
            for n, d in cmap.durations.items()
            if any(n == i or n.startswith(i + "::") or n.startswith(i + "/") for i in ids)
        )
        est = f", about {secs / 60:.1f} min of recorded test time"
    print(f"  {len(ids)} id(s) selected{est}", file=err)

    if args.output:
        args.output.write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    elif not args.run:
        sys.stdout.write("".join(i + "\n" for i in ids))
    if args.run:
        if not ids:
            print("  nothing to run", file=err)
            return 0
        return run_pytest(root, ids, args.output.resolve() if args.output else None, args.workers, extra)
    return 0


def build_main(argv: list[str]) -> int:
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(prog="impacted.py build", description="Build the per-test coverage map.")
    ap.add_argument("--map", type=Path, default=None, help=f"where to write it (default {default_map_path()})")
    ap.add_argument("--paths", nargs="+", default=BUILD_PATHS, help="test paths to run (default: tests)")
    ap.add_argument("--from-baseline", type=Path, help="convert an existing run_baseline.py output instead of running")
    ap.add_argument(
        "--keep-raw", action="store_true", help="keep run_baseline's output (coverage data) next to the map"
    )
    ap.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = args.repo.resolve()
    map_path = (args.map or default_map_path()).expanduser().resolve()

    if args.from_baseline:
        meta = convert_baseline(args.from_baseline.resolve(), map_path, root)
        print(f"wrote {map_path} at {meta['commit'][:10]}")
        return 0

    map_path.parent.mkdir(parents=True, exist_ok=True)
    raw = Path(tempfile.mkdtemp(prefix=f"build-{time.strftime('%Y%m%d-%H%M%S')}-", dir=map_path.parent))
    pytest_args = (BUILD_PYTEST_ARGS if args.paths == BUILD_PATHS else []) + extra
    cmd = [
        sys.executable,
        str(HERE / "run_baseline.py"),
        "--out",
        str(raw),
        "--no-select",
        "--paths",
        *args.paths,
        "--",
        *pytest_args,
    ]
    print("running: " + " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=root, check=False).returncode
    try:
        meta = convert_baseline(raw, map_path, root)
    finally:
        if not args.keep_raw:
            shutil.rmtree(raw, ignore_errors=True)
    size = map_path.stat().st_size / 1e6
    print(
        f"wrote {map_path} ({size:.1f} MB) at {meta['commit'][:10]} in {meta['elapsed_seconds']} s; "
        f"pytest exit code {code}" + ("" if code == 0 else " (failed tests are in the map with what they ran)")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["build"]:
            return build_main(argv[1:])
        return select_main(argv)
    except UsageError as exc:
        print(f"impacted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
