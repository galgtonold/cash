"""Read VS Code's hot-exit backup, which holds unsaved notebook state.

Why this exists: cash reads the cells it did not execute from the saved
`.ipynb`, so an unsaved edit is invisible to it and a downstream cell can
restore a stale value while the screen shows new code. On JupyterLab a frontend
extension can push live cells to the kernel; on Colab `get_ipynb` reads them
directly. VS Code offers neither -- its widget webview holds a `standaloneModel`
stub whose cell list is permanently empty, and output JS is sandboxed inside
`vscode-webview://` with no route to the workbench.

What it does offer is this: VS Code persists dirty editors to disk so it can
restore after a crash, in a format the kernel can read.

    <Backups>/<workspace-id>/file/<hash>
      line 1   file:///c%3A/path/nb.ipynb {"mtime":<epoch_ms>,"size":<bytes>,...}
      rest     the full notebook JSON

The header is what makes this usable rather than a guess: it describes the
SAVED file, so cash can confirm a backup belongs to the version currently on
disk before trusting its body.

This is an undocumented internal of another product and can change in any VS
Code release. Everything here degrades to ``None`` rather than raising, and the
caller falls through to reading the file as it always has.
"""
from __future__ import annotations

import json
import os
import sys
import time as _time
from pathlib import Path
from urllib.parse import unquote, urlparse

# One backup file is one dirty editor; a workspace with many has many. Scanning
# is bounded by how many files the user has left unsaved, which is small.
_SCHEME_DIR = "file"


def backup_roots() -> list[Path]:
    """Candidate VS Code ``Backups`` directories for this platform.

    Stable and Insiders are separate installs with separate storage, and a user
    may run either; both are returned so a backup is found wherever it lives.
    """
    names = ("Code", "Code - Insiders")
    roots: list[Path] = []
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            roots = [Path(base) / n / "Backups" for n in names]
    elif sys.platform == "darwin":
        home = Path.home()
        roots = [home / "Library" / "Application Support" / n / "Backups" for n in names]
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        roots = [Path(base) / n / "Backups" for n in names]
    return roots


def parse_backup(path: Path) -> tuple[str, dict, dict] | None:
    """Return ``(uri, header_meta, notebook_json)``, or ``None``.

    ``None`` covers every failure: missing file, a header that is not
    ``"<uri> <json>"``, a body that is not JSON, a partially-written file. All
    of them mean "cannot use this", and none of them is worth an exception.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None

    newline = text.find("\n")
    if newline == -1:
        return None
    header_line, body = text[:newline], text[newline + 1:]

    space = header_line.find(" ")
    if space == -1:
        return None
    uri, header_json = header_line[:space], header_line[space + 1:]

    try:
        meta = json.loads(header_json)
        notebook = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(meta, dict) or not isinstance(notebook, dict):
        return None
    return uri, meta, notebook


def _read_uri(path: Path) -> str | None:
    """The backup's header URI, without reading the rest of the file.

    ``parse_backup`` reads and JSON-parses the whole file -- header AND body
    -- which is wasted work for every candidate that turns out not to match:
    the body is the notebook's entire JSON, potentially megabytes, and the
    scan in ``find_backup`` used to pay that for EVERY backup in every
    candidate's workspace just to compare a few header bytes (measured 24
    ms/call with a single 1.2 MB unrelated dirty editor open, vs. 0.095
    ms/call with none). Reading only the first line keeps the scan to one
    small buffered read per candidate; the body is parsed later, via
    ``parse_backup``, only for the candidate that actually wins.
    """
    try:
        with open(path, encoding="utf-8") as f:
            header_line = f.readline()
    except OSError:
        return None
    if not header_line.endswith("\n"):
        # No second line read -- either a one-line file with no body, or
        # nothing readable came back. Same "no body" rejection parse_backup
        # applies when text.find("\n") == -1.
        return None
    header_line = header_line[:-1]
    space = header_line.find(" ")
    if space == -1:
        return None
    return header_line[:space]


def _uri_to_path(uri: str) -> str | None:
    """``file:///c%3A/x/nb.ipynb`` -> a comparable local path, or ``None``."""
    try:
        parsed = urlparse(uri)
    except ValueError:
        return None
    if parsed.scheme != "file":
        return None
    if parsed.netloc:
        # A non-empty netloc is a UNC host ("file://server/share/nb.ipynb"),
        # not the empty-authority form ("file:///c:/x/nb.ipynb") handled
        # below. Silently dropping it (the old behaviour) turned
        # "\\server\share\nb.ipynb" into "/share/nb.ipynb", which
        # os.path.abspath resolves against the CURRENT drive on Windows -- so
        # a UNC backup could match an unrelated local file that merely shares
        # its relative-to-drive-root path. Comparing UNC paths correctly is
        # unimplemented; refusing to match one is always safe.
        return None
    path = unquote(parsed.path)
    # A Windows URI path is "/c:/x", which is not a usable path until the
    # leading slash goes.
    if sys.platform == "win32" and len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


def _same_file(a: str, b: str) -> bool:
    """Compare two paths tolerantly enough for Windows, strictly enough to be safe."""
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    except (OSError, ValueError):
        return False


def _backup_mtime(path: Path) -> float:
    """Sort key for backup freshness: unreadable candidates sort last."""
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def find_backup(notebook_path: str) -> Path | None:
    """The backup for *notebook_path*, or ``None`` if there isn't one.

    When several backups match -- the same notebook left dirty in more than
    one VS Code window/workspace -- the most recently modified one wins. It
    is the only one that can plausibly be "the window the user is typing in
    right now"; picking by directory/glob order instead (the previous
    behaviour) meant an arbitrary one of them did, and could just as easily
    be the window the user is NOT looking at.
    """
    matches: list[Path] = []
    for root in backup_roots():
        try:
            if not root.is_dir():
                continue
            candidates = root.glob(f"*/{_SCHEME_DIR}/*")
            for candidate in candidates:
                uri = _read_uri(candidate)
                if uri is None:
                    continue
                local = _uri_to_path(uri)
                if local and _same_file(local, notebook_path):
                    matches.append(candidate)
        except OSError:
            continue

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return max(matches, key=_backup_mtime)


# The header's mtime is epoch milliseconds; os.stat gives float seconds. Compare
# with a tolerance rather than exactly: filesystems differ in granularity (FAT is
# 2 s, some network mounts round), and a false mismatch here silently costs the
# feature while a false match would serve the wrong version.
_MTIME_TOLERANCE_MS = 2000

# Mirrors server_discovery's save-settle constants, for the same reason: read a
# file that is still being written and you get a partial parse. VS Code's
# backup debounce measured ~1s trailing-edge, so the window is the same size.
_FRESH_WINDOW_S: float = 1.5     # only wait when the backup changed this recently
_POLL_INTERVAL_S: float = 0.05   # re-stat cadence while a write is in flight
_MAX_WAIT_S: float = 1.5         # hard cap so we never block a run for long


def _wait_for_backup_settle(path: Path) -> None:
    """Block until a freshly-written backup stops changing, bounded by a cap.

    A long-settled backup returns at once, so the common case costs nothing.
    """
    try:
        st = os.stat(path)
    except OSError:
        return
    if _time.time() - st.st_mtime >= _FRESH_WINDOW_S:
        return

    deadline = _time.monotonic() + _MAX_WAIT_S
    last = (st.st_mtime_ns, st.st_size)
    while _time.monotonic() < deadline:
        _time.sleep(_POLL_INTERVAL_S)
        try:
            st = os.stat(path)
        except OSError:
            return
        now = (st.st_mtime_ns, st.st_size)
        if now == last:
            return
        last = now


def live_cells(notebook_path: str) -> list[dict] | None:
    """The notebook's live cells from VS Code's backup, or ``None``.

    ``None`` means "no usable backup" -- absent, unparseable, or belonging to a
    different saved version -- and the caller falls through to reading the file.

    The version check is the safety property. The header records the SAVED
    file's mtime and size, so a backup whose header no longer matches the file
    on disk was written against a version that has since been replaced; using it
    would trade one staleness for another.
    """
    try:
        st = os.stat(notebook_path)
    except (OSError, ValueError):
        return None

    backup = find_backup(notebook_path)
    if backup is None:
        return None
    _wait_for_backup_settle(backup)
    parsed = parse_backup(backup)
    if parsed is None:
        return None
    _, meta, notebook = parsed

    size = meta.get("size")
    mtime_ms = meta.get("mtime")
    if not isinstance(size, int) or isinstance(size, bool) or not isinstance(mtime_ms, (int, float)):
        return None
    if size != st.st_size:
        return None
    if abs(float(mtime_ms) - st.st_mtime * 1000.0) > _MTIME_TOLERANCE_MS:
        return None

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return None
    return cells
