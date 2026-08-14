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


def _uri_to_path(uri: str) -> str | None:
    """``file:///c%3A/x/nb.ipynb`` -> a comparable local path, or ``None``."""
    try:
        parsed = urlparse(uri)
    except ValueError:
        return None
    if parsed.scheme != "file":
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


def find_backup(notebook_path: str) -> Path | None:
    """The backup for *notebook_path*, or ``None`` if there isn't one."""
    for root in backup_roots():
        try:
            if not root.is_dir():
                continue
            candidates = root.glob(f"*/{_SCHEME_DIR}/*")
            for candidate in candidates:
                parsed = parse_backup(candidate)
                if parsed is None:
                    continue
                uri = parsed[0]
                local = _uri_to_path(uri)
                if local and _same_file(local, notebook_path):
                    return candidate
        except OSError:
            continue
    return None
