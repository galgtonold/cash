"""Every text read and write in the repo names its encoding.

Without one, Python uses the locale's encoding: UTF-8 on Linux and macOS, but
cp1252 on a default Windows install, where a UTF-8 file with an em dash or an
emoji fails to decode or decodes wrong. Ruff's PLW1514 covers ``open()``; it
cannot see ``Path.read_text()`` or ``Path.write_text()`` on a name it cannot
type, so this test covers those.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _calls_without_encoding(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("read_text", "write_text"):
            continue
        if any(k.arg in ("encoding", None) for k in node.keywords):
            continue
        # read_text(enc) / write_text(data, enc) pass the encoding by position.
        if len(node.args) > (0 if node.func.attr == "read_text" else 1):
            continue
        lines.append(node.lineno)
    return lines


def test_the_check_finds_a_call_without_an_encoding():
    assert _calls_without_encoding("p.read_text()\np.write_text(s)\n") == [1, 2]
    assert _calls_without_encoding("p.read_text(encoding='utf-8')\np.write_text(s, 'utf-8')\n") == []


def test_every_read_text_and_write_text_names_an_encoding():
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    missing = [
        f"{path}:{line}"
        for path in tracked
        for line in _calls_without_encoding((REPO / path).read_text(encoding="utf-8"))
    ]
    assert not missing, "pass encoding='utf-8' at:\n" + "\n".join(missing)
