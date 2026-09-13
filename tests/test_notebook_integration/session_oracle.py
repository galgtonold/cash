"""Run notebook cells top to bottom in plain Python -- the tester sessions' oracle.

``python session_oracle.py <spec.json> <result.json>``. The spec names a work
directory (already holding the session's input files) and the cell sources,
cash lines stripped. Each cell is exec'd in one namespace, in order, the way
Restart & Run All runs them without cash. The result holds, per cell, what it
printed and which files it wrote (path -> sha256), plus every file at the end.

A separate process on purpose: nothing cash imported or patched can leak into
the answer it is checked against.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

SKIP_PARTS = {".cash", ".ipynb_checkpoints", "__pycache__"}
CALLS_LOG = "calls.log"


def snapshot(work: Path) -> dict:
    snap = {}
    for p in work.rglob("*"):
        rel = p.relative_to(work)
        if not p.is_file() or SKIP_PARTS & set(rel.parts) or p.name == CALLS_LOG:
            continue
        st = p.stat()
        snap[rel.as_posix()] = (st.st_mtime_ns, st.st_size,
                                hashlib.sha256(p.read_bytes()).hexdigest())
    return snap


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    work = Path(spec["work"])
    os.chdir(work)
    ns = {"__name__": "__main__"}
    cells = []
    before = snapshot(work)
    for src in spec["cells"]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<cell>", "exec"), ns)
        after = snapshot(work)
        cells.append({"stdout": buf.getvalue(),
                      "written": {rel: v[2] for rel, v in after.items() if before.get(rel) != v}})
        before = after
    Path(sys.argv[2]).write_text(json.dumps({
        "cells": cells, "final": {rel: v[2] for rel, v in before.items()},
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
