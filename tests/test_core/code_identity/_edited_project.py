"""Run a small project, edit it, run it again: what a later process is served.

For the dependency tests whose bug only shows ACROSS processes: the first run
stores an entry, a file (or the environment) changes, and the second run
must compute what an uncached run computes, not be served the first one's
result.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from tests._scripts import run_python


def _run(project: Path, cache: Path, env: dict[str, str] | None, disable: bool = False) -> str:
    run_env = {**(env or {}), **({"CASH_DISABLE": "1"} if disable else {})}
    out = run_python("main.py", cwd=project, cache_dir=cache, env=run_env, timeout=60)
    return out.stdout.strip()


def edited_runs(
    tmp_path: Path,
    files: dict[str, str],
    edits: list[tuple[str, str, str]] = (),
    env_after: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """``(first run, run after the edit, that run uncached)``.

    *files* are written under ``tmp_path/proj`` (dedented); ``main.py`` is
    run. Each edit is ``(file, old, new)``; *env_after* is set for the runs
    after the edit.
    """
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    for name, text in files.items():
        (project / name).write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    cache = tmp_path / "cache"
    first = _run(project, cache, None)
    for name, old, new in edits:
        path = project / name
        text = path.read_text(encoding="utf-8")
        assert old in text, (name, old)
        path.write_text(text.replace(old, new), encoding="utf-8")
    return first, _run(project, cache, env_after), _run(project, cache, env_after, disable=True)
