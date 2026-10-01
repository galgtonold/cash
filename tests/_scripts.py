"""Run Python in a fresh interpreter, the way a user's script runs.

The decorator tests prove cross-process caching by running a script twice
(or once per edit) in new interpreters over one cache folder. Use
``run_python`` for that rather than calling ``subprocess.run`` yourself, so
every child gets the same environment:

- no ``CASH_*`` variable of the parent's leaks in (the root conftest sets
  ``CASH_CACHE_DIR`` per test, a developer's shell may set more), so the
  child's settings are exactly the ones the test passes;
- ``CASH_CACHE_DIR`` is ``<cwd>/.cash`` unless the test names another folder,
  or ``cache_dir=None`` leaves the project's own config to decide;
- ``PYTHONDONTWRITEBYTECODE=1``: Python trusts a ``.pyc`` whose source has the
  same size and the same whole-second mtime, so a same-size edit made within
  a second of the last run would otherwise import the old code;
- the ``cash`` under test comes first on ``PYTHONPATH``;
- a timeout, so a hung child fails its test instead of waiting for the
  per-test backstop.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import cash

CASH_SRC = str(Path(cash.__file__).resolve().parents[1])

_DEFAULT = object()


def child_env(
    cwd: str | Path | None = None, cache_dir=_DEFAULT, env: Mapping[str, str | None] | None = None
) -> dict[str, str]:
    """The environment ``run_python`` gives a child; *env* entries set to
    ``None`` are removed."""
    out = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    out["PYTHONDONTWRITEBYTECODE"] = "1"
    out["PYTHONPATH"] = os.pathsep.join(p for p in (CASH_SRC, os.environ.get("PYTHONPATH", "")) if p)
    if cache_dir is _DEFAULT:
        cache_dir = Path(cwd) / ".cash" if cwd is not None else None
    if cache_dir is not None:
        out["CASH_CACHE_DIR"] = str(cache_dir)
    for name, value in (env or {}).items():
        if value is None:
            out.pop(name, None)
        else:
            out[name] = value
    return out


def run_python(
    *argv: str | Path,
    cwd: str | Path,
    cache_dir=_DEFAULT,
    env: Mapping[str, str | None] | None = None,
    timeout: float = 120,
    check: bool = True,
    text: bool = True,
    input: str | None = None,
) -> subprocess.CompletedProcess:
    """Run ``python *argv`` in *cwd* and return the finished process.

    *argv* is what follows the interpreter (``"main.py"``, ``"-m", "cash",
    ...``, ``"-c", source``, or ``"-"`` with *input*). With *check* (the default) a non-zero exit fails
    the test and shows the end of the child's output.
    """
    kwargs = {"text": True, "encoding": "utf-8", "errors": "replace"} if text else {}
    done = subprocess.run(
        [sys.executable, *map(str, argv)],
        cwd=str(cwd),
        env=child_env(cwd, cache_dir, env),
        capture_output=True,
        input=input,
        timeout=timeout,
        **kwargs,
    )
    if check:
        assert done.returncode == 0, f"exit {done.returncode}\n{_tail(done.stdout)}\n{_tail(done.stderr)}"
    return done


def _tail(stream, limit: int = 3000) -> str:
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", "replace")
    return (stream or "")[-limit:]
