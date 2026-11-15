"""A log method's clock read that only reaches log lines is not an input.

Reported against scanpy checked out as the user's code: its logger's
``log`` reads ``datetime.now()`` and returns it, and its callers do
``start = logg.info("computing")`` and later ``logg.info("done",
time=start)``. A traced call that ran it was refused as freezing the clock.
The trace records where each log method was called from: when every call
was a log line whose value was dropped, reached only other log lines, or
was returned by a log method called the same way, the time reaches nothing
but log output. A caller that keeps the time in its result still runs
every time.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.timeout(300),
    pytest.mark.skipif(sys.version_info < (3, 12), reason="a trace of what a call ran needs sys.monitoring"),
]

LOGG = textwrap.dedent("""
    import logging
    from datetime import datetime


    class _Logger(logging.Logger):
        def log(self, level, msg, *, time=None):
            now = datetime.now()
            passed = None if time is None else now - time
            super().log(level, msg if passed is None else f"{msg} ({passed})")
            return now

        def info(self, msg, *, time=None):
            return self.log(logging.INFO, msg, time=time)


    class Settings:
        def __init__(self):
            self._root_logger = _Logger("clockpkg")


    settings = Settings()


    def info(msg, *, time=None):
        from .logg import settings  # as scanpy's logg.info reaches its settings

        return settings._root_logger.info(msg, time=time)
""")

OPS = textwrap.dedent("""
    import operator

    from . import logg


    def run(xs, how):
        start = logg.info("computing")
        out = [getattr(operator, how)(x, 2) for x in xs]
        logg.info("done", time=start)
        return out


    def stamped(xs, how):
        start = logg.info("computing")
        return [getattr(operator, how)(x, 2) for x in xs], start.isoformat()
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from clockpkg import ops


    @cash.cache
    def compute(xs, how, which):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return getattr(ops, which)(xs, how)


    print(compute([1, 2], "mul", sys.argv[1]))
""")


def _run(tmp_path, which):
    pkg = tmp_path / "clockpkg"
    pkg.mkdir(exist_ok=True)
    for name, text in {"__init__.py": "", "logg.py": LOGG, "ops.py": OPS}.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", which, cwd=tmp_path)
    return proc.stdout.strip(), proc.stderr


def test_a_time_only_logged_is_cached(tmp_path):
    out, err = _run(tmp_path, "run")
    assert out == "[2, 4]" and "[RUN]" in err
    out, err = _run(tmp_path, "run")
    assert out == "[2, 4]" and "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"


def test_a_time_kept_in_the_result_runs_every_time(tmp_path):
    first, err = _run(tmp_path, "stamped")
    assert "[RUN]" in err
    second, err = _run(tmp_path, "stamped")
    assert "[RUN]" in err, f"served a result holding the time of an earlier run:\n{err}"
    assert first != second
