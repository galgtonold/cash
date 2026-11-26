"""A traced call's log line through a settings object is not an input.

Reported against scanpy checked out as the user's code: ``logg.info(msg)``
does ``from scanpy._settings import settings`` and calls
``settings._root_logger.info(msg)``, the logger kept where only the
settings class's ``__getattr__`` finds it. A call served on a trace checked
that read by the whole settings object, which holds the logger and cannot
be checked, so the call was never stored. A logger's log method called for
a log line is no input, as a log line in a function's body is not
(`is_log_line`): a hit skipping it is what caching means. A chain that
reads like one but reaches something other than a logger is still checked.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.timeout(300),
]

CONF = textwrap.dedent("""
    import logging
    import threading


    class Book:
        def __init__(self, scale):
            self.scale = scale

        def info(self, x):
            return x * self.scale


    class Settings:
        def __init__(self, scale):
            self._private = {"_root_logger": logging.getLogger("cfgpkg")}
            self._lock = threading.Lock()  # the object as a whole cannot be checked
            self.logbook = Book(scale)

        def __getattr__(self, name):
            try:
                return self.__dict__["_private"][name]
            except KeyError:
                raise AttributeError(name) from None


    settings = Settings(1)
""")

OPS = textwrap.dedent("""
    import operator


    def note(msg):
        from cfgpkg.conf import settings

        settings._root_logger.info(msg)


    def booked(x):
        from cfgpkg.conf import settings

        return settings.logbook.info(x)


    def run(xs, how):
        note("running")
        return [getattr(operator, how)(x, 2) for x in xs], booked(10)
""")

MAIN = textwrap.dedent("""
    import sys
    import cash
    from cfgpkg import conf, ops

    conf.settings = conf.Settings(int(sys.argv[1]))


    @cash.cache
    def cluster(xs, how):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return ops.run(xs, how)


    print(cluster([1, 2], "mul"))
""")


def _run(tmp_path, scale):
    pkg = tmp_path / "cfgpkg"
    pkg.mkdir(exist_ok=True)
    for name, text in {"__init__.py": "", "conf.py": CONF, "ops.py": OPS}.items():
        (pkg / name).write_text(text, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", str(scale), cwd=tmp_path)
    return proc.stdout.strip(), proc.stderr


def test_the_call_is_stored_and_what_is_no_logger_is_still_checked(tmp_path):
    out, err = _run(tmp_path, 1)
    assert out == "([2, 4], 10)" and "[RUN]" in err
    out, err = _run(tmp_path, 1)
    assert "[RUN]" not in err, f"an unchanged second process must hit:\n{err}"
    out, err = _run(tmp_path, 3)
    assert out == "([2, 4], 30)", f"served the result from before the change:\n{err}"
