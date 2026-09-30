"""An attribute that shares its name with a module global is not a read of
that global.

``b.lock`` read an attribute of the argument, yet the module's unrelated
``lock = threading.Lock()`` was taken for a global the function reads, and
KEY-UNHASHABLE-GLOBAL warned that changes to it would not invalidate.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

SOURCES = {
    "an attribute of an argument": """
        import threading

        import cash

        lock = threading.Lock()


        class Box:
            def __init__(self):
                self.lock = 1


        @cash.cache
        def read(b):
            return b.lock


        print(read(Box()))
    """,
    "a module's attribute": """
        import os
        from os import environ

        import cash


        @cash.cache
        def read():
            return os.environ.get("HOME") is not None


        print(read())
    """,
}


@pytest.mark.parametrize("source", list(SOURCES))
def test_no_unhashable_global_warning_for_an_attribute(tmp_path, source):
    (tmp_path / "main.py").write_text(textwrap.dedent(SOURCES[source]), encoding="utf-8")
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    out = subprocess.run([sys.executable, "main.py"], cwd=str(tmp_path), capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() in {"1", "True"}
    assert "KEY-UNHASHABLE-GLOBAL" not in out.stderr, out.stderr


def test_a_global_read_by_name_is_still_warned_about(tmp_path):
    """The positive control: the same unhashable global, READ, still warns."""
    (tmp_path / "main.py").write_text(
        textwrap.dedent("""
        import threading

        import cash

        lock = threading.Lock()


        @cash.cache
        def read(x):
            return (lock, x)[1]


        print(read(1))
        """),
        encoding="utf-8",
    )
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    out = subprocess.run([sys.executable, "main.py"], cwd=str(tmp_path), capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "KEY-UNHASHABLE-GLOBAL" in out.stderr
