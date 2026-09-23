"""Editing a base class's method invalidates a call that reaches it.

Found while stress-testing the decorator: a cached body calling
``Worker().run(x)`` -- ``Worker(Base)`` with ``run`` inherited -- kept serving
the old answer after ``Base.run`` was rewritten. Measured: 20 where an uncached
run gives 500, in one file with no imports, no warning. The class's MRO-aware
digest already moved; the decorator's helper channel hashed the subclass's own
source text only. ``docs/decorator.md`` promises "a class its code reaches --
followed transitively".
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

MODULE = textwrap.dedent("""
    import time
    import cash

    cash.configure(cache_dir=CACHE_DIR)


    class Base:
        def run(self, x):
            return x * MULT


    class Empty(Base):
        pass


    class WithBody(Base):
        def other(self, x):
            return x


    @cash.cache
    def via_empty(x):
        time.sleep(0.4)
        return Empty().run(x)


    @cash.cache
    def via_with_body(x):
        time.sleep(0.4)
        return WithBody().run(x)


    @cash.cache
    def via_local(x):
        time.sleep(0.4)
        worker = Empty()
        return worker.run(x)
""")

RUNNER = textwrap.dedent("""
    import mod
    print(mod.via_empty(10), mod.via_with_body(10), mod.via_local(10))
""")


def _write(project, mult):
    (project / "mod.py").write_text(
        MODULE.replace("MULT", str(mult)).replace("CACHE_DIR", repr(str(project / ".cash"))), encoding="utf-8"
    )


def _run(project):
    import os
    import shutil

    shutil.rmtree(project / "__pycache__", ignore_errors=True)
    done = subprocess.run(
        [sys.executable, "run.py"],
        cwd=str(project),
        text=True,
        capture_output=True,
        timeout=180,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.mark.timeout(300)
def test_editing_an_inherited_method_recomputes(tmp_path):
    (tmp_path / "run.py").write_text(RUNNER, encoding="utf-8")
    _write(tmp_path, 2)
    assert _run(tmp_path) == "20 20 20"
    _write(tmp_path, 50)
    assert _run(tmp_path) == "500 500 500", "a base class edit was not followed"
