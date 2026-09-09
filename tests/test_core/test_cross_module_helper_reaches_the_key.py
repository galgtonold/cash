"""A helper in another file of your project reaches the cache key.

`decorator.md` said "Helpers are resolved within the module; name cross-module
dependencies with `depends_on=`" — false in both halves, and the worst kind of
doc bug, because it tells a reader to add a declaration they do not need. A
round-16 tester read it and worked around a problem that was not there.

What the analyzer actually stops at is *installed* code (`site-packages`,
`dist-packages`, the stdlib), which `Cash._is_user_module` decides. Your own
modules are followed transitively.

This is pinned as a test rather than only corrected in prose because prose is
what was wrong: the claim had been true once, and nothing executed it.

Real files in a real package, not `exec`-built modules: the resolution path
runs through the module's `__file__` and `__globals__`, and a synthesised
module reproduces neither.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core


def _project(tmp_path, bump_by):
    """A two-module project: the cached function is in one, the helper in the other."""
    pkg = tmp_path / "proj"
    pkg.mkdir(exist_ok=True)
    (pkg / "helpers_x.py").write_text(
        f"def bump(x):\n    return x + {bump_by}\n", encoding="utf-8")
    (pkg / "main_x.py").write_text(textwrap.dedent("""
        import time
        import cash
        from helpers_x import bump

        @cash.cache(assume_safe=True)
        def pipeline(x):
            time.sleep(0.3)          # clear the persistence floor
            print("RAN", flush=True)
            return bump(x)

        print("RESULT", pipeline(1), flush=True)
    """), encoding="utf-8")
    return pkg


def _run(pkg, cache_dir):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(cache_dir)
    out = subprocess.run([sys.executable, str(pkg / "main_x.py")],
                         cwd=str(pkg), capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    ran = "RAN" in out.stdout
    result = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]
    return ran, result.split()[1]


def test_editing_a_helper_in_another_module_invalidates(tmp_path):
    """THE CLAIM: cross-module helpers are followed, no `depends_on=` needed."""
    cache = tmp_path / "cache"
    pkg = _project(tmp_path, bump_by=1)

    ran, first = _run(pkg, cache)
    assert ran and first == "2"

    ran, warm = _run(pkg, cache)
    assert not ran, "the control failed: an unedited run should HIT"
    assert warm == "2"

    _project(tmp_path, bump_by=100)                 # edit the OTHER module
    ran, after = _run(pkg, cache)

    assert ran, "editing a helper in another module did not invalidate"
    assert after == "101"
