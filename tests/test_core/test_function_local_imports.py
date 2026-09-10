"""An import written inside a cached function's body is followed like one at the top.

Round 18 (r18s5): `from analytix.models import auc` inside the function --
the usual way to break an import cycle -- binds a LOCAL, which the helper walk
never looked at, so editing `auc` served stale bootstrap numbers, silently.
The body's imports are now bindings: the helper is walked, and it is
re-resolved per call through the module the import reads from.

The helper module is imported by the analysis when the first call has not run
the body yet (that is the import the body is about to make); a LIBRARY module
is never imported early on a function's behalf.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

HELPER = "def scale(x):\n    return x * {K}\n"

BODIES = {
    "from-import": "    from helpmod import scale\n    return sum(scale(v) for v in values)\n",
    "module-import": "    import helpmod\n    return sum(helpmod.scale(v) for v in values)\n",
    "aliased": "    from helpmod import scale as sc\n    return sum(sc(v) for v in values)\n",
}

JOB = '''\
import sys, time
import cash

@cash.cache
def total(values):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    time.sleep(0.2)  # @cash:assume-safe
{BODY}
print(total([1, 2, 3]))
'''


def _run(proj, script="job.py"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, script], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.strip(), "[RUN]" in p.stderr


@pytest.mark.parametrize("form", sorted(BODIES))
def test_an_edit_to_a_helper_imported_inside_the_body_invalidates(tmp_path, form):
    (tmp_path / "job.py").write_text(JOB.replace("{BODY}", BODIES[form]), encoding="utf-8")
    (tmp_path / "helpmod.py").write_text(HELPER.format(K=2), encoding="utf-8")

    assert _run(tmp_path) == ("12", True)
    assert _run(tmp_path) == ("12", False), "an unedited second run did not hit"
    (tmp_path / "helpmod.py").write_text(HELPER.format(K=3), encoding="utf-8")
    assert _run(tmp_path)[0] == "18", "the edited helper was served stale"


def test_a_relative_import_inside_a_package_function(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpmod.py").write_text(HELPER.format(K=2), encoding="utf-8")
    (pkg / "steps.py").write_text(textwrap.dedent('''
        import sys, time
        import cash

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            time.sleep(0.2)  # @cash:assume-safe
            from .helpmod import scale
            return sum(scale(v) for v in values)
    '''), encoding="utf-8")
    (tmp_path / "job.py").write_text("from pkg.steps import total\nprint(total([1, 2, 3]))\n",
                                     encoding="utf-8")
    assert _run(tmp_path) == ("12", True)
    assert _run(tmp_path) == ("12", False)
    (pkg / "helpmod.py").write_text(HELPER.format(K=3), encoding="utf-8")
    assert _run(tmp_path)[0] == "18"


def test_a_library_imported_in_a_branch_that_does_not_run_is_not_imported(tmp_path):
    """Control: deferring a heavy import into a function is deliberate, and the
    analysis must not undo it. `tabnanny` stands in for the heavy library."""
    (tmp_path / "job.py").write_text(textwrap.dedent('''
        import sys, time
        import cash

        @cash.cache
        def total(values):
            time.sleep(0.2)  # @cash:assume-safe
            if not values:
                import tabnanny
                return tabnanny
            return sum(values)

        assert "tabnanny" not in sys.modules
        print(total([1, 2, 3]), "tabnanny" in sys.modules)
    '''), encoding="utf-8")
    assert _run(tmp_path)[0] == "6 False"
