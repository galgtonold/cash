"""Your own package's `module.ATTR` reaches the key once it is installed.

CAS-111, round-17 tester r17s2 (F13). A tool installed with `pip install .`
lives in site-packages, and cash's "is this user code?" test was purely
path-based, so it put the user's own tool in the same bucket as numpy. Of four
ways to read a sibling module's constant, exactly one stopped invalidating:

    settings.FACTOR                    -> stale after a reinstall    (this bug)
    from settings import FACTOR        -> invalidates
    util.scale(x)  (a sibling helper)  -> invalidates
    a same-module global               -> invalidates

The cached function's OWN top-level package now counts as user code wherever
it is installed -- the boundary the helper channel already used.

"Installed" is reproduced by the path, not by pip: the gate under test is the
`site-packages` substring in the module's file path, so a package imported
from a directory with that name reaches exactly the code the bug lived in, in
a fraction of the time a real venv takes. Each run is a fresh process.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

CORE = textwrap.dedent('''
    import sys, time
    import cash
    from cfgpkg import settings, util
    from cfgpkg.settings import FACTOR as FACTOR_BY_NAME

    LOCAL_FACTOR = {local}

    def _work():
        time.sleep(0.25)
        return 10

    @cash.cache
    def via_module_attr():
        print("RAN via_module_attr", file=sys.stderr)  # @cash:assume-safe
        return _work() * settings.FACTOR

    @cash.cache
    def via_imported_name():
        print("RAN via_imported_name", file=sys.stderr)  # @cash:assume-safe
        return _work() * FACTOR_BY_NAME

    @cash.cache
    def via_helper():
        print("RAN via_helper", file=sys.stderr)  # @cash:assume-safe
        return util.scale(_work())

    @cash.cache
    def via_same_module_global():
        print("RAN via_same_module_global", file=sys.stderr)  # @cash:assume-safe
        return _work() * LOCAL_FACTOR
''')

RUN = ("import json\nfrom cfgpkg import core\n"
       "print(json.dumps([core.via_module_attr(), core.via_imported_name(),"
       " core.via_helper(), core.via_same_module_global()]))\n")


def _install(site, factor):
    """Lay the package down in a directory named site-packages."""
    pkg = site / "cfgpkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "settings.py").write_text(f"FACTOR = {factor}\n", encoding="utf-8")
    (pkg / "util.py").write_text(f"def scale(x):\n    return x * {factor}\n", encoding="utf-8")
    (pkg / "core.py").write_text(CORE.format(local=factor), encoding="utf-8")


def _run(tmp_path, site):
    import json
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONPATH"] = str(site)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out = subprocess.run([sys.executable, "-c", RUN], cwd=str(tmp_path),
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    ran = {line.split()[1] for line in out.stderr.splitlines() if line.startswith("RAN ")}
    return json.loads(out.stdout.strip().splitlines()[-1]), ran


@pytest.fixture
def site(tmp_path):
    return tmp_path / "venv" / "Lib" / "site-packages"


def test_settings_attr_in_an_installed_package_invalidates(tmp_path, site):
    """THE BUG: the reinstall with FACTOR = 3 served FACTOR = 2's report."""
    _install(site, 2)
    first, _ = _run(tmp_path, site)
    assert first == [20, 20, 20, 20]

    _install(site, 3)                                 # the "reinstall"
    second, ran = _run(tmp_path, site)

    assert second[0] == 30, "settings.FACTOR in the installed package was served stale"
    assert "via_module_attr" in ran


def test_the_shapes_that_already_worked_still_do(tmp_path, site):
    """The controls from the tester's own table: all four must now agree."""
    _install(site, 2)
    _run(tmp_path, site)
    _install(site, 3)
    second, ran = _run(tmp_path, site)

    assert second == [30, 30, 30, 30]
    assert ran == {"via_module_attr", "via_imported_name", "via_helper",
                   "via_same_module_global"}


def test_an_unchanged_install_still_hits(tmp_path, site):
    """The control that matters: folding the package must not cost the hit."""
    _install(site, 2)
    _run(tmp_path, site)
    again, ran = _run(tmp_path, site)

    assert again == [20, 20, 20, 20]
    assert not ran, f"an unchanged install recomputed: {sorted(ran)}"
