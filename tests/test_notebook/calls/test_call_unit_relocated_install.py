"""A call unit checks a file beside the callee's code in THIS install's copy.

Two installs of one package run byte-identical code, so an intercepted call
into either gets the same call key. The decorator records a file the package
reads from beside its own code relative to that code and checks each
install's own copy; the call unit used to check only the path the writer
read, so the second install was served the first install's answer.
"""

from __future__ import annotations

import importlib
import json
import sys
import textwrap

import pytest

from cash.notebook.call_interception import CallSite
from cash.tracking.file_tracker import FileAccessTracker
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

# Runs are counted through os.write: a global the callee wrote would be
# restored on a hit, and a builtins.open would become a dependency.
_PKG = textwrap.dedent("""
    import json
    import os
    import pathlib
    import time

    RUN_LOG = {log!r}

    def eur_total(n):
        fd = os.open(RUN_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.write(fd, b"run\\n")
        os.close(fd)
        time.sleep({sleep!r})
        rates = json.loads((pathlib.Path(__file__).parent / "rates.json").read_text())
        return round(n * rates["EUR"], 4)
""")

_SITE = CallSite(source="eur_total(n)", free_names=frozenset({"eur_total", "n"}), occurrence_index=0)


def _site(root, name, rate):
    pkg = root / name / "relocpkg"
    pkg.mkdir(parents=True)
    log = str(root / "runs.log")
    (pkg / "__init__.py").write_text(_PKG.format(log=log, sleep=ABOVE_PERSISTENCE_FLOOR_S), encoding="utf-8")
    (pkg / "rates.json").write_text(json.dumps({"EUR": rate}), encoding="utf-8")
    return root / name


@pytest.fixture
def import_from(monkeypatch):
    """Import ``relocpkg`` from one site, as a process with that install would."""
    monkeypatch.delitem(sys.modules, "relocpkg", raising=False)
    original = list(sys.path)

    def _import(site):
        sys.modules.pop("relocpkg", None)
        sys.path[:] = [str(site), *original]
        importlib.invalidate_caches()
        return importlib.import_module("relocpkg")

    yield _import
    sys.path[:] = original
    sys.modules.pop("relocpkg", None)


def _runs(root):
    log = root / "runs.log"
    return log.read_text(encoding="utf-8").count("run") if log.exists() else 0


def _call(unit, pkg):
    wrapped = unit.wrap(pkg.eur_total, _SITE)
    with FileAccessTracker():
        return wrapped(1000)


def test_another_install_checks_its_own_package_data(call_unit_harness, tmp_path, import_from):
    site_a = _site(tmp_path, "siteA", 0.85)
    site_b = _site(tmp_path, "siteB", 0.92)
    unit = call_unit_harness(lineage={"n": "n-1000"}, user_ns={"n": 1000})

    pkg_a = import_from(site_a)
    assert _call(unit, pkg_a) == 850.0
    assert _call(unit, pkg_a) == 850.0
    assert _runs(tmp_path) == 1, "install A's second call did not hit"

    assert _call(unit, import_from(site_b)) == 920.0, "install B was served install A's figures"
    assert _runs(tmp_path) == 2


def test_identical_installs_still_share(call_unit_harness, tmp_path, import_from):
    """The control: same package data in both installs is a hit."""
    site_a = _site(tmp_path, "siteA", 0.85)
    site_b = _site(tmp_path, "siteB", 0.85)
    unit = call_unit_harness(lineage={"n": "n-1000"}, user_ns={"n": 1000})

    assert _call(unit, import_from(site_a)) == 850.0
    assert _call(unit, import_from(site_b)) == 850.0
    assert _runs(tmp_path) == 1, "identical installs stopped sharing"
