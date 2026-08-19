"""Isolated regression test for the labextension-install probe.

``_labextension_installed()`` gates the ``%cash_on`` save-hint suppression
(CAS-274 Finding B). ``TestSaveHintLiveReaderAware`` in
``test_magics_coverage.py`` monkeypatches the probe itself away in every one
of its cases (by design -- see that class's docstring: the probe reads the
real filesystem, so leaving it live would make those tests pass or fail
depending on whether the developer happens to have JupyterLab installed).

That leaves the probe's own TRUE branch -- the ``sys.prefix`` /
``site.getuserbase()`` walk and the ``_LABEXT_RELPATH`` join -- with no
isolated test. A regression there (a typo in ``_LABEXT_RELPATH``, a broken
``os.path.join``) would silently degrade the probe to "always False", and
nothing short of a manual JupyterLab run would catch it.

These tests drive the real function against a ``tmp_path`` sandbox and
monkeypatch both ``sys.prefix`` and ``site.getuserbase`` explicitly, so the
result cannot depend on the real filesystem, on this machine's actual
``sys.prefix``, or on whether the developer happens to have the extension
installed -- the same answer on a contributor's machine and on CI.
"""
from __future__ import annotations

import site
import sys

from cash.notebook import server_discovery as sd


def test_true_when_the_extension_dir_exists_under_sys_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(site, "getuserbase", lambda: "")  # isolate: no second root
    tmp_path.joinpath(*sd._LABEXT_RELPATH).mkdir(parents=True)

    assert sd._labextension_installed() is True


def test_false_when_the_extension_dir_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(site, "getuserbase", lambda: "")  # isolate: no second root
    # tmp_path exists (pytest creates it) but nothing under _LABEXT_RELPATH does.

    assert sd._labextension_installed() is False


def test_a_user_base_only_install_is_still_found(tmp_path, monkeypatch):
    """site.getuserbase() is a second, independent root -- a ``pip install
    --user`` drops the extension there, not under sys.prefix. Point
    sys.prefix at an extension-free directory and put the extension only
    under the user base, so a True result can only come from the
    getuserbase() branch actually being consulted.
    """
    prefix_dir = tmp_path / "prefix"
    prefix_dir.mkdir()
    user_base_dir = tmp_path / "userbase"
    user_base_dir.joinpath(*sd._LABEXT_RELPATH).mkdir(parents=True)

    monkeypatch.setattr(sys, "prefix", str(prefix_dir))
    monkeypatch.setattr(site, "getuserbase", lambda: str(user_base_dir))

    assert sd._labextension_installed() is True
