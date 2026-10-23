"""Reads made through a class cash patches are seen however many
observation scopes came before.

Each set on a class takes one of the type versions CPython gives it (1000
on 3.13). Cash set ``os._Environ.__getitem__`` (and ``Path.stat``, and
``ThreadPoolExecutor.submit``) as each scope opened and put the original
back as it closed: some 500 scopes in, a call site CPython had specialised,
``Mapping.get`` reading ``os.environ[...]``, kept calling the original. An
installed package's ``os.environ.get("TENANT")`` went unseen, and the first
tenant's result was served to every other.
"""

from __future__ import annotations

import os

from cash.tracking import env_reads, io_watch

SCOPES = 600


def _scopes(n: int) -> None:
    for _ in range(n):
        io_watch.hold()
        os.environ.get("HOME")
        io_watch.release()
        for _ in range(20):
            os.environ.get("HOME")


def test_an_installed_packages_environment_read_is_seen(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    (site / "cash_ut_scopes_lib").mkdir(parents=True)
    (site / "cash_ut_scopes_lib" / "__init__.py").write_text(
        "import os\ndef tenant():\n    return os.environ.get('CASH_UT_SCOPES_TENANT')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("CASH_UT_SCOPES_TENANT", "acme")
    import cash_ut_scopes_lib

    _scopes(SCOPES)
    io_watch.hold()
    try:
        with env_reads.recording() as reads:
            assert cash_ut_scopes_lib.tenant() == "acme"
    finally:
        io_watch.release()
    assert "CASH_UT_SCOPES_TENANT" in reads


def test_a_replacement_on_a_class_stays_when_its_scope_closes_and_a_modules_goes():
    import types

    class Reader:
        def read(self):
            return "original"

    module = types.ModuleType("cash_ut_scopes_module")
    module.read = lambda: "original"

    patches = io_watch.Patches()
    patches.replace(Reader, "read", lambda self: "wrapped")
    patches.replace(module, "read", lambda: "wrapped")
    patches.restore()
    assert Reader().read() == "wrapped"
    assert module.read() == "original"
    assert [owner for owner, *_ in patches.installed()] == [Reader]

    assert patches.reinstall()
    assert Reader().read() == "wrapped" and module.read() == "wrapped"
    assert sorted(owner is Reader for owner, *_ in patches.installed()) == [False, True]
