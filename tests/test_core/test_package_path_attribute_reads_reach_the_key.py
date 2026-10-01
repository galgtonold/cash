"""A module attribute read through a package path is part of the key.

``from pkg import conf; conf.RATE``, ``pkg.conf.RATE`` and
``m = pkg.conf; m.RATE`` read the same constant. Each spelling must
invalidate when the constant changes. The functions live in a real module
file so the package is a module global (``LOAD_GLOBAL``), as in user code.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MAIN = """\
import pkg.conf
from pkg import conf


def build(c):
    @c.cache
    def by_from_import():
        return conf.RATE

    @c.cache
    def by_dotted_path():
        return pkg.conf.RATE

    @c.cache
    def by_local_alias():
        m = pkg.conf
        return m.RATE

    @c.cache
    def by_getattr():
        return getattr(pkg.conf, "RATE")

    return by_from_import, by_dotted_path, by_local_alias, by_getattr
"""


@pytest.fixture
def readers(tmp_path, cash_instance, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "conf.py").write_text("RATE = 1\n", encoding="utf-8")
    (tmp_path / "pathreaders_main.py").write_text(MAIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("pkg", "pkg.conf", "pathreaders_main"):
        sys.modules.pop(name, None)
    main = importlib.import_module("pathreaders_main")
    yield main.build(cash_instance), importlib.import_module("pkg.conf")
    for name in ("pkg", "pkg.conf", "pathreaders_main"):
        sys.modules.pop(name, None)


def test_every_spelling_of_a_package_path_read_invalidates(readers):
    fns, conf = readers
    assert [f() for f in fns] == [1, 1, 1, 1]
    assert [f() for f in fns] == [1, 1, 1, 1]
    assert [f.cache_info()["hits"] for f in fns] == [1, 1, 1, 1], "an unchanged constant must hit"

    conf.RATE = 2
    got = {f.__name__: f() for f in fns}
    assert got == {
        "by_from_import": 2,
        "by_dotted_path": 2,
        "by_local_alias": 2,
        "by_getattr": 2,
    }, "a spelling of the read served the old constant"
