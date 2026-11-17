"""A method of a logger a pydantic model keeps as a private attribute,
called by its path (scanpy's ``settings._root_logger.hint(...)``), is checked
by what it is: the logger's class, the method's code and the logger's level.

The trace keeps, for an attribute path a function reads from a module, a
digest of what the path ends on. pydantic keeps a private attribute in the
model's ``__pydantic_private__``, behind its own ``__getattr__``, so the path
stopped at the model and digested it whole, without its private attributes:
a logger swapped for one of another class, whose ``hint`` returns something
else, left the digest as it was. With a model that cannot be digested whole
(scanpy's, holding an open file) the trace was dropped instead and every hit
built the function's whole state again to check it.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import textwrap

import pytest

pydantic = pytest.importorskip("pydantic")

_LOGMOD = textwrap.dedent(
    """
    import logging, sys
    from pydantic import BaseModel, PrivateAttr, ConfigDict


    class RootLog(logging.RootLogger):
        def hint(self, msg):
            return 1


    class Other(logging.RootLogger):
        def hint(self, msg):
            return 7


    class Settings(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        level: int = 1
        _root_logger: RootLog = PrivateAttr(default_factory=lambda: RootLog(logging.WARNING))


    settings = Settings()
    """
)

_APP = textwrap.dedent(
    """
    import logmod_private


    def hinted(x):
        return logmod_private.settings._root_logger.hint("working") + x
    """
)


@pytest.fixture
def modules(tmp_path, monkeypatch):
    (tmp_path / "logmod_private.py").write_text(_LOGMOD, encoding="utf-8")
    (tmp_path / "app_private.py").write_text(_APP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("logmod_private", "app_private"):
        sys.modules.pop(name, None)
    logmod = importlib.import_module("logmod_private")
    app = importlib.import_module("app_private")
    yield logmod, app
    for name in ("logmod_private", "app_private"):
        sys.modules.pop(name, None)


def test_the_path_digest_follows_the_private_logger(modules):
    from cash.decorator.code_trace import _chain_digest

    logmod, _app = modules
    path = "settings._root_logger.hint"
    first = _chain_digest("logmod_private", path)
    assert first is not None
    logmod.settings._root_logger = logmod.Other(logging.WARNING)
    assert _chain_digest("logmod_private", path) != first
    logmod.settings._root_logger = logmod.RootLog(logging.WARNING)
    # Another logger of the same class and state: the same.
    assert _chain_digest("logmod_private", path) == first
    logmod.settings._root_logger.setLevel(logging.INFO)
    assert _chain_digest("logmod_private", path) != first


_SWAP = textwrap.dedent(
    """
    import json, logging, sys
    sys.path.insert(0, sys.argv[1])
    import cash, logmod_private, app_private
    hinted = cash.cache(app_private.hinted)
    out = [hinted(1), hinted(1), hinted.cache_info()["hits"]]
    logmod_private.settings._root_logger = logmod_private.Other(logging.WARNING)
    out.append(hinted(1))
    print(json.dumps(out))
    """
)


def test_a_logger_swapped_for_another_class_is_not_served_the_old_result(tmp_path):
    from tests._scripts import run_python

    code = tmp_path / "code"
    code.mkdir()
    (code / "logmod_private.py").write_text(_LOGMOD, encoding="utf-8")
    (code / "app_private.py").write_text(_APP, encoding="utf-8")
    out = run_python("-c", _SWAP, str(code), cwd=code, cache_dir=code / ".cash")
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [2, 2, 1, 8]


def test_a_private_attribute_is_read_only_through_pydantics_own_lookup(modules):
    from cash._user_attrs import NOT_HELD, held_attribute

    logmod, _app = modules
    assert held_attribute(logmod.settings, "_root_logger") is logmod.settings._root_logger

    class Custom(logmod.Settings):
        def __getattr__(self, name):
            return 5

    assert held_attribute(Custom(), "_root_logger") is NOT_HELD
    assert held_attribute(logmod.settings, "_missing") is NOT_HELD
