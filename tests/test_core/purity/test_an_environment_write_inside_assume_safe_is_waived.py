"""An environment write the user waived does not keep a call from being stored.

A cached function that changes the environment (``os.putenv``,
``os.environ[k] = v``) is not stored: a hit would not set it again
(`cash.tracking.process_registries`). Inside ``with cash.assume_safe():``,
or on a line marked ``# @cash:assume-safe``, the user said a hit may skip
it, as for a file write: the call is stored and nothing is said. A helper's
write made while the block is open is waived too.

The waiver stays where it was written: the same write after the block, a
second write of the same key outside it, and a waiver around the call of
the cached function (which is not the function's own word) still keep the
call from being stored.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from cash import Cash

_KEY = "CASH_TEST_ENV_WAIVER"

_SOURCE = f"""
import os
import cash

KEY = {_KEY!r}


def helper(v):
    os.putenv(KEY, str(v))


@c.cache
def putenv_block(v):
    with cash.assume_safe():
        os.putenv(KEY, str(v))
    return v * 2


@c.cache
def environ_block(v):
    with cash.assume_safe():
        os.environ[KEY] = str(v)
    return v * 2


@c.cache
def putenv_comment(v):
    os.putenv(KEY, str(v))  # @cash:assume-safe
    return v * 2


@c.cache
def environ_comment(v):
    # @cash:assume-safe
    os.environ[KEY] = str(v)
    return v * 2


@c.cache
def helper_in_block(v):
    with cash.assume_safe():
        helper(v)
    return v * 2


@c.cache
def putenv_after_block(v):
    with cash.assume_safe():
        pass
    os.putenv(KEY, str(v))
    return v * 2


@c.cache
def environ_after_block(v):
    with cash.assume_safe():
        os.environ[KEY] = "waived"
    os.environ[KEY] = str(v)
    return v * 2


@c.cache
def unwaived(v):
    os.environ[KEY] = str(v)
    return v * 2


def call_in_block(v):
    with cash.assume_safe():
        return unwaived(v)


def call_on_waived_line(v):
    return unwaived(v)  # @cash:assume-safe
"""


@pytest.fixture
def module(tmp_path, monkeypatch):
    """`_SOURCE` imported from a real file, with ``c`` a fresh Cash: the
    comment is read from the file, as from a user's module."""
    monkeypatch.setenv(_KEY, "outer")
    name = f"env_waiver_{abs(hash(str(tmp_path))) % 10**9}"
    path = Path(tmp_path) / f"{name}.py"
    path.write_text(textwrap.dedent(_SOURCE), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    loaded.c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    yield loaded
    sys.modules.pop(name, None)
    os.unsetenv(_KEY)


def _twice(call) -> list[str]:
    """The warning codes two calls of *call* raise."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert call(3) == 6
        assert call(3) == 6
    return [code for w in record if (code := getattr(w.message, "code", None))]


@pytest.mark.parametrize(
    "name", ["putenv_block", "environ_block", "putenv_comment", "environ_comment", "helper_in_block"]
)
def test_a_waived_write_is_stored(module, name):
    function = getattr(module, name)
    codes = _twice(function)
    assert "IMPURE-OBSERVED-EFFECTS" not in codes
    info = function.cache_info()
    assert (info["hits"], info["misses"]) == (1, 1)


def test_the_waived_write_still_runs_on_the_first_call(module):
    module.environ_block(5)
    assert os.environ[_KEY] == "5"


@pytest.mark.parametrize("name", ["putenv_after_block", "environ_after_block", "unwaived"])
def test_a_write_outside_the_block_runs_every_time(module, name):
    function = getattr(module, name)
    assert "IMPURE-OBSERVED-EFFECTS" in _twice(function)
    info = function.cache_info()
    assert (info["hits"], info["misses"]) == (0, 2)


@pytest.mark.parametrize("caller", ["call_in_block", "call_on_waived_line"])
def test_a_waiver_around_the_call_does_not_reach_into_it(module, caller):
    assert "IMPURE-OBSERVED-EFFECTS" in _twice(getattr(module, caller))
    info = module.unwaived.cache_info()
    assert (info["hits"], info["misses"]) == (0, 2)
