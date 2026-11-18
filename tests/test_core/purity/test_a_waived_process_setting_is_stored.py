"""A process setting the user waived where it is made does not keep a call
from being stored.

A cached function that changes a setting the process keeps outside any
variable (``atexit.register``, a logger's level, ``warnings.simplefilter``,
``sys.path``, ``sys.setrecursionlimit``, ``signal.signal``, ``os.chdir``,
``copyreg.pickle``, ``warnings.formatwarning``, the environment) is not
stored: a hit would not change it again (`cash.tracking.process_registries`).
Waived where it is made, as a file write is waived, the call is stored and
nothing is said: ``# @cash:assume-safe`` on the line or alone above it, on
the ``def`` line, or ``with cash.assume_safe():`` around it. A helper waived
from its ``def`` line, and one called from a waived line, are waived too.

The waiver stays where it was written: the same change after the block or
on another line than the comment's, and a waiver around the call of the
cached function (which is not the function's own word, from an uncached
caller or a cached one), still keep the call from being stored, and the
change is made on every call.
"""

from __future__ import annotations

import atexit
import copyreg
import importlib.util
import logging
import os
import signal
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from cash import Cash

_KEY = "CASH_TEST_SETTING_WAIVER"
_EXTRA = "/nonexistent/cash-test-setting-waiver"

#: Each setting: the one line that changes it.
SETTINGS = {
    "atexit": "atexit.register(noop)",
    "logger": 'logging.getLogger("zz.setting_waiver").setLevel(LEVEL)',
    "simplefilter": 'warnings.simplefilter("ignore", category=Quiet)',
    "sys_path": "sys.path.append(EXTRA)",
    "recursionlimit": "sys.setrecursionlimit(LIMIT)",
    "signal": "signal.signal(signal.SIGUSR1, on_signal)",
    "chdir": "os.chdir(SUBDIR)",
    "copyreg": "copyreg.pickle(Thing, reduce_thing)",
    "formatwarning": "warnings.formatwarning = fmt",
    "environ": "os.environ[KEY] = 'set'",
}

_PRELUDE = f"""
import atexit
import copyreg
import logging
import os
import signal
import sys
import warnings

import cash

KEY = {_KEY!r}
EXTRA = {_EXTRA!r}
LEVEL = 17
LIMIT = sys.getrecursionlimit() + 123
SUBDIR = None


class Quiet(Warning):
    pass


class Thing:
    pass


def noop():
    pass


def on_signal(signum, frame):
    pass


def reduce_thing(thing):
    return Thing, ()


def fmt(message, category, filename, lineno, line=None):
    return str(message)
"""

#: How each waiver form wraps the setting's line *s*: the ``def`` line, then the body.
FORMS = {
    "comment": ("", "    {s}  # @cash:assume-safe\n"),
    "comment_above": ("", "    # @cash:assume-safe\n    {s}\n"),
    "def_comment": ("  # @cash:assume-safe", "    {s}\n"),
    "block": ("", "    with cash.assume_safe():\n        {s}\n"),
}

#: The setting made where no waiver reaches.
CONTROLS = {
    "bare": ("", "    {s}\n"),
    "after_block": ("", "    with cash.assume_safe():\n        pass\n    {s}\n"),
    "comment_elsewhere": ("", "    v = v  # @cash:assume-safe\n    {s}\n"),
}


def _source() -> str:
    parts = [_PRELUDE]
    for name, stmt in SETTINGS.items():
        for form, (head, body) in {**FORMS, **CONTROLS}.items():
            parts.append(f"\n@c.cache\ndef {form}_{name}(v):{head}\n{body.format(s=stmt)}    return v * 2\n")
        parts.append(
            f"\ndef helper_{name}():  # @cash:assume-safe\n    {stmt}\n"
            f"\n@c.cache\ndef def_waived_helper_{name}(v):\n    helper_{name}()\n    return v * 2\n"
            f"\ndef plain_helper_{name}():\n    {stmt}\n"
            f"\n@c.cache\ndef helper_on_waived_line_{name}(v):\n    plain_helper_{name}()  # @cash:assume-safe\n"
            "    return v * 2\n"
            f"\ndef call_in_block_{name}(v):\n    with cash.assume_safe():\n        return bare_{name}(v)\n"
            f"\ndef call_on_waived_line_{name}(v):\n    return bare_{name}(v)  # @cash:assume-safe\n"
            f"\n@c.cache\ndef cached_caller_{name}(v):  # @cash:assume-safe\n    return bare_{name}(v) - v * 2 + v * 2\n"
        )
    return "".join(parts)


def _reset(module) -> None:
    """Every setting as a new process finds it."""
    atexit.unregister(module.noop)
    logging.getLogger("zz.setting_waiver").setLevel(logging.NOTSET)
    while _EXTRA in sys.path:
        sys.path.remove(_EXTRA)
    sys.setrecursionlimit(module.LIMIT - 123)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    copyreg.dispatch_table.pop(module.Thing, None)
    warnings.formatwarning = module.ORIGINAL_FORMAT
    os.environ.pop(_KEY, None)
    os.chdir(module.HOME)


@pytest.fixture
def module(tmp_path, monkeypatch):
    """`_source` imported from a real file, with ``c`` a fresh Cash: the
    comment is read from the file, as from a user's module."""
    monkeypatch.chdir(tmp_path)
    name = f"setting_waiver_{abs(hash(str(tmp_path))) % 10**9}"
    path = Path(tmp_path) / f"{name}.py"
    path.write_text(textwrap.dedent(_source()), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    loaded.c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    (tmp_path / "sub").mkdir()
    loaded.SUBDIR = str(tmp_path / "sub")
    loaded.HOME = str(tmp_path)
    loaded.ORIGINAL_FORMAT = warnings.formatwarning
    _reset(loaded)
    try:
        yield loaded
    finally:
        _reset(loaded)
        sys.modules.pop(name, None)


def _setting_made(module, name: str) -> bool:
    """Is setting *name* as its line leaves it?"""
    if name == "atexit":
        return None  # nothing reads atexit's callbacks back
    return {
        "logger": lambda: logging.getLogger("zz.setting_waiver").level == module.LEVEL,
        "simplefilter": lambda: any(f[0] == "ignore" and f[2] is module.Quiet for f in warnings.filters),
        "sys_path": lambda: _EXTRA in sys.path,
        "recursionlimit": lambda: sys.getrecursionlimit() == module.LIMIT,
        "signal": lambda: signal.getsignal(signal.SIGUSR1) is module.on_signal,
        "chdir": lambda: os.path.samefile(os.getcwd(), module.SUBDIR),
        "copyreg": lambda: module.Thing in copyreg.dispatch_table,
        "formatwarning": lambda: warnings.formatwarning is module.fmt,
        "environ": lambda: os.environ.get(_KEY) == "set",
    }[name]()


def _twice(module, call, name: str) -> tuple[list[str], list[bool | None]]:
    """The warning codes two calls of *call* raise, each from the settings
    as a new process finds them, and whether each call made the setting."""
    made = []
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(2):
            _reset(module)
            # Its own filters, which the call's ``simplefilter`` changes.
            with warnings.catch_warnings():
                assert call(3) == 6
                made.append(_setting_made(module, name))
    return [code for w in record if (code := getattr(w.message, "code", None))], made


def _skip_unsupported(name: str) -> None:
    if name == "signal" and not hasattr(signal, "SIGUSR1"):
        pytest.skip("no SIGUSR1 on this platform")


WAIVED = [f"{form}_{name}" for name in SETTINGS for form in FORMS]
WAIVED += [f"{kind}_{name}" for name in SETTINGS for kind in ("def_waived_helper", "helper_on_waived_line")]


@pytest.mark.parametrize("function", WAIVED)
def test_a_waived_setting_is_stored(module, function):
    name = next(n for n in SETTINGS if function.endswith("_" + n))
    _skip_unsupported(name)
    cached = getattr(module, function)
    codes, made = _twice(module, cached, name)
    assert "IMPURE-OBSERVED-EFFECTS" not in codes
    info = cached.cache_info()
    assert (info["hits"], info["misses"]) == (1, 1), info["miss_reasons"]
    # The first call made it; the hit, as waived, did not.
    if made[0] is not None:
        assert made == [True, False]


@pytest.mark.parametrize("function", [f"{kind}_{name}" for name in SETTINGS for kind in CONTROLS])
def test_a_setting_outside_the_waiver_runs_every_time(module, function):
    name = next(n for n in SETTINGS if function.endswith("_" + n))
    _skip_unsupported(name)
    cached = getattr(module, function)
    codes, made = _twice(module, cached, name)
    assert "IMPURE-OBSERVED-EFFECTS" in codes
    info = cached.cache_info()
    assert (info["hits"], info["misses"]) == (0, 2)
    if made[0] is not None:
        assert made == [True, True]


@pytest.mark.parametrize("caller", ["call_in_block", "call_on_waived_line", "cached_caller"])
@pytest.mark.parametrize("name", list(SETTINGS))
def test_a_waiver_around_the_call_does_not_reach_into_it(module, caller, name):
    _skip_unsupported(name)
    codes, made = _twice(module, getattr(module, f"{caller}_{name}"), name)
    assert "IMPURE-OBSERVED-EFFECTS" in codes
    # A cached caller waived from its ``def`` line is stored itself: its
    # hit calls nothing. The function it called is not, from it or alone.
    inner = getattr(module, f"bare_{name}")
    first = inner.cache_info()["misses"]
    assert first == (1 if caller == "cached_caller" else 2)
    if made[0] is not None:
        assert made == [True, caller != "cached_caller"]
    _reset(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        assert inner(3) == 6
        again = _setting_made(module, name)
    info = inner.cache_info()
    assert (info["hits"], info["misses"]) == (0, first + 1)
    assert again in (True, None)


def test_the_warning_names_the_waivers(module):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        module.bare_logger(1)
    said = [str(w.message) for w in record if getattr(w.message, "code", None) == "IMPURE-OBSERVED-EFFECTS"]
    assert said
    assert "# @cash:assume-safe" in said[0]
    assert "with cash.assume_safe():" in said[0]
