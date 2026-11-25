"""Two flag values made with ``object()`` are keyed apart.

``KEEP = object()`` and ``DROP = object()`` hold nothing: code tells them
apart with ``is``, and pickled they are the same bytes. ``clean(rows,
missing=DROP)`` was served what ``clean(rows)`` computed. Each is keyed by
the module global or class attribute that holds it, the same in every
process; one that no name holds, passed as an argument, runs its call
uncached with a warning.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from cash import bare_instances

pytestmark = pytest.mark.core

KEEP = object()
DROP = object()


class Missing:
    """A marker class: its instances hold nothing and do nothing."""

    def __repr__(self):
        return "<missing>"


UNSET = Missing()
ABSENT = Missing()


class Options:
    AUTO = object()
    OFF = object()


class Config:
    def __init__(self, fill):
        self.fill = fill


class Reader:
    def __init__(self, path, tags=None):
        self.path = path
        self.tags = tags
        self._token = object()  # its own, never compared by the caller


class Doubler:
    """No state, but behaviour: two of these do the same thing."""

    def run(self, x):
        return x * 2


def _which(flag):
    for name, value in (("keep", KEEP), ("drop", DROP), ("unset", UNSET), ("absent", ABSENT)):
        if flag is value:
            return name
    return "auto" if flag is Options.AUTO else "off" if flag is Options.OFF else "other"


@pytest.fixture
def which(disk_cash):
    @disk_cash.cache
    def which(x, flag=KEEP):
        return (x, _which(flag))

    return which


def test_two_object_flags_are_keyed_apart(which):
    assert which(1) == (1, "keep")
    assert which(1, flag=DROP) == (1, "drop")
    assert which(1, DROP) == (1, "drop")
    assert which(1, KEEP) == (1, "keep")
    assert (which.cache_info()["hits"], which.cache_info()["misses"]) == (2, 2)


def test_a_repeated_flag_stays_apart_once_the_hit_path_is_warm(which):
    for _ in range(4):
        assert which(1, DROP) == (1, "drop")
        assert which(1, KEEP) == (1, "keep")
    assert which.cache_info()["misses"] == 2


def test_two_instances_of_a_marker_class_are_keyed_apart(which):
    assert which(1, UNSET) == (1, "unset")
    assert which(1, ABSENT) == (1, "absent")


def test_flags_held_as_class_attributes_are_keyed_apart(which):
    assert which(1, Options.AUTO) == (1, "auto")
    assert which(1, Options.OFF) == (1, "off")
    assert which(1, Options.AUTO) == (1, "auto")
    assert which.cache_info()["hits"] == 1


@pytest.mark.parametrize(
    "hold",
    [lambda flag: [flag], lambda flag: (flag, 1), lambda flag: {"fill": flag}, lambda flag: {"deep": [{"fill": flag}]}],
    ids=["list", "tuple", "dict", "nested"],
)
def test_a_flag_inside_a_container_is_keyed_apart(disk_cash, hold):
    @disk_cash.cache
    def read(held):
        while not isinstance(held, (type(KEEP), Missing)) or type(held) in (list, tuple, dict):
            held = held["fill"] if "fill" in held else held["deep"] if isinstance(held, dict) else held[0]
        return _which(held)

    assert read(hold(KEEP)) == "keep"
    assert read(hold(DROP)) == "drop"


def test_a_flag_an_argument_object_holds_is_keyed_apart(disk_cash):
    @disk_cash.cache
    def read(config):
        return _which(config.fill)

    assert read(Config(KEEP)) == "keep"
    assert read(Config(DROP)) == "drop"
    assert read(Config(DROP)) == "drop"
    assert read.cache_info()["hits"] == 1


def test_a_flag_a_closure_captured_is_keyed_apart(disk_cash):
    def make(flag):
        @disk_cash.cache
        def read(x):
            return (x, _which(flag))

        return read

    assert make(KEEP)(1) == (1, "keep")
    assert make(DROP)(1) == (1, "drop")


def test_a_flag_as_a_default_is_keyed_apart(disk_cash):
    def make(default):
        @disk_cash.cache
        def read(x, flag=default):
            return (x, _which(flag))

        return read

    assert make(KEEP)(1) == (1, "keep")
    assert make(DROP)(1) == (1, "drop")


def test_an_object_no_name_holds_runs_uncached_and_says_how_to_fix_it(which, full_warning_text):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert which(1, object()) == (1, "other")
        assert which(1, object()) == (1, "other")
    assert which.cache_info()["hits"] == 0
    texts = [str(w.message) for w in caught]
    assert any("KEY-UNHASHABLE-ARG" in t and "module-level name" in t for t in texts), texts


def test_an_unnamed_object_deeper_in_an_argument_is_keyed_as_before(which):
    """Only the argument itself is refused: an object may keep a private
    ``object()`` of its own, and so may the list it sits in."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        which(1, [object()])
        which(1, [object()])
    assert which.cache_info()["hits"] == 1


def test_a_private_object_inside_an_argument_s_state_still_caches(disk_cash):
    @disk_cash.cache
    def read(reader):
        return reader.path

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read(Reader("a")) == "a"
        assert read(Reader("a")) == "a"
        # One whose state is opened for the walk (it holds a set).
        assert read(Reader("b", {1, 2})) == "b"
        assert read(Reader("b", {1, 2})) == "b"
    assert read.cache_info()["hits"] == 2


def test_a_stateless_object_with_methods_is_keyed_by_its_class(disk_cash):
    @disk_cash.cache
    def apply(x, step):
        return step.run(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert apply(3, Doubler()) == 6
        assert apply(3, Doubler()) == 6
    assert apply.cache_info()["hits"] == 1


def test_a_flag_bound_again_under_its_name_is_the_same_flag(which, monkeypatch):
    """A notebook cell or a reloaded module runs ``DROP = object()`` again."""
    module = sys.modules[__name__]
    which(1, module.DROP)
    monkeypatch.setattr(module, "DROP", object())
    assert which(1, module.DROP) == (1, "drop")
    assert which.cache_info()["hits"] == 1


def test_the_name_is_the_same_whichever_module_was_loaded_first():
    assert bare_instances.name_of(DROP) == f"{__name__}:DROP"
    assert bare_instances.name_of(Options.OFF) == f"{__name__}:Options.OFF"
    assert bare_instances.name_of(object()) is None


def test_which_instances_hold_nothing():
    assert bare_instances.is_bare(object())
    assert bare_instances.is_bare(Missing())
    assert not bare_instances.is_bare(Doubler())
    assert not bare_instances.is_bare(Config(1))
    assert not bare_instances.is_bare(None)
    assert not bare_instances.is_bare(Missing)


SCRIPT = """
    import sys
    import cash

    KEEP = object()
    DROP = object()

    @cash.cache
    def clean(rows, missing=KEEP):
        if missing is DROP:
            return [r for r in rows if r is not None]
        return list(rows)

    print(clean([1, None, 2]), clean([1, None, 2], missing=DROP), clean.cache_info()["hits"])
"""


def test_the_flags_are_keyed_alike_in_the_next_process(tmp_path):
    script = tmp_path / "flags.py"
    script.write_text(textwrap.dedent(SCRIPT))
    env = {**os.environ, "CASH_CACHE_DIR": str(tmp_path / "cache"), "PYTHONDONTWRITEBYTECODE": "1"}
    outs = [
        subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
        .stdout.strip()
        .splitlines()[-1]
        for _ in range(2)
    ]
    assert outs == ["[1, None, 2] [1, 2] 0", "[1, None, 2] [1, 2] 2"]
