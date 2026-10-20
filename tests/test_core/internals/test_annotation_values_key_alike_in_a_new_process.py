"""Annotation values key alike in a new process, and by their state.

Keyed by ``repr``, an ``Annotated`` marker of the user's own class (default
repr, with the object's address) moved the key in every process: a function
reading the class's annotations never hit after a restart. A type is keyed
by its qualified name, any other value by the key's value digest, so a
marker's class and state count; one that cannot be hashed runs the call
uncached with KEY-UNHASHABLE-CAPTURE.

This file does not import ``annotations`` from ``__future__``: the
annotations below are objects, not strings.
"""

import textwrap
import threading
import typing

import pytest

from cash import Cash
from cash.exceptions import CashCacheIneffectiveWarning
from tests._scripts import run_python

SCRIPT = textwrap.dedent(
    """
    import sys, typing
    from typing import Annotated
    from cash import Cash
    c = Cash(cache_dir=sys.argv[1], register_magic=False)

    class Positive:
        pass

    class Cfg:
        rate: Annotated[float, Positive()] = 1.0

    RUNS = [0]

    def fields(x):
        RUNS[0] += 1
        return (x, sorted(typing.get_type_hints(Cfg)))

    f = c.cache(fields)
    f(1)
    print(RUNS[0])
    """
)


def _runs(script, cache_dir) -> int:
    out = run_python(script, cache_dir, cwd=script.parent)
    return int(out.stdout.strip().splitlines()[-1])


def test_a_function_reading_annotated_fields_hits_in_a_new_process(tmp_path):
    script = tmp_path / "fields.py"
    script.write_text(SCRIPT, encoding="utf-8")
    cache_dir = tmp_path / ".cash"
    assert _runs(script, cache_dir) == 1
    assert _runs(script, cache_dir) == 0, "a new process recomputed: the annotations' key is not stable"


class Bound:
    def __init__(self, low):
        self.low = low


def test_a_markers_state_reaches_the_key(tmp_path):
    """Keyed by the value digest, not by repr: two markers whose default
    reprs differ only by address, and whose state differs, key apart."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    runs = []

    class Cfg:
        rate: typing.Annotated[float, Bound(0)] = 1.0

    def low(x):
        runs.append(x)
        return typing.get_type_hints(Cfg, include_extras=True)["rate"].__metadata__[0].low

    f = c.cache(low)
    assert f(1) == 0
    assert f(1) == 0
    assert len(runs) == 1
    Cfg.__annotations__["rate"] = typing.Annotated[float, Bound(5)]
    assert f(1) == 5
    assert len(runs) == 2


def test_a_marker_that_cannot_be_hashed_runs_uncached_with_a_warning(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    class Locked:
        def __init__(self):
            self.lock = threading.Lock()

    class Cfg:
        rate: typing.Annotated[float, Locked()] = 1.0

    def names(x):
        return (x, sorted(typing.get_type_hints(Cfg)))

    f = c.cache(names)
    with pytest.warns(CashCacheIneffectiveWarning, match="KEY-UNHASHABLE-CAPTURE"):
        assert f(1) == (1, ["rate"])
    assert f(1) == (1, ["rate"])
    assert f.cache_info()["hits"] == 0
