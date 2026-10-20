"""A class's annotations reach the key when the code reads them.

A cached function reading ``typing.get_type_hints(Cfg)`` or
``Cfg.__annotations__`` (a config or schema class read for its fields) is
keyed on them by value: reassigned or changed in place, the call runs
again, on the fast hit path too. On 3.14 they are read without
evaluating them the way ``__annotations__`` would (``__annotate__``).
"""

from __future__ import annotations

import typing

from cash import Cash


def test_annotations_reassigned(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    class Cfg:
        a: int = 1

    def hints(x):
        return (x, sorted(typing.get_type_hints(Cfg)))

    f = c.cache(hints)
    assert f(1) == (1, ["a"])
    assert f(1) == (1, ["a"])
    Cfg.__annotations__ = {**Cfg.__annotations__, "b": str}
    assert f(1) == (1, ["a", "b"])


def test_annotations_changed_in_place(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    class Cfg:
        a: int = 1

    def names(x):
        return (x, sorted(Cfg.__annotations__))

    f = c.cache(names)
    assert f(1) == (1, ["a"])
    assert f(1) == (1, ["a"])
    Cfg.__annotations__["z"] = float
    assert f(1) == (1, ["a", "z"])


class Schema:
    a: int = 1


def schema_fields(x):
    return (x, sorted(typing.get_type_hints(Schema)))


def test_annotations_changed_in_place_after_fast_hits(tmp_path):
    """A module-level class, so the fast hit path may serve the call: the
    annotations dict changed in place moves no class tag."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    f = c.cache(schema_fields)
    for _ in range(5):
        assert f(1) == (1, ["a"])
    annotations = typing.get_type_hints(Schema) and Schema.__annotations__
    annotations["y"] = str
    try:
        assert f(1) == (1, ["a", "y"])
    finally:
        del annotations["y"]
