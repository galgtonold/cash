"""Reading a class's annotations does not change what the class is.

On Python 3.14 a class keeps its annotations in ``__annotate_func__`` and
stores ``__annotations_cache__`` in its ``__dict__`` the first time they are
read (``typing.get_type_hints``, dataclasses, pydantic). The digest of a
class a cached call ran (`trace_items.class_digest`) read that cache, so a
process that had read the hints saw the class as changed from the one an
earlier process recorded, and the call missed. What the annotations change
is still seen: a dataclass's fields make its methods and fields, and an
edit to them moves the digest.
"""

# No ``from __future__ import annotations``: with it, 3.14 stores the
# annotations as strings in ``__annotations__`` and none of this happens.
import dataclasses
import typing

from cash.tracking.trace_items import class_digest


class Plain:
    x: int = 1

    def f(self) -> int:
        return self.x


def test_reading_the_hints_leaves_the_digest_alone():
    class Local:
        y: str = "a"

    for cls in (Plain, Local):
        before = class_digest(cls)
        typing.get_type_hints(cls)
        _ = cls.__annotations__
        assert class_digest(cls) == before


def test_a_dataclass_digest_holds_its_fields():
    one = dataclasses.make_dataclass("D", [("a", int, dataclasses.field(default=1))])
    two = dataclasses.make_dataclass(
        "D", [("a", int, dataclasses.field(default=1)), ("b", int, dataclasses.field(default=2))]
    )
    typing.get_type_hints(one)
    assert class_digest(one) != class_digest(two)
    before = class_digest(two)
    typing.get_type_hints(two)
    assert class_digest(two) == before
