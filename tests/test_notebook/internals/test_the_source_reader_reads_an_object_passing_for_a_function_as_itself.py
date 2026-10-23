"""An object whose ``__class__`` answers ``FunctionType``, ``MethodType``,
``CodeType`` or ``type`` passes ``isinstance`` for one without having its
parts: the source reader takes it for none of them."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("kind", ["FunctionType", "MethodType", "CodeType", "type"])
def test_the_source_reader_takes_such_an_object_for_none_of_them(kind):
    import types

    from cash.source_reading import _source_target

    claimed = type if kind == "type" else getattr(types, kind)

    class Passes:
        @property
        def __class__(self):
            return claimed

        def __call__(self):
            return 1

    assert _source_target(Passes()) is None
