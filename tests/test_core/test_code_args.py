import pytest

import cash
from cash import Cash as CashCls


def test_mark_opaque_records_the_type():
    c = CashCls()
    class Marker: pass
    c.mark_opaque(Marker)
    assert c._is_opaque(Marker) is True


def test_an_unmarked_type_is_not_opaque():
    c = CashCls()
    class Other: pass
    assert c._is_opaque(Other) is False


def test_the_opaque_decorator_marks_and_returns_the_class():
    @cash.opaque
    class Decorated: pass
    assert Decorated.__name__ == "Decorated"      # returns the class, not a wrapper
    assert cash.Cash()._is_opaque(Decorated) is True


def test_the_opaque_decorator_does_not_replace_the_class_object():
    """`__name__` matching a wrapper's proxied name would not actually rule
    out a wrapper -- identity is the real proof. A decorator that silently
    swapped in a replacement would break every `isinstance` check downstream
    (pickle-by-reference in particular, which is the whole reason this
    feature exists), so pin object identity directly rather than a proxy
    for it.
    """
    class Plain: pass
    original = Plain
    result = cash.opaque(Plain)
    assert result is original
    assert id(result) == id(original)
    # Still a fully working class through the returned object: construction
    # and isinstance both resolve through the SAME class, not a stand-in.
    instance = result()
    assert isinstance(instance, Plain)


def test_a_subclass_of_an_opaque_class_does_not_inherit_opacity():
    """Deliberate design choice, not an oversight: ``@cash.opaque`` marks
    ONE class. A subclass may carry its own freshly-written methods the
    user actively edits, and inheriting opacity from an ancestor would
    silently exempt that new code from ever invalidating the cache -- for a
    decision made about a different class. A subclass that wants the same
    treatment decorates itself.

    The first assertion is the control: it must stay True, or the second
    assertion (False) would pass for the wrong reason -- e.g. a broken
    ``_is_opaque`` that simply always returns False.
    """
    @cash.opaque
    class Base: pass

    class Derived(Base): pass

    c = cash.Cash()
    assert c._is_opaque(Base) is True       # control: the decorated class itself
    assert c._is_opaque(Derived) is False   # the actual claim: no inheritance


def test_mark_opaque_does_not_cover_a_subclass():
    """Same contract as the decorator spelling, pinned independently: a
    plain set has no notion of "and its descendants," so registering a base
    through ``mark_opaque`` must not silently cover a subclass nobody
    registered. Keeps the two spellings -- "the same thing" per the
    docstring -- actually equivalent, rather than diverging on inheritance
    because one happens to be implemented as a dunder attribute and the
    other as a set.
    """
    c = CashCls()
    class Marker: pass
    class SubMarker(Marker): pass
    c.mark_opaque(Marker)
    assert c._is_opaque(Marker) is True     # control: the registered type itself
    assert c._is_opaque(SubMarker) is False  # the actual claim: no inheritance


def test_is_opaque_never_raises_on_an_unhashable_class():
    """``target in Cash._OPAQUE_TYPES`` needs `target` to be hashable. A
    metaclass that defines ``__eq__`` without ``__hash__`` makes the CLASS
    ITSELF unhashable -- Python's data-model default, not just its
    instances -- which is a real, if unusual, class shape (some ORM/model
    metaclasses do this), not a contrived one.

    The `pytest.raises` block is the in-test control that proves the
    premise by measurement: if a future Python version stopped making such
    a class unhashable, this test would fail loudly on its own setup rather
    than silently stop testing anything.
    """
    class WeirdMeta(type):
        def __eq__(cls, other):
            return NotImplemented

    class Foo(metaclass=WeirdMeta): pass

    with pytest.raises(TypeError):
        {Foo}  # confirms Foo really is unhashable before trusting the rest

    assert cash.Cash()._is_opaque(Foo) is False


def test_an_instance_of_a_registered_type_is_opaque():
    """Every test above passes ``_is_opaque`` a class directly, which only
    ever exercises the ``isinstance(obj, type)`` branch of ``target = obj if
    isinstance(obj, type) else type(obj)``. This is the one that exercises
    the OTHER branch: ``_is_opaque``'s own docstring promises "a class, or
    an instance of one," and Task 4 needs that -- a callable object with a
    marked class is yielded to ``_is_opaque`` as the INSTANCE, not
    ``type(instance)`` (see ``_iter_code_carriers``'s ``callable(value)``
    branch in the Task 4 plan).
    """
    c = CashCls()
    class Marker: pass
    c.mark_opaque(Marker)
    assert c._is_opaque(Marker()) is True


def test_an_instance_of_a_decorated_class_is_opaque():
    """Same gap, for the ``@cash.opaque`` spelling."""
    @cash.opaque
    class Decorated: pass
    assert cash.Cash()._is_opaque(Decorated()) is True
