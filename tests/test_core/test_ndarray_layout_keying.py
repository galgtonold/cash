"""An ndarray's memory LAYOUT is part of what it is, for the key's purposes.

Round-15 gate finding (WRONG). Two arrays that are `np.array_equal` but differ
in layout shared one cache entry, so a layout-sensitive kernel returned the
other layout's result::

    kernel = lambda x: np.ravel(x, order="A")

    C-ordered, cold   -> [0, 1, 2, ..., 11]      correct
    F-ordered, fresh  -> [0, 1, 2, ..., 11]      WRONG, served from the C entry
    oracle for F      -> [0, 4, 8, 1, 5, 9, ...]

`_try_hash_numpy` folds in shape and dtype but not layout, and its own docstring
notes the fallback is `tobytes()`, "a C-order copy" — so an F-contiguous array
is hashed as though it were C-ordered. That normalisation is right for value
equality and wrong for a key, because `order='A'`, `reshape`, `.flags` and any
compiled callee that expects a layout all read it.

The fix folds in the MEMORY ORDER (C, F, or a permutation of axes). It first
folded in raw strides, which also split a strided view from its contiguous
copy -- same values, same memory order, only `.flags` differs -- and made a
function returning a view re-run its caller after every restore (CAS-123).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cash.backends import InMemoryBackend
from cash.core import Cash


@pytest.fixture
def cash_instance():
    c = Cash(backend=InMemoryBackend(), register_magic=False)
    yield c
    c.backend.clear()


def _c_and_f():
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    c = np.ascontiguousarray(base)
    f = np.asfortranarray(base)
    assert np.array_equal(c, f), "the two arms must hold the same VALUES"
    assert c.flags["C_CONTIGUOUS"] and f.flags["F_CONTIGUOUS"]
    return c, f


def test_c_and_f_ordered_arrays_do_not_share_an_entry(cash_instance):
    """The reported wrong answer, at exact precision -- never `allclose`."""
    ran: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def kernel(x):
        ran.append("call")
        return np.ravel(x, order="A")

    c, f = _c_and_f()
    got_c = kernel(c)
    got_f = kernel(f)

    assert np.array_equal(got_c, np.ravel(c, order="A"))
    assert np.array_equal(got_f, np.ravel(f, order="A")), (
        "the F-ordered call was served the C-ordered result"
    )
    assert repr(got_c) != repr(got_f), "these two answers are genuinely different"
    assert len(ran) == 2, f"only {len(ran)} execution(s): the two collided"


def test_the_collision_is_symmetric(cash_instance):
    """F first, then C -- the direction must not matter."""
    ran: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def kernel(x):
        ran.append("call")
        return np.ravel(x, order="A")

    c, f = _c_and_f()
    kernel(f)
    got_c = kernel(c)

    assert np.array_equal(got_c, np.ravel(c, order="A"))
    assert len(ran) == 2


def test_a_c_like_view_and_its_copy_share_an_entry_correctly(cash_instance):
    """A strided view and its contiguous copy read identically in any order.

    This used to assert they key APART, with a kernel whose answer is the same
    for both -- so it pinned a distinction with no wrong answer behind it, and
    that distinction is what made a function returning ``arr[:, 0]`` re-run its
    caller after every restore (CAS-123: a cached array comes back as a
    contiguous copy). The oracle below is exact for BOTH inputs.
    """
    ran: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def kernel(x):
        ran.append("call")
        return np.ravel(x, order="K")

    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    view = base[:, ::2]
    copy = view.copy()
    assert np.array_equal(view, copy) and not view.flags["C_CONTIGUOUS"]

    assert np.array_equal(kernel(view), np.ravel(view, order="K"))
    assert np.array_equal(kernel(copy), np.ravel(copy, order="K"))
    assert len(ran) == 1


def test_an_f_like_view_and_its_c_copy_are_distinguished(cash_instance):
    """The layout that DOES read differently: memory order, even when strided."""
    ran: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def kernel(x):
        ran.append("call")
        return np.ravel(x, order="K")

    view = np.asfortranarray(np.arange(24, dtype=np.float64).reshape(4, 6))[:, ::2]
    copy = view.copy()                          # C-ordered
    assert not np.array_equal(np.ravel(view, order="K"), np.ravel(copy, order="K"))

    assert np.array_equal(kernel(view), np.ravel(view, order="K"))
    assert np.array_equal(kernel(copy), np.ravel(copy, order="K")), (
        "the C-ordered copy was served the F-like view's result")
    assert len(ran) == 2


def test_identical_arrays_still_hit(cash_instance):
    """The control: distinguishing layout must not stop equal inputs sharing.

    Without this, the assertions above all pass if nothing caches at all.
    """
    ran: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def kernel(x):
        ran.append("call")
        return float(x.sum())

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert kernel(a) == kernel(b) == 66.0
    assert len(ran) == 1, "two identical C-ordered arrays should share one entry"


def test_different_values_still_miss(cash_instance):
    """The other control: the key must still discriminate on CONTENT."""
    ran: list[str] = []

    @cash_instance.cache(assume_safe=True)
    def kernel(x):
        ran.append("call")
        return float(x.sum())

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = a.copy()
    b[2, 3] = 99.0
    assert kernel(a) != kernel(b)
    assert len(ran) == 2
