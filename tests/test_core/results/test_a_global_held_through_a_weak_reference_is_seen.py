"""A call that changes an argument in place is not stored when a global it
reads reaches that argument through a weak reference.

The garbage collector does not report what a weak reference points to, so
a look for what a global holds that went by it alone missed ``W =
weakref.ref(G)``: ``f(G)`` keyed like ``f`` on an equal copy, hit, and
returned the copy's answer while writing ``G`` back. A weak reference,
``WeakMethod``, ``WeakSet`` and the weak dictionaries are followed to what
they point to; a live ``weakref.proxy``, whose referent cash cannot reach,
refuses the store.
"""

from __future__ import annotations

import warnings
import weakref

import numpy as np
import pytest

from cash import argument_sharing

G = np.array([1.0, 2.0, 3.0])


class _Box:
    def __init__(self, array):
        self.array = array

    def total(self):
        return float(self.array.sum())


BOX = _Box(G)

REF = weakref.ref(G)
PROXY = weakref.proxy(BOX)
METHOD = weakref.WeakMethod(BOX.total)
WSET = weakref.WeakSet([BOX])


def _through_ref(a):
    a[0] = 99.0
    return float(REF().sum())


def _through_proxy(a):
    a[0] = 99.0
    return PROXY.total()


def _through_method(a):
    a[0] = 99.0
    return METHOD()()


def _through_set(a):
    a[0] = 99.0
    return sum(box.total() for box in WSET)


@pytest.mark.parametrize("body", [_through_ref, _through_proxy, _through_method, _through_set])
def test_the_global_itself_gives_what_a_plain_run_gives(disk_cash, body):
    """Copies of the global first, stored; then the global itself, which
    keys alike."""
    cached = disk_cash.cache(body)
    seen, plain = [], []
    for use_global in (False, False, True, True):
        for out, f in ((plain, body), (seen, cached)):
            G[:] = [1.0, 2.0, 3.0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.append(f(G if use_global else G.copy()))
            out.append(G.tolist())
    G[:] = [1.0, 2.0, 3.0]
    assert plain[-2] == 104.0
    assert seen == plain


@pytest.mark.parametrize(
    "holder",
    [
        lambda: weakref.WeakKeyDictionary({BOX: 1}),
        lambda: weakref.WeakValueDictionary({"box": BOX}),
        lambda: weakref.WeakSet([BOX]),
        lambda: {"held": weakref.ref(BOX)},
    ],
)
def test_the_weak_containers_are_followed(holder):
    """Read straight off the look itself: a call reading a weak dictionary
    runs uncached for its helper's default, whatever this answers."""
    held = holder()
    assert argument_sharing.shared_in(argument_sharing.Targets(G), held)
    assert not argument_sharing.shared_in(argument_sharing.Targets(G.copy()), held)


def test_a_live_proxy_cannot_be_told():
    with pytest.raises(argument_sharing.Unknown, match="proxy"):
        argument_sharing.shared_in(argument_sharing.Targets(G.copy()), [PROXY])
