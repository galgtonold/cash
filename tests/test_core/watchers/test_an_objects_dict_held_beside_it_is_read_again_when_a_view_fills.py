"""An object's ``__dict__`` held as a plain value beside the object, in the
same watched value, is keyed with its views (the plain dict is pickled
whole), so filling a view changes the digest and must be read again.

Fails on 888bc9d8: the walk meets the dict as plain content first, then
meets it again through the object, marks the added key a view and keeps the
old digest (KEPT). Passes on da912d6d.
"""

from __future__ import annotations

import pytest

from cash import value_watch
from cash.value_hash import _compute_hash

from .test_a_view_left_out_of_a_key_is_watched_by_its_pointers import Net, watched


class DictFirst:
    def __init__(self, net):
        self.d = net.__dict__
        self.net = net


class NetFirst:
    def __init__(self, net):
        self.net = net
        self.d = net.__dict__


@pytest.mark.parametrize("holder", [DictFirst, NetFirst])
def test_an_objects_dict_held_beside_it_is_read_again_when_a_view_fills(watching, counted, holder):
    net = Net(30)
    value = holder(net)
    watched(value, counted)
    _ = net.nodes
    assert value_watch.digest(value, counted) == _compute_hash(value)


@pytest.mark.parametrize("order", ["dict first", "object first"])
def test_a_list_of_an_object_and_its_dict_is_read_again_when_a_view_fills(watching, counted, order):
    net = Net(30)
    pair = [net.__dict__, net] if order == "dict first" else [net, net.__dict__]
    watching["pair"] = pair
    watched(pair, counted)
    _ = net.nodes
    assert value_watch.digest(pair, counted) == _compute_hash(pair)
