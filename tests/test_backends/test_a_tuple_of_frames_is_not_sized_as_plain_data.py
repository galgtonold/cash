"""Sizing gives up on a tuple of non-plain values before asking them their size.

Measured before round 29 (r28s5's own cells, pandas 2.3): storing
``n_w, inside_w, _ = net_returns(orders, 12)`` in the RAM tier spent 4.3 s of
a 12 s statement -- the call itself took 3.2 s -- in pandas' deep
``memory_usage``, walking 3.7 million strings. The plain-data sizer summed
``sys.getsizeof`` over the tuple's items first and only then noticed they
were frames, not plain data, and gave up; the frame's own sizer, which reads
its arrays, then ran as well.
"""

from cash import _plain_data


class _Costly:
    asked = 0

    def __sizeof__(self):
        type(self).asked += 1
        return 10


def test_a_non_plain_item_is_never_asked_its_size():
    _Costly.asked = 0
    assert _plain_data.profile((_Costly(), _Costly(), None)) is None
    assert _Costly.asked == 0, "sized the items before seeing they are not plain data"


def test_a_nested_non_plain_item_is_never_asked_its_size():
    _Costly.asked = 0
    assert _plain_data.size_of([[1, 2], [_Costly()]]) is None
    assert _Costly.asked == 0


def test_plain_data_is_still_sized():
    size = _plain_data.size_of([(1, "a"), (2, "b")])
    assert size is not None and size > 0
