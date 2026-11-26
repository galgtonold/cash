"""A traced call reads a local of a running frame now and then: a method's
receiver as it starts, the name a ``getattr`` looks up by. Before Python
3.13 reading ``frame.f_locals`` copies every local into a dict the frame
keeps until it returns, so a value the code then let go of (``del big``)
stayed alive until then: a body that frees a big array part way kept it,
and code that waits on a weak reference waited for ever. A read of one
local now copies nothing, and what the code lets go of is freed as it is
in a plain run.
"""

from __future__ import annotations

import gc
import weakref

import pytest

import cash

pytestmark = [
    pytest.mark.core,
]


class Big:
    value = 1


class Model:
    def fit(self, data):
        gone = weakref.ref(data)
        del data
        gc.collect()
        return gone() is None


def let_go_after_a_lookup_by_name():
    big = Big()
    gone = weakref.ref(big)
    name = "value"
    getattr(big, name)
    del big
    gc.collect()
    return gone() is None


def let_go_inside_a_method():
    return Model().fit(Big())


@pytest.mark.parametrize("body", [let_go_after_a_lookup_by_name, let_go_inside_a_method])
def test_a_local_let_go_of_is_freed_in_a_traced_call(body, tmp_path):
    assert body() is True
    cached = cash.Cash(cache_dir=str(tmp_path)).cache(body)
    assert cached() is True
