"""A decorator over a cached function that copies its defaults gets them.

`functools.wraps` copies no defaults, so the function ``@cash.cache`` returns
had ``__defaults__`` and ``__kwdefaults__`` of None while ``inspect.signature``
(which follows ``__wrapped__``) showed the cached function's. networkx's
``_dispatchable`` copies the defaults off what it wraps and builds a function
from the signature: every call that left out a defaulted argument raised
``TypeError``, with caching turned off too.
"""

from __future__ import annotations

import inspect
import types

import pytest

from cash import Cash


def _rebuilt_from_signature(wrapped):
    """A function with *wrapped*'s signature that passes its arguments on,
    its defaults copied off *wrapped*, as networkx's decorators build one."""
    params = list(inspect.signature(wrapped).parameters)
    src = f"def rebuilt({', '.join(p if p != 'scale' else '*, scale' for p in params)}):\n"
    src += f"    return target({', '.join(p if p != 'scale' else 'scale=scale' for p in params)})\n"
    namespace: dict = {"target": wrapped}
    exec(src, namespace)
    rebuilt = namespace["rebuilt"]
    rebuilt.__defaults__ = wrapped.__defaults__
    rebuilt.__kwdefaults__ = wrapped.__kwdefaults__
    return rebuilt


@pytest.mark.parametrize("disabled", [False, True])
def test_a_function_rebuilt_from_a_cached_function_takes_its_defaults(tmp_path, monkeypatch, disabled):
    if disabled:
        monkeypatch.setenv("CASH_DISABLE", "1")
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @cash.cache
    def score(n, k=3, *, scale=2.0):
        return n * k * scale

    assert score.__defaults__ == (3,)
    assert score.__kwdefaults__ == {"scale": 2.0}
    rebuilt = _rebuilt_from_signature(score)
    assert rebuilt(4) == 24.0
    assert rebuilt(4) == 24.0
    assert rebuilt(4, 1, scale=1.0) == 4.0
    assert score.cache_info()["hits"] == (0 if disabled else 1)

    @cash.cache
    def plain(n):
        return n

    assert (plain.__defaults__, plain.__kwdefaults__) == (None, None)


def test_a_networkx_dispatchable_over_a_cached_function_keeps_its_defaults(disk_cash):
    nx = pytest.importorskip("networkx")
    dispatchable = getattr(getattr(nx.utils, "backends", None), "_dispatchable", None)
    if not isinstance(dispatchable, (type, types.FunctionType)):
        pytest.skip("this networkx has no _dispatchable")

    @dispatchable
    @disk_cash.cache
    def size_scaled(G, k=3, *, scale=2.0):
        return G.number_of_nodes() * k * scale

    assert size_scaled(nx.path_graph(4)) == 24.0
