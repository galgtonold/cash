"""A set however deep in an argument, and a container subclass's own state,
key the same in every process and as part of one walk.

The canonical form sorts every set it meets, so a set's iteration order, which
PYTHONHASHSEED picks, cannot reach the key. A part cut off at a fixed depth and
handed to pickle as it is keeps that order, and the key changes from process to
process: a persistent entry never hits again. A container subclass's own
attributes are walked with the container, so a list two of them share is
marked as shared and a loop back to the container is caught. A value nested
deeper than the walk can follow, a long linked list say, runs the call
uncached as an unhashable argument: it is never keyed on part of itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from cash import CashCacheIneffectiveWarning

PROGRAM = textwrap.dedent("""
    import hashlib
    from cash.object_hashing import canonical_bytes

    class Tagged(dict):
        pass

    S = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}

    def wrapped(value, times):
        for _ in range(times):
            value = [value]
        return value

    in_state = Tagged(a=1)
    in_state.meta = wrapped(S, 6)
    values = {"in a subclass's state": in_state, "sixty lists down": wrapped(S, 60)}
    for name, value in values.items():
        print(name, "=", hashlib.sha256(canonical_bytes(value)).hexdigest())
""")


def _keys(seed: str) -> dict[str, str]:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    done = subprocess.run([sys.executable, "-c", PROGRAM], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    return dict(line.split(" = ") for line in done.stdout.strip().splitlines())


@pytest.mark.timeout(300)
def test_a_deep_set_keys_alike_under_every_hash_seed():
    runs = [_keys(seed) for seed in ("1", "2", "3")]
    assert len(runs[0]) == 2
    assert runs[0] == runs[1] == runs[2]


class Tagged(dict):
    pass


def test_a_list_two_attributes_share_keys_apart_from_two_equal_lists(cash_instance):
    @cash_instance.cache
    def one_list(d):
        return d.a is d.b

    shared = Tagged()
    shared.a = shared.b = []
    separate = Tagged()
    separate.a, separate.b = [], []
    # Keyed alike, the second call is served the first one's True.
    assert one_list(shared) is True
    assert one_list(separate) is False


def test_a_subclass_whose_attribute_is_itself_runs_uncached(cash_instance):
    runs = []

    @cash_instance.cache
    def size(d):
        runs.append(1)
        return len(d)

    looped = Tagged(x=1)
    looped.me = looped
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert size(looped) == 1
        assert size(looped) == 1
    assert len(runs) == 2
    codes = [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert "KEY-UNHASHABLE-ARG" in codes


class Node:
    def __init__(self, nxt):
        self.nxt = nxt


def _chain(n):
    head = None
    for _ in range(n):
        head = Node(head)
    return head


def _nested(n):
    value = []
    for _ in range(n):
        value = [value]
    return value


@pytest.mark.parametrize("make", [lambda: _chain(3000), lambda: _nested(5000)], ids=["linked list", "nested list"])
def test_a_value_deeper_than_the_walk_runs_uncached_as_unhashable(cash_instance, make):
    value = make()
    runs = []

    @cash_instance.cache
    def f(x):
        runs.append(1)
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(value) == 1
        assert f(value) == 1
    assert len(runs) == 2
    codes = [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert "KEY-UNHASHABLE-ARG" in codes
    assert "KEY-BUILD-FAILED" not in codes
