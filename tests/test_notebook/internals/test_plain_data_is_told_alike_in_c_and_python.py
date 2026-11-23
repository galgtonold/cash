"""Whether a container holds only plain data is answered alike in C and Python.

A statement reading a container through a module, or a library's function
returning one, has what it holds looked through for an object found by its
name; a container of numbers, strings and such containers is read at once
(``lookups_seen._pure_data``), in C where the extension is loaded. The C
reading must answer as the Python one does, or a statement would be
served, or refused, on one build and not the other.
"""

from __future__ import annotations

import logging
import random

import pytest

from cash import _native
from cash.tracking import lookups_seen

pure_data = getattr(_native.cwatch, "pure_data", None)
needs_c = pytest.mark.skipif(pure_data is None, reason=f"the C extension is not loaded: {_native.REASON}")


def _cases():
    cyclic: list = [1]
    cyclic.append(cyclic)
    rng = random.Random(7)

    def plain(depth):
        pick = rng.randrange(8 if depth else 5)
        if pick == 0:
            return rng.random()
        if pick == 1:
            return str(rng.randrange(100))
        if pick == 2:
            return None
        if pick == 3:
            return (rng.randrange(3), b"x", 1j, True)
        if pick == 4:
            return frozenset({rng.randrange(5), ("a", 1)})
        if pick == 5:
            return [plain(depth - 1) for _ in range(rng.randrange(4))]
        if pick == 6:
            return {str(i): plain(depth - 1) for i in range(rng.randrange(4))}
        return {(i, "k"): plain(depth - 1) for i in range(rng.randrange(3))}

    cases = [[], {}, (), set(), frozenset(), 5, "s", None, cyclic, [logging.getLogger("x")], {logging.root: 1}]
    cases += [[plain(4) for _ in range(5)] for _ in range(40)]
    cases += [[plain(3), object()], {"a": [1, {"b": bytearray(b"x")}]}, [[0] * 10] * 10, {1: {2}}]
    return cases


@needs_c
@pytest.mark.parametrize("cap", [lookups_seen.PURE_DATA_ITEMS, 0, 1, 7, 30])
def test_c_and_python_answer_alike(cap):
    for case in _cases():
        assert pure_data(case, cap) is lookups_seen._pure_data_py(case, cap), (case, cap)


def test_a_container_of_numbers_is_plain_and_one_holding_a_logger_is_not():
    assert lookups_seen._pure_data([{"id": i, "v": [i, (i, "a")]} for i in range(1_000)])
    assert not lookups_seen._pure_data([{"id": 1, "v": [logging.getLogger("x")]}])
    assert not lookups_seen._pure_data([0] * (lookups_seen.PURE_DATA_ITEMS + 1))
