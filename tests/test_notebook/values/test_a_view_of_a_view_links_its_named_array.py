"""A view of a view still finds the named array under it.

``big = np.arange(6.0).reshape(3, 2)`` is a view itself, so ``big[0].base`` is
the buffer ``arange`` made, not ``big``: rows a loop kept (``parts[i] =
big[i]``) were not linked to ``big``, and a change through them left ``big``'s
lineage, and the cells reading it, as they were.
"""

from __future__ import annotations

import numpy as np

from cash.notebook.shared_objects import view_base_names
from cash.notebook.statement.derivation_edges import detect_derivation_edges


def test_a_view_of_a_view_finds_the_named_array_under_it():
    big = np.arange(6.0).reshape(3, 2)
    other = np.arange(4.0).reshape(2, 2)
    parts = {i: big[i] for i in range(3)}
    ns = {"big": big, "parts": parts, "other": other}
    assert view_base_names([parts], ns) == {"big"}

    edges: dict[str, set[str]] = {}
    ns["row"] = big[0]
    detect_derivation_edges(edges, "row", ns["row"], ns)
    assert edges == {"row": {"big"}}


def test_two_views_of_one_buffer_that_do_not_overlap_are_not_linked():
    buf = np.arange(10.0).reshape(10)
    low, high = buf[:5], buf[5:]
    ns = {"low": low, "high": high}
    edges: dict[str, set[str]] = {}
    detect_derivation_edges(edges, "low", low, ns)
    assert "high" not in edges.get("low", set())
