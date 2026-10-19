"""The order a statement left its sparse inputs in, put back on a hit.

scipy puts a CSR, CSC, BSR or COO matrix in canonical form IN PLACE when it
is merely read: ``NMF().fit(X)``, ``X > 0`` and friends sort its indices
and sum its duplicates. The values stay the same, so cash does not count
that as changing the input (`mutation_fingerprint._canonical_sparse`), and
a served ``nmf = fit_nmf(X)`` left ``X`` unsorted where a run sorts it:
``X.indices`` and ``X.data`` came in another order than without cash. The
run notes which of its inputs it put in order; a hit does the same with
scipy's own ``sum_duplicates``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["put_in_order", "unordered_sparse", "were_put_in_order"]


def _unordered(value: Any) -> bool:
    """Is *value* a scipy sparse matrix or array not in canonical form?"""
    if "scipy.sparse" not in sys.modules or not type(value).__module__.startswith("scipy.sparse"):
        return False
    try:
        return hasattr(value, "sum_duplicates") and not value.has_canonical_format
    except Exception:  # noqa: BLE001 - a format that cannot say: left as it is
        return False


def unordered_sparse(names: Iterable[str], namespace: Mapping[str, Any]) -> dict[str, int]:
    """``{name: id(value)}`` for each of *names* bound to a sparse matrix
    not in canonical form: what a run may put in order."""
    return {name: id(value) for name in names if _unordered(value := namespace.get(name))}


def were_put_in_order(before: Mapping[str, int], namespace: Mapping[str, Any]) -> list[str]:
    """Which of *before* (`unordered_sparse`) are bound to the same matrix
    still, now in canonical form."""
    found = []
    for name, ident in before.items():
        value = namespace.get(name)
        if id(value) == ident and not _unordered(value):
            found.append(name)
    return sorted(found)


def put_in_order(names: Iterable[str] | None, namespace: Mapping[str, Any]) -> None:
    """Put each of *names* that a run put in order in canonical form, as
    the run did: scipy's ``sum_duplicates``."""
    for name in names or ():
        value = namespace.get(name)
        if _unordered(value):
            try:
                value.sum_duplicates()
            except Exception:  # noqa: BLE001 - left as the restore found it
                pass
