"""``# @cash:cache-calls`` must say so when it does nothing (CAS-243).

The directive is opt-in and its eligibility rule is not obvious — a call that
reads the statement's own target is declined, because it *is* the fold and no
order-independent value can be pulled out of it. So the failure mode is silence:
the user writes the directive on `s = merge(s, x)`, nothing is eligible, nothing
is cached, and there is no signal distinguishing that from a cache that simply
missed.

`CashCacheIneffectiveWarning` is the right family — this is literally a cache
the user asked for and did not get.

Unit tests because the warning is a ``warnings.warn`` on the statement path,
where ``pytest.warns`` can assert on it precisely; under ``nb_runner`` it lands
in a subprocess kernel's stderr and can only be string-matched.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.exceptions import CashCacheIneffectiveWarning
from cash.notebook.ipython.magics import CashMagics


class _MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = _MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend, cash
    backend.clear()
    shell.user_ns.clear()


_HELPERS = "def compute(x):\n    return x + 1\ndef merge(a, b):\n    return a + b\n"


def test_warns_when_no_call_is_eligible(magics_fixture):
    """`s = merge(s, x)` — the call reads the target, so nothing is extractable."""
    magics, shell, _, _ = magics_fixture
    magics.cash("", _HELPERS + "s = 0\nx = 1")

    with pytest.warns(CashCacheIneffectiveWarning, match="cache-calls"):
        magics.cash("", "# @cash:cache-calls\ns = merge(s, x)")


def test_does_not_warn_when_a_call_is_eligible(magics_fixture):
    """The positive control. Without it, a warning that always fires passes."""
    magics, shell, _, _ = magics_fixture
    magics.cash("", _HELPERS + "s = 0\nx = 1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        magics.cash("", "# @cash:cache-calls\ns += compute(x)")

    offenders = [w for w in caught
                 if issubclass(w.category, CashCacheIneffectiveWarning)
                 and "cache-calls" in str(w.message)]
    assert not offenders, f"warned despite an eligible call: {[str(w.message) for w in offenders]}"


def test_does_not_warn_without_the_directive(magics_fixture):
    """No directive, no opinion — an ordinary statement must stay silent."""
    magics, shell, _, _ = magics_fixture
    magics.cash("", _HELPERS + "s = 0\nx = 1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        magics.cash("", "s = merge(s, x)")

    offenders = [w for w in caught if "cache-calls" in str(w.message)]
    assert not offenders, f"warned with no directive present: {offenders}"


def test_warns_once_per_statement_not_once_per_iteration(magics_fixture):
    """A 100-iteration loop must warn once, not a hundred times.

    The processor already follows this pattern for the persist-amplification
    and entropy-reseed notes; a per-iteration warning would be unreadable and
    would train users to filter the whole category out.
    """
    magics, shell, _, _ = magics_fixture
    magics.cash("", _HELPERS + "s = 0")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        magics.cash("", "# @cash:cache-calls\nfor x in range(5):\n    s = merge(s, x)")

    hits = [w for w in caught if "cache-calls" in str(w.message)]
    assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"
