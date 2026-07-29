"""The production call site must apply the object-level gate too (CAS-243).

``wrap_eligible_calls``'s ``gate`` parameter (Task 4) was tested at the AST
level but never wired into ``statement/processor.py``'s
``_code_and_tree_for_execution`` (Task 4 landed the function, not the wiring —
confirmed by ``grep -rn "gate=" src/`` returning zero hits before this change).
Structural eligibility alone (the free-variable rule in
``eligible_call_nodes``) says nothing about whether a call is *substantively*
cacheable -- ``time.time()`` doesn't read the statement's own target, so it is
structurally eligible, but caching it would be exactly the kind of thing
``decide_cacheability``'s forbidden-function scan exists to refuse for a plain
statement. Without the gate wired in, an intercepted call skipped that
judgment entirely: the call still got wrapped into
``__cash_call__(time.time, 0)()`` (a harmless no-op here, since ``time.time``
is a builtin and ``CallCache.resolve``'s own type check passes it straight
through) -- but the processor believed it had found something cacheable and
stayed silent, instead of telling the user their directive did nothing.

Not exercised via ``nb_runner`` (no real kernel needed): this is a warning on
the synchronous statement path, and ``CashMagics`` + a mock shell reaches
``_code_and_tree_for_execution`` directly, matching the pattern
``test_cache_calls_noop_warning.py`` already uses for the AST-only case.
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


def test_a_forbidden_call_warns_even_though_it_is_structurally_eligible(magics_fixture):
    """``time.time()`` reads nothing of ``out`` -- structurally eligible -- but
    ``decide_cacheability`` forbids it for an ordinary statement, and the gate
    must apply that same refusal here: no site survives, so this is
    indistinguishable, from the user's side, from "nothing was eligible."
    """
    magics, shell, _, _ = magics_fixture
    magics.cash("", "import time\nout = []")

    with pytest.warns(CashCacheIneffectiveWarning, match="cache-calls"):
        magics.cash("", "# @cash:cache-calls\nout.append(time.time())")


def test_an_ordinary_eligible_call_does_not_warn(magics_fixture):
    """Positive control for the test above, using the SAME statement shape
    (``out.append(...)``) so only the callee's cacheability differs.
    """
    magics, shell, _, _ = magics_fixture
    magics.cash("", "def compute(x):\n    return x + 1\nout = []\nx = 1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        magics.cash("", "# @cash:cache-calls\nout.append(compute(x))")

    offenders = [w for w in caught
                 if issubclass(w.category, CashCacheIneffectiveWarning)
                 and "cache-calls" in str(w.message)]
    assert not offenders, f"warned despite a genuinely cacheable call: {[str(w.message) for w in offenders]}"
