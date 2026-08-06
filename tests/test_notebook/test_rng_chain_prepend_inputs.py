"""A prepended RNG statement must bring its own ordinary inputs with it.

``UpstreamChecker._prepend_rng_chain_for_reexecuted_draws`` re-runs the RNG
chain ahead of a re-executed draw so that draw lands at the stream position it
holds top-to-bottom (ADR-017 / CAS-223). It selects statements by whether they
touch an RNG module and has no notion of what those statements *read*, so a
draw sharing a cell with an ordinary assignment was scheduled without it::

    np.random.seed(42)
    n = 500
    base = np.random.randn(n)     # prepended
                                  # ``n = 500`` was not -> NameError

Reconstruction then died with ``UpstreamStateError: ... name 'n' is not
defined``. It failed loudly rather than serving a wrong value, but the plan it
built was malformed.

These tests drive the planner directly: it is a pure function of the notebook
cells, the already-planned statements, and the live namespace, so the failure
reproduces deterministically without a kernel.
"""

from unittest.mock import MagicMock

from cash.notebook.upstream import UpstreamChecker


def _checker(live: dict | None = None) -> UpstreamChecker:
    """An UpstreamChecker whose shell exposes ``live`` as the live namespace."""
    shell = MagicMock()
    shell.user_ns = dict(live or {})
    return UpstreamChecker(shell=shell)


# Cell 3 draws with a *scaled* result: editing MULT re-executes that draw
# without touching the seed, which is what arms the RNG-chain prepend.
NOTEBOOK = [
    "import numpy as np",
    "np.random.seed(42)",
    "n = 500\nbase = np.random.randn(n)",
    "arr = np.random.randn(5) * MULT",
]
REEXECUTED_DRAW = "arr = np.random.randn(5) * MULT"
CHAIN_DRAW = "base = np.random.randn(n)"


class TestPrependedStatementInputs:
    def test_prepended_draw_brings_its_own_input(self):
        """``n = 500`` must be scheduled ahead of the draw that reads ``n``."""
        checker = _checker(live={"np": object(), "MULT": 3})

        out = checker._prepend_rng_chain_for_reexecuted_draws(
            NOTEBOOK, [REEXECUTED_DRAW], 4,
        )

        assert CHAIN_DRAW in out, "the RNG chain must still be rebuilt"
        assert "n = 500" in out, (
            "the prepended draw reads 'n'; without its definition the "
            "reconstruction raises NameError"
        )
        assert out.index("n = 500") < out.index(CHAIN_DRAW)

    def test_live_names_are_not_re_derived(self):
        """A name already bound in the kernel needs no statement of its own.

        Re-deriving it would re-run work the live namespace already holds --
        cheap for ``n = 500``, not cheap for ``n = load_config()``.
        """
        checker = _checker(live={"np": object(), "MULT": 3, "n": 500})

        out = checker._prepend_rng_chain_for_reexecuted_draws(
            NOTEBOOK, [REEXECUTED_DRAW], 4,
        )

        assert CHAIN_DRAW in out
        assert "n = 500" not in out
        assert "import numpy as np" not in out

    def test_input_definitions_resolve_transitively(self):
        """A pulled-in definition pulls in what *it* reads, in source order."""
        notebook = [
            "import numpy as np",
            "np.random.seed(0)",
            "rows = 10",
            "n = rows * 50",
            "base = np.random.randn(n)",
            "arr = np.random.randn(5) * MULT",
        ]
        checker = _checker(live={"np": object(), "MULT": 3})

        out = checker._prepend_rng_chain_for_reexecuted_draws(
            notebook, ["arr = np.random.randn(5) * MULT"], 6,
        )

        assert out.index("rows = 10") < out.index("n = rows * 50")
        assert out.index("n = rows * 50") < out.index("base = np.random.randn(n)")

    def test_no_draw_in_plan_is_still_a_no_op(self):
        """The guard rails stay put: nothing re-executed means nothing prepended."""
        checker = _checker(live={"np": object()})

        statements = ["total = 1 + 2"]
        out = checker._prepend_rng_chain_for_reexecuted_draws(
            NOTEBOOK, statements, 4,
        )

        assert out == statements
