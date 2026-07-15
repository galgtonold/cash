"""Tests for unseeded-randomness warnings and the ``# @cash:allow-random`` directive.

CAS-114. The detector, the warning helper, and the annotation field all existed
long before the feature did: ``allow_random`` was parsed into ``CacheAnnotation``
and never read, and ``check_and_warn_randomness`` had zero callers, so an
unseeded ``np.random.rand(1000)`` emitted no warning at all. This module pins the
wiring that made ~10 doc pages true.

The contract, in one line: **allow-random is advisory.** It suppresses a warning
and changes nothing about caching. Unseeded random statements are cacheable by
design (``decide_cacheability`` has no randomness reason-source), with or without
the directive — ``TestAnnotationDoesNotChangeCacheability`` is what stops that
from silently regressing into a de-facto ``no-cache`` alias.

These are unit tests because the warning is a Python ``warnings.warn`` raised on
the statement-processing path: ``pytest.warns`` can assert on it precisely here,
whereas under ``nb_runner`` it lands in a subprocess kernel's cell stderr and can
only be string-matched. The real-kernel end of the contract lives in
``tests/test_notebook_integration/test_allow_random_annotation_integration.py``.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.randomness import CashRandomnessWarning


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


def _last_metric(magics, code: str) -> dict:
    """Run one cell and return the last metric the processor recorded."""
    captured: list[list[dict]] = []
    real_render = magics._render_interactive_badge

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    magics._render_interactive_badge = capture  # type: ignore[assignment]
    try:
        magics.cash("", code)
    finally:
        magics._render_interactive_badge = real_render  # type: ignore[assignment]
    assert captured, "no metrics captured"
    return captured[-1][-1]


def _run_capturing_warnings(magics, code: str) -> list[warnings.WarningMessage]:
    """Execute *code* and return only the randomness warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        magics.cash("", code)
    return [w for w in caught if issubclass(w.category, CashRandomnessWarning)]


class TestWarningEmitted:
    """The positive case: the feature exists at all."""

    def test_unseeded_random_warns(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics, "import numpy as np\nx = np.random.rand(1000)",
        )
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "Unseeded randomness detected" in msg
        assert "numpy.random.rand" in msg

    def test_warning_names_the_suppression_directive(self, magics_fixture):
        """The message must tell the user the escape hatch — it is the only
        place the directive is discoverable from the notebook itself."""
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics, "import random\nx = random.random()",
        )
        assert len(caught) == 1
        assert "@cash:allow-random" in str(caught[0].message)

    def test_stdlib_random_warns(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics, "import random\nx = random.randint(0, 10)",
        )
        assert len(caught) == 1
        assert "random.randint" in str(caught[0].message)


class TestWarningClassContract:
    """``CashRandomnessWarning`` must behave like the rest of the warning family.

    It was declared ``UserWarning`` and left out of the top-level exports, which
    was invisible while nothing raised it. Now that it fires, a user needs to be
    able to reach it and filter it the way ``CashWarning`` documents.
    """

    def test_exported_from_the_package_root(self):
        import cash

        assert cash.CashRandomnessWarning is CashRandomnessWarning
        assert "CashRandomnessWarning" in cash.__all__

    def test_is_in_the_cash_warning_family(self):
        import cash

        assert issubclass(CashRandomnessWarning, cash.CashWarning)

    def test_remains_a_user_warning(self):
        """CashWarning derives from UserWarning, so pre-existing filters that
        target UserWarning must keep catching this."""
        assert issubclass(CashRandomnessWarning, UserWarning)

    def test_blanket_cash_warning_filter_suppresses_it(self, magics_fixture):
        """The recipe in CashWarning's docstring must actually work.

        Regression guard: the helper used to force ``simplefilter('always')``
        inside a ``catch_warnings()`` block, which overrode user filters and
        made these warnings unsuppressable by the documented mechanism.
        """
        import cash

        magics, _shell, _backend, _cash = magics_fixture
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=cash.CashWarning)
            magics.cash("", "import numpy as np\nx = np.random.rand(1000)")
        assert [w for w in caught if issubclass(w.category, CashRandomnessWarning)] == []

    def test_attributed_to_the_cash_pseudo_file(self, magics_fixture):
        """No user frame exists when the warning is raised (the statement hasn't
        run yet), so it is attributed to ``<cash>`` — the same pseudo-filename
        the statement is compiled under — rather than leaking an internal path."""
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics, "import numpy as np\nx = np.random.rand(1000)",
        )
        assert len(caught) == 1
        assert caught[0].filename == "<cash>"


class TestSuppression:
    """``# @cash:allow-random`` silences the warning.

    This is the control the pre-existing probe never had: it asserted only on
    caching (which the directive does not affect) and so could not have caught
    the directive being inert.
    """

    def test_allow_random_suppresses_warning(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics,
            "import numpy as np\n# @cash:allow-random\nx = np.random.rand(1000)",
        )
        assert caught == []

    def test_allowrandom_alias_suppresses_warning(self, magics_fixture):
        """The no-hyphen alias parses (annotations.py) and must suppress too."""
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics,
            "import numpy as np\n# @cash:allowrandom\nx = np.random.rand(1000)",
        )
        assert caught == []

    def test_suppression_is_scoped_to_the_annotated_statement(self, magics_fixture):
        """The directive must not leak onto a neighbouring random statement."""
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics,
            "import numpy as np\n"
            "# @cash:allow-random\n"
            "quiet = np.random.rand(1000)\n"
            "\n"
            "loud = np.random.randn(1000)\n",
        )
        assert len(caught) == 1
        assert "numpy.random.randn" in str(caught[0].message)


class TestSeededControl:
    """A seeded module is not a reproducibility hazard, so it must stay quiet."""

    def test_seeded_numpy_does_not_warn(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics,
            "import numpy as np\nnp.random.seed(42)\nx = np.random.rand(1000)",
        )
        assert caught == []

    def test_seeding_persists_across_cells(self, magics_fixture):
        """Seeding is session state: a seed in one cell quiets a later cell."""
        magics, _shell, _backend, _cash = magics_fixture
        magics.cash("", "import numpy as np\nnp.random.seed(42)")
        caught = _run_capturing_warnings(magics, "x = np.random.rand(1000)")
        assert caught == []

    def test_seeding_is_tracked_per_module(self, magics_fixture):
        """Seeding numpy must NOT quiet a stdlib ``random`` draw — they are
        independent RNGs, and pretending otherwise would hide a real hazard."""
        magics, _shell, _backend, _cash = magics_fixture
        magics.cash("", "import numpy as np\nimport random\nnp.random.seed(42)")
        caught = _run_capturing_warnings(magics, "x = random.random()")
        assert len(caught) == 1
        assert "random.random" in str(caught[0].message)

    def test_no_random_call_does_not_warn(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(magics, "x = sum(range(100))")
        assert caught == []


class TestDedupe:
    """Policy: once per statement per session.

    Rationale: the warning describes the *source*, so it should fire when the
    user writes or edits that source and then stop. Warning per *run* would spam
    the notebook on every re-execution of a cached cell — which, in a library
    whose whole point is re-running cells cheaply, would train users to filter
    the warning out entirely.
    """

    def test_repeated_run_of_same_statement_warns_once(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        code = "import numpy as np\nx = np.random.rand(1000)"
        first = _run_capturing_warnings(magics, code)
        second = _run_capturing_warnings(magics, code)
        assert len(first) == 1
        assert second == [], "an unchanged re-run must not re-warn"

    def test_editing_the_statement_warns_again(self, magics_fixture):
        """Dedupe keys on source, so edited code is a new warning."""
        magics, _shell, _backend, _cash = magics_fixture
        first = _run_capturing_warnings(
            magics, "import numpy as np\nx = np.random.rand(1000)",
        )
        second = _run_capturing_warnings(
            magics, "import numpy as np\nx = np.random.rand(2000)",
        )
        assert len(first) == 1
        assert len(second) == 1

    def test_distinct_statements_each_warn(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics,
            "import numpy as np\n"
            "a = np.random.rand(1000)\n"
            "b = np.random.randn(1000)\n",
        )
        assert len(caught) == 2

    def test_random_draw_in_a_loop_warns_once_not_per_iteration(self, magics_fixture):
        """Loop bodies are dispatched per iteration with a per-iteration context
        comment prepended. Stripping that discriminator before the scan is what
        keeps a 50-iteration loop from emitting 50 identical warnings."""
        magics, _shell, _backend, _cash = magics_fixture
        caught = _run_capturing_warnings(
            magics,
            "import numpy as np\n"
            "acc = []\n"
            "for i in range(50):\n"
            "    acc.append(np.random.rand())\n",
        )
        assert len(caught) == 1


class TestAnnotationDoesNotChangeCacheability:
    """``allow-random`` is advisory: it must never behave like ``no-cache``.

    Note the ``@cash:persist`` in these tests. Without it a fast statement sits
    below the cost model's 0.01 s floor and is never stored, so a "does it still
    cache?" assertion would measure the cost model rather than the directive —
    the exact confound that made the pre-existing wave5 probe meaningless.
    """

    def test_annotated_statement_still_restores_from_cache(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        code = (
            "import numpy as np\n"
            "# @cash:persist\n"
            "# @cash:allow-random\n"
            "x = np.random.rand(1000)"
        )
        magics.cash("", code)
        m = _last_metric(magics, code)
        assert m["status"] == CacheStatus.RESTORED

    def test_unannotated_random_statement_also_restores(self, magics_fixture):
        """The baseline: randomness never blocked caching, annotation or not."""
        magics, _shell, _backend, _cash = magics_fixture
        code = "import numpy as np\n# @cash:persist\nx = np.random.rand(1000)"
        magics.cash("", code)
        m = _last_metric(magics, code)
        assert m["status"] == CacheStatus.RESTORED

    def test_annotation_adds_no_uncacheable_reason(self, magics_fixture):
        magics, _shell, _backend, _cash = magics_fixture
        m = _last_metric(
            magics,
            "import numpy as np\n# @cash:allow-random\nx = np.random.rand(1000)",
        )
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("uncacheable_reasons") == []

    def test_annotation_does_not_change_the_cache_key(self, magics_fixture):
        """The directive is a comment about intent, not an input to the cache.

        Both spellings must land in the same cache slot, otherwise adding the
        annotation would silently orphan the entry the statement already had.
        """
        magics, shell, _backend, _cash = magics_fixture
        plain = _last_metric(magics, "import numpy as np\nx = np.random.rand(1000)")
        shell.user_ns.pop("x", None)
        annotated = _last_metric(
            magics,
            "import numpy as np\n# @cash:allow-random\nx = np.random.rand(1000)",
        )
        assert annotated["cache_key"] == plain["cache_key"]
