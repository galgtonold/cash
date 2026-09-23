"""Tests for unseeded-randomness warnings and the ``# @cash:allow-random`` directive.

The detector, the warning helper, and the annotation field all existed
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

from cash.notebook.cache_status import CacheStatus
from cash.tracking.randomness import CashRandomnessWarning
from tests._cell_driver import run_cash_cell


def _last_metric(magics, code: str) -> dict:
    """Run one cell and return the last metric the processor recorded."""
    captured: list[list[dict]] = []
    real_render = magics.badges.render

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    magics.badges.render = capture  # type: ignore[assignment]
    try:
        run_cash_cell(magics, code)
    finally:
        magics.badges.render = real_render  # type: ignore[assignment]
    assert captured, "no metrics captured"
    return captured[-1][-1]


def _run_capturing_warnings(magics, code: str) -> list[warnings.WarningMessage]:
    """Execute *code* and return only the randomness warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(magics, code)
    return [w for w in caught if issubclass(w.category, CashRandomnessWarning)]


class TestWarningEmitted:
    """The positive case: the feature exists at all."""

    def test_unseeded_random_warns(self, cash_magics):
        caught = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\nx = np.random.rand(1000)",
        )
        assert len(caught) == 1
        msg = str(caught[0].message)
        assert "Unseeded randomness detected" in msg
        assert "numpy.random.rand" in msg

    def test_warning_names_the_suppression_directive(self, cash_magics):
        """The message must tell the user the escape hatch — it is the only
        place the directive is discoverable from the notebook itself."""
        caught = _run_capturing_warnings(
            cash_magics,
            "import random\nx = random.random()",
        )
        assert len(caught) == 1
        assert "@cash:allow-random" in str(caught[0].message)

    def test_stdlib_random_warns(self, cash_magics):
        caught = _run_capturing_warnings(
            cash_magics,
            "import random\nx = random.randint(0, 10)",
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

    def test_blanket_cash_warning_filter_suppresses_it(self, cash_magics):
        """The recipe in CashWarning's docstring must actually work.

        Regression guard: the helper used to force ``simplefilter('always')``
        inside a ``catch_warnings()`` block, which overrode user filters and
        made these warnings unsuppressable by the documented mechanism.
        """
        import cash

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=cash.CashWarning)
            run_cash_cell(cash_magics, "import numpy as np\nx = np.random.rand(1000)")
        assert [w for w in caught if issubclass(w.category, CashRandomnessWarning)] == []

    def test_attributed_to_the_cash_pseudo_file(self, cash_magics):
        """No user frame exists when the warning is raised (the statement hasn't
        run yet), so it is attributed to ``<cash>`` — the same pseudo-filename
        the statement is compiled under — rather than leaking an internal path."""
        caught = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\nx = np.random.rand(1000)",
        )
        assert len(caught) == 1
        assert caught[0].filename == "<cash>"


class TestSuppression:
    """``# @cash:allow-random`` silences the warning.

    This is the control the pre-existing probe never had: it asserted only on
    caching (which the directive does not affect) and so could not have caught
    the directive being inert.
    """

    def test_allow_random_suppresses_warning(self, cash_magics):
        caught = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\n# @cash:allow-random\nx = np.random.rand(1000)",
        )
        assert caught == []

    def test_suppression_is_scoped_to_the_annotated_statement(self, cash_magics):
        """The directive must not leak onto a neighbouring random statement."""
        caught = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\n# @cash:allow-random\nquiet = np.random.rand(1000)\n\nloud = np.random.randn(1000)\n",
        )
        assert len(caught) == 1
        assert "numpy.random.randn" in str(caught[0].message)


class TestSeededControl:
    """A seeded module is not a reproducibility hazard, so it must stay quiet."""

    def test_seeded_numpy_does_not_warn(self, cash_magics):
        caught = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\nnp.random.seed(42)\nx = np.random.rand(1000)",
        )
        assert caught == []

    def test_seeding_persists_across_cells(self, cash_magics):
        """Seeding is session state: a seed in one cell quiets a later cell."""
        run_cash_cell(cash_magics, "import numpy as np\nnp.random.seed(42)")
        caught = _run_capturing_warnings(cash_magics, "x = np.random.rand(1000)")
        assert caught == []

    def test_seeding_is_tracked_per_module(self, cash_magics):
        """Seeding numpy must NOT quiet a stdlib ``random`` draw — they are
        independent RNGs, and pretending otherwise would hide a real hazard."""
        run_cash_cell(cash_magics, "import numpy as np\nimport random\nnp.random.seed(42)")
        caught = _run_capturing_warnings(cash_magics, "x = random.random()")
        assert len(caught) == 1
        assert "random.random" in str(caught[0].message)

    def test_no_random_call_does_not_warn(self, cash_magics):
        caught = _run_capturing_warnings(cash_magics, "x = sum(range(100))")
        assert caught == []


class TestDedupe:
    """Policy: once per statement per session.

    Rationale: the warning describes the *source*, so it should fire when the
    user writes or edits that source and then stop. Warning per *run* would spam
    the notebook on every re-execution of a cached cell — which, in a library
    whose whole point is re-running cells cheaply, would train users to filter
    the warning out entirely.
    """

    def test_repeated_run_of_same_statement_warns_once(self, cash_magics):
        """Recompute path: every run executes, so the *same* message is deduped.

        ``# @cash:no-cache`` pins that path. Without it this test silently
        depended on the cost model: ``np.random.rand(1000)`` is microseconds on a
        fast machine, so it fell below the 0.01 s "too cheap to cache" floor and
        the re-run recomputed (silent). On a loaded Windows CI runner the same
        draw measured *over* the floor, so it was cached and the re-run
        **restored** — emitting the distinct "restored from cache" warning, which
        owns a separate dedupe slot. That timing dependency, not the dedupe
        logic, is what made this test flaky. The cached path is covered below.
        """
        code = "import numpy as np\n# @cash:no-cache\nx = np.random.rand(1000)"
        first = _run_capturing_warnings(cash_magics, code)
        second = _run_capturing_warnings(cash_magics, code)
        assert len(first) == 1
        assert second == [], "an unchanged re-run must not re-warn"

    def test_rerunning_a_cached_statement_does_not_spam(self, cash_magics):
        """Cached path — the case this class's rationale is really about.

        The restore warning makes a *different* claim ("the number you see is a
        replay"), so it earns its own dedupe slot and fires once. What must not
        happen is a warning on every re-execution of a cached cell: the third run
        is silent. ``# @cash:persist`` forces the value past the cost-model floor
        so the path is deterministic rather than timing-dependent.
        """
        code = "import numpy as np\n# @cash:persist\nx = np.random.rand(1000)"
        first = _run_capturing_warnings(cash_magics, code)
        second = _run_capturing_warnings(cash_magics, code)
        third = _run_capturing_warnings(cash_magics, code)
        assert len(first) == 1, "the compute warns once"
        assert len(second) == 1, "the first restore warns once, with its own text"
        assert "restored from cache" in str(second[0].message)
        assert third == [], "a third run must be silent — no per-run spam"

    def test_editing_the_statement_warns_again(self, cash_magics):
        """Dedupe keys on source, so edited code is a new warning."""
        first = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\nx = np.random.rand(1000)",
        )
        second = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\nx = np.random.rand(2000)",
        )
        assert len(first) == 1
        assert len(second) == 1

    def test_distinct_statements_each_warn(self, cash_magics):
        caught = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\na = np.random.rand(1000)\nb = np.random.randn(1000)\n",
        )
        assert len(caught) == 2

    def test_random_draw_in_a_loop_warns_once_not_per_iteration(self, cash_magics):
        """Loop bodies are dispatched per iteration with a per-iteration context
        comment prepended. Stripping that discriminator before the scan is what
        keeps a 50-iteration loop from emitting 50 identical warnings."""
        caught = _run_capturing_warnings(
            cash_magics,
            "import numpy as np\nacc = []\nfor i in range(50):\n    acc.append(np.random.rand())\n",
        )
        assert len(caught) == 1


class TestAnnotationDoesNotChangeCacheability:
    """``allow-random`` is advisory: it must never behave like ``no-cache``.

    Note the ``@cash:persist`` in these tests. Without it a fast statement sits
    below the cost model's 0.01 s floor and is never stored, so a "does it still
    cache?" assertion would measure the cost model rather than the directive —
    the exact confound that made an earlier probe meaningless.
    """

    def test_annotated_statement_still_restores_from_cache(self, cash_magics):
        code = "import numpy as np\n# @cash:persist\n# @cash:allow-random\nx = np.random.rand(1000)"
        run_cash_cell(cash_magics, code)
        m = _last_metric(cash_magics, code)
        assert m["status"] == CacheStatus.RESTORED

    def test_unannotated_random_statement_also_restores(self, cash_magics):
        """The baseline: randomness never blocked caching, annotation or not."""
        code = "import numpy as np\n# @cash:persist\nx = np.random.rand(1000)"
        run_cash_cell(cash_magics, code)
        m = _last_metric(cash_magics, code)
        assert m["status"] == CacheStatus.RESTORED

    def test_annotation_adds_no_uncacheable_reason(self, cash_magics):
        m = _last_metric(
            cash_magics,
            "import numpy as np\n# @cash:allow-random\nx = np.random.rand(1000)",
        )
        assert m["status"] == CacheStatus.COMPUTED
        assert m.get("uncacheable_reasons") == []

    def test_annotation_does_not_change_the_cache_key(self, cash_magics, mock_shell):
        """The directive is a comment about intent, not an input to the cache.

        Both spellings must land in the same cache slot, otherwise adding the
        annotation would silently orphan the entry the statement already had.
        """
        plain = _last_metric(cash_magics, "import numpy as np\nx = np.random.rand(1000)")
        mock_shell.user_ns.pop("x", None)
        annotated = _last_metric(
            cash_magics,
            "import numpy as np\n# @cash:allow-random\nx = np.random.rand(1000)",
        )
        assert annotated["cache_key"] == plain["cache_key"]


class TestStaleRandomnessAnnouncedOnRestore:
    """The replay warning.

    The first version warned on the COLD run — where the value is freshly computed and
    correct — and went silent on every restore after it, where the value is a
    frozen replay. The alarm was quiet exactly when it mattered.

    The mechanism was NOT that the warning path is skipped on a cache hit: it is
    reached (the call sits before the lookup on the common path). It is that the
    ``(code, message)`` dedupe had already been satisfied by the cold run. But
    simply dropping the dedupe would only have re-emitted "may not be
    reproducible" — the wrong claim for a restore, and the pre-lookup call site
    cannot make the right one, because the hit/miss outcome does not exist yet.
    Hence a second, distinct warning at a site that can observe the outcome.
    """

    def _restore_warnings(self, magics, code: str, runs: int = 2) -> list[list[str]]:
        """Run *code* *runs* times; return the randomness messages per run."""
        per_run = []
        for _ in range(runs):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_cash_cell(magics, code)
            per_run.append([str(w.message) for w in caught if issubclass(w.category, CashRandomnessWarning)])
        return per_run

    def test_restore_announces_the_replay(self, cash_magics, mock_shell):
        import numpy as np

        mock_shell.user_ns["np"] = np
        runs = self._restore_warnings(
            cash_magics,
            "# @cash:persist\nx = np.random.rand(200000)",
        )
        # Cold run: source-level advice.
        assert len(runs[0]) == 1
        assert "Unseeded randomness detected" in runs[0][0]
        # Restore: the value is a replay, and it says so.
        assert len(runs[1]) == 1
        message = runs[1][0]
        assert "Unseeded randomness restored from cache" in message
        assert "replay" in message
        assert "@cash:no-cache" in message

    def test_replay_warning_is_deduped_like_the_cold_one(self, cash_magics, mock_shell):
        """The anti-spam contract has to survive: the fact does not change
        between run 2 and run 20, so it is stated once per session."""
        import numpy as np

        mock_shell.user_ns["np"] = np
        runs = self._restore_warnings(
            cash_magics,
            "# @cash:persist\nx = np.random.rand(200000)",
            runs=4,
        )
        assert len(runs[0]) == 1  # cold: "detected"
        assert len(runs[1]) == 1  # first restore: "restored from cache"
        assert runs[2] == []  # thereafter: silence
        assert runs[3] == []

    def test_non_random_restore_is_silent(self, cash_magics):
        """Control: the noise floor. This is what stops the fix from becoming a
        'you hit the cache' banner over every restore in the notebook."""
        runs = self._restore_warnings(
            cash_magics,
            "# @cash:persist\ny = sum(i * i for i in range(200000))",
        )
        assert runs == [[], []]

    def test_seeded_restore_is_silent(self, cash_magics, mock_shell):
        """Control: a seeded draw replays honestly — the cached value is exactly
        what a recompute would produce, so there is nothing to report."""
        import numpy as np

        mock_shell.user_ns["np"] = np
        run_cash_cell(cash_magics, "np.random.seed(0)")
        runs = self._restore_warnings(
            cash_magics,
            "# @cash:persist\nx = np.random.rand(200000)",
        )
        assert runs == [[], []]

    def test_allow_random_suppresses_both_warnings(self, cash_magics, mock_shell):
        """The directive means 'I know'. A half-suppression would be worse than
        none: the user would think they had silenced it and still get noise."""
        import numpy as np

        mock_shell.user_ns["np"] = np
        runs = self._restore_warnings(
            cash_magics,
            "# @cash:persist\n# @cash:allow-random\nx = np.random.rand(200000)",
        )
        assert runs == [[], []]

    def test_replay_warning_does_not_change_cacheability(self, cash_magics, mock_shell):
        """The new warning is advisory too — it must not turn into a de-facto
        no-cache, which would silently undo the documented policy."""
        import numpy as np

        mock_shell.user_ns["np"] = np
        code = "# @cash:persist\nx = np.random.rand(200000)"
        first = _last_metric(cash_magics, code)
        second = _last_metric(cash_magics, code)
        assert first["status"] == CacheStatus.COMPUTED
        assert second["status"] == CacheStatus.RESTORED
