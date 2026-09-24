"""RANDOM-UNSEEDED says "cash caches the fit ... frozen replay" only for a fit
that is cached.

A bare ``clf.fit(X, y)`` mutates its receiver, so it is NOT CACHED and fits
afresh every run. On a re-run its receiver joins the statement's outputs (the
mutation is pre-routed), and the inline-fit check read that as a cached fitted
estimator: the second run warned that cash caches the fit as a frozen replay
and badged the row ``[random: unseeded]`` -- both false.
"""

from __future__ import annotations

import warnings

import pytest

from cash.tracking.randomness import CashRandomnessWarning
from tests._cell_driver import run_cash_cell

pytest.importorskip("sklearn")

SETUP = [
    "import numpy as np\nfrom sklearn.ensemble import RandomForestClassifier",
    "X = np.random.RandomState(0).rand(200, 4)\ny = (X[:, 0] > 0.5).astype(int)",
    "clf = RandomForestClassifier(n_estimators=5)",  # unseeded: no random_state
]


def _run_twice(magics, fit_cell: str) -> tuple[list[str], dict]:
    """Run the setup and *fit_cell* twice, as a notebook re-run does; return
    the randomness warnings and the fit's last metric."""
    captured: list[list[dict]] = []
    real_render = magics.badges.render

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    magics.badges.render = capture  # type: ignore[assignment]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                for cell in [*SETUP, fit_cell]:
                    run_cash_cell(magics, cell)
    finally:
        magics.badges.render = real_render  # type: ignore[assignment]
    messages = [str(w.message) for w in caught if issubclass(w.category, CashRandomnessWarning)]
    return messages, captured[-1][-1]


def test_a_bare_fit_rerun_does_not_claim_a_frozen_replay(cash_magics):
    messages, metric = _run_twice(cash_magics, "clf.fit(X, y)")
    assert any("In-place mutation on: clf" in r for r in metric.get("uncacheable_reasons") or []), metric
    assert not [m for m in messages if "frozen replay" in m or "clf.fit()" in m], messages
    assert not metric.get("random_unseeded"), "an uncached fit was badged as a frozen replay"


def test_a_cache_fit_still_warns(cash_magics):
    """Positive control: the opted-in fit is cached, so the warning is true."""
    messages, metric = _run_twice(cash_magics, "# @cash:cache-fit\nclf.fit(X, y)")
    assert any("frozen replay" in m for m in messages), messages
    assert metric.get("random_unseeded"), metric
