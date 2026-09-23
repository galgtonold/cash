"""``X = vec.fit_transform(texts)`` can be cached -- by asking for it.

TF-IDF was never cached ("NOT CACHED - In-place mutation on:
vec") and cost 11-17 s on every pass. ``# @cash:cache-fit`` exists for exactly
this trade, but covered only a bare ``est.fit(...)`` / ``partial_fit``: the
assignment form of ``fit_transform`` was refused even when asked. It stays
opt-in (see the directive's identity caveat), and the refusal now says how
to opt in.
"""

import pytest

pytest.importorskip("sklearn")

from cash.notebook.cache_status import CacheStatus  # noqa: E402

SETUP = (
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "texts = [f'the cat {i % 97} sat by dog {i % 89}' for i in range(40_000)]\n"
)


def _run(cash_magics, code):
    """Through the processor with the annotation parsed as the cell executor
    parses it: from the source, per statement."""
    import ast

    from cash.analysis.annotations import get_statement_annotations

    node = ast.parse(code).body[0]
    return cash_magics._statement_processor.process_statement(
        ast.unparse(node), annotation=get_statement_annotations(code, node)
    )


def test_asked_for_it_is_cached_with_the_fitted_vectorizer(cash_magics):
    ns = cash_magics.shell.user_ns
    for line in SETUP.strip().splitlines() + ["vec = TfidfVectorizer()"]:
        _run(cash_magics, line)
    code = "# @cash:cache-fit\nX = vec.fit_transform(texts)"
    first = _run(cash_magics, code)
    assert first["status"] == CacheStatus.COMPUTED
    assert not first.get("uncacheable_reasons"), first.get("uncacheable_reasons")

    # as after a restart: the unfitted vectorizer is built again, X is gone
    _run(cash_magics, "vec = TfidfVectorizer()")
    del ns["X"]
    assert not hasattr(ns["vec"], "vocabulary_")
    again = _run(cash_magics, code)
    assert again["status"] == CacheStatus.RESTORED, again
    assert ns["X"].shape[0] == 40_000
    assert hasattr(ns["vec"], "vocabulary_"), "the restore left the vectorizer unfitted"


def test_not_asked_says_how(cash_magics):
    for line in SETUP.strip().splitlines() + ["vec = TfidfVectorizer()"]:
        _run(cash_magics, line)
    m = _run(cash_magics, "X = vec.fit_transform(texts)")
    reasons = " ".join(m.get("uncacheable_reasons") or [])
    assert "In-place mutation on: vec" in reasons, reasons
    assert "@cash:cache-fit" in reasons, reasons
