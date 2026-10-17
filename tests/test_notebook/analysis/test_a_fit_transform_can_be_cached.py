"""``X = vec.fit_transform(texts)`` is cached with the fitted vectorizer.

TF-IDF that is not cached costs 11-17 s on every pass. The statement binds the
matrix and fits ``vec`` in place; both are stored, so a hit leaves ``vec``
fitted too.
"""

import pytest

pytest.importorskip("sklearn")

from cash.notebook.cache_status import CacheStatus

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
    code = "X = vec.fit_transform(texts)"
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


def test_cached_without_the_directive(cash_magics):
    ns = cash_magics.shell.user_ns
    for line in SETUP.strip().splitlines() + ["vec = TfidfVectorizer()"]:
        _run(cash_magics, line)
    first = _run(cash_magics, "X = vec.fit_transform(texts)")
    assert not first.get("uncacheable_reasons"), first.get("uncacheable_reasons")

    _run(cash_magics, "vec = TfidfVectorizer()")
    del ns["X"]
    again = _run(cash_magics, "X = vec.fit_transform(texts)")
    assert again["status"] == CacheStatus.RESTORED, again
    assert hasattr(ns["vec"], "vocabulary_"), "the restore left the vectorizer unfitted"
