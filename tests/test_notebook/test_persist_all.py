"""`persist_all` mode caches every statement, bypassing the cost-aware floors.

Normally a statement that computes in under ~10 ms is not written to cache (the
storage/restore overhead would exceed the recompute cost), so a re-run
re-computes it. With `persist_all` on - via config (`Cash(persist_all=True)` /
config file), `cash.configure`, or the `%cash_persist on` magic - every
statement is cached, as if each carried `# @cash:persist`.
"""

from __future__ import annotations

import pytest

from cash import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics


@pytest.fixture
def persist_all_processor(mock_shell, clean_backend):
    """The statement processor of a Cash configured with ``persist_all=True``."""
    magics = CashMagics(mock_shell, Cash(backend=clean_backend, register_magic=False, persist_all=True))
    return magics._statement_processor


def test_trivial_statement_not_cached_by_default(statement_processor):
    p = statement_processor
    assert p.persist_all is False
    p.process_statement("x = 1 + 1")
    m2 = p.process_statement("x = 1 + 1")
    # Too cheap to cache -> recomputed, not restored.
    assert m2["status"] == CacheStatus.COMPUTED


def test_persist_all_config_caches_trivial_statement(persist_all_processor):
    p = persist_all_processor
    assert p.persist_all is True
    m1 = p.process_statement("y = 2 + 3")
    assert m1["status"] == CacheStatus.COMPUTED
    m2 = p.process_statement("y = 2 + 3")
    assert m2["status"] in (CacheStatus.RESTORED, CacheStatus.SKIPPED)


def test_cash_persist_magic_toggles_at_runtime(cash_magics, statement_processor):
    magics, p = cash_magics, statement_processor

    magics.cash_persist("on")
    assert magics._cash_instance.config.persist_all is True
    assert p.persist_all is True
    p.process_statement("z = 4 + 5")
    assert p.process_statement("z = 4 + 5")["status"] in (
        CacheStatus.RESTORED,
        CacheStatus.SKIPPED,
    )

    magics.cash_persist("off")
    assert magics._cash_instance.config.persist_all is False
    assert p.persist_all is False
    p.process_statement("w = 6 + 7")
    assert p.process_statement("w = 6 + 7")["status"] == CacheStatus.COMPUTED


def test_configure_reaches_a_running_pipeline(cash_magics, statement_processor):
    """``cash.configure(persist_all=True)`` is documented as a hot field. The
    processor copied the flag once, at construction, so flipping it on a
    running session changed config and nothing else."""
    p = statement_processor
    # Far above any scheduling stall, so "too cheap to cache" is certain.
    cash_magics._cash_instance.config.min_execution_time_to_cache_seconds = 3600.0
    assert p.persist_all is False
    cash_magics._cash_instance.reconfigure(persist_all=True)
    assert p.persist_all is True
    p.process_statement("v = 3 + 4")
    assert p.process_statement("v = 3 + 4")["status"] in (CacheStatus.RESTORED, CacheStatus.SKIPPED)

    cash_magics._cash_instance.reconfigure(persist_all=False)
    assert p.persist_all is False
    p.process_statement("u = 5 + 6")
    assert p.process_statement("u = 5 + 6")["status"] == CacheStatus.COMPUTED


def test_cash_persist_writes_config(cash_magics):
    """``%cash_persist`` kept its own copy; config (what ``cash.configure``
    and everything else reads) never heard of it."""
    cash_magics.cash_persist("on")
    assert cash_magics._cash_instance.config.persist_all is True
    cash_magics.cash_persist("")  # toggle
    assert cash_magics._cash_instance.config.persist_all is False


def test_explicit_no_cache_still_wins_over_persist_all(persist_all_processor):
    """A statement annotated @cash:no-cache must not be cached even in
    persist_all mode (skip_cache takes precedence)."""
    p = persist_all_processor
    from cash.analysis.annotations import CacheAnnotation

    no_cache = CacheAnnotation(no_cache=True)
    p.process_statement("q = 8 + 9", annotation=no_cache)
    m2 = p.process_statement("q = 8 + 9", annotation=no_cache)
    assert m2["status"] == CacheStatus.COMPUTED
