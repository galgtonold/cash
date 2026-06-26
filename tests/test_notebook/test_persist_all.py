"""`persist_all` mode caches every statement, bypassing the cost-aware floors.

Normally a statement that computes in under ~10 ms is not written to cache (the
storage/restore overhead would exceed the recompute cost), so a re-run
re-computes it. With `persist_all` on - via config (`Cash(persist_all=True)` /
config file), `cash.configure`, or the `%cash_persist on` magic - every
statement is cached, as if each carried `# @cash:persist`.
"""
from __future__ import annotations

from cash import Cash
from cash.backends import InMemoryBackend
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics

from tests.conftest import MockShell


def _make(persist_all: bool):
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False, persist_all=persist_all)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    return magics, shell, magics._statement_processor


def test_trivial_statement_not_cached_by_default():
    _magics, _shell, p = _make(persist_all=False)
    assert p.persist_all is False
    p.process_statement("x = 1 + 1")
    m2 = p.process_statement("x = 1 + 1")
    # Too cheap to cache -> recomputed, not restored.
    assert m2['status'] == CacheStatus.COMPUTED


def test_persist_all_config_caches_trivial_statement():
    _magics, _shell, p = _make(persist_all=True)
    assert p.persist_all is True
    m1 = p.process_statement("y = 2 + 3")
    assert m1['status'] == CacheStatus.COMPUTED
    m2 = p.process_statement("y = 2 + 3")
    assert m2['status'] in (CacheStatus.RESTORED, CacheStatus.SKIPPED)


def test_cash_persist_magic_toggles_at_runtime():
    magics, _shell, p = _make(persist_all=False)

    magics.cash_persist("on")
    assert magics._persist_all is True
    assert p.persist_all is True
    p.process_statement("z = 4 + 5")
    assert p.process_statement("z = 4 + 5")['status'] in (
        CacheStatus.RESTORED, CacheStatus.SKIPPED,
    )

    magics.cash_persist("off")
    assert magics._persist_all is False
    assert p.persist_all is False
    p.process_statement("w = 6 + 7")
    assert p.process_statement("w = 6 + 7")['status'] == CacheStatus.COMPUTED


def test_explicit_no_cache_still_wins_over_persist_all():
    """A statement annotated @cash:no-cache must not be cached even in
    persist_all mode (skip_cache takes precedence)."""
    _magics, _shell, p = _make(persist_all=True)
    from cash.notebook.annotations import CacheAnnotation
    no_cache = CacheAnnotation(no_cache=True)
    p.process_statement("q = 8 + 9", annotation=no_cache)
    m2 = p.process_statement("q = 8 + 9", annotation=no_cache)
    assert m2['status'] == CacheStatus.COMPUTED
