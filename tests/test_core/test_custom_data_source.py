"""Custom DataSource subclasses must actually invalidate the cache.

The state hasher folds a DataSource's *token* into the cache key, so a custom
source must return a value that changes (a version/digest), not a bool. A bool
can't track changes - the cache would silently never invalidate - so Cash warns.
"""
from __future__ import annotations

import warnings

from cash import Cash, CashCacheIneffectiveWarning, DataSource, InMemoryBackend


class VersionSource(DataSource):
    """Returns the current version value as its token (the correct pattern)."""

    def __init__(self):
        self.version = "v1"

    def get_id(self) -> str:
        return "version"

    def has_changed(self):
        return self.version          # a VALUE, not a bool

    def update_state(self) -> None:
        pass


class TokenOverrideSource(DataSource):
    """Overrides state_token() directly; has_changed left as a bool."""

    def __init__(self):
        self.rev = 0

    def get_id(self) -> str:
        return "tok"

    def has_changed(self) -> bool:
        return True

    def update_state(self) -> None:
        pass

    def state_token(self):
        return self.rev


class BoolSource(DataSource):
    """The broken pattern: has_changed returns a bool."""

    def get_id(self) -> str:
        return "bool"

    def has_changed(self) -> bool:
        return True

    def update_state(self) -> None:
        pass


def test_value_source_invalidates_via_depends_on():
    src = VersionSource()
    c = Cash(backend=InMemoryBackend())
    calls = {"n": 0}

    @c.cache(depends_on=[src])
    def g(x):
        calls["n"] += 1
        return f"{x}:{src.version}"

    assert g(5) == "5:v1"
    src.version = "v2"
    assert g(5) == "5:v2"            # invalidated
    src.version = "v3"
    assert g(5) == "5:v3"
    assert calls["n"] == 3


def test_value_source_invalidates_via_dynamic_depends_on():
    src = VersionSource()
    c = Cash(backend=InMemoryBackend())
    calls = {"n": 0}

    @c.cache(dynamic_depends_on=lambda x: src)
    def g(x):
        calls["n"] += 1
        return f"{x}:{src.version}"

    g(5)
    src.version = "v2"
    g(5)
    assert calls["n"] == 2


def test_state_token_override_invalidates():
    src = TokenOverrideSource()
    c = Cash(backend=InMemoryBackend())
    calls = {"n": 0}

    @c.cache(depends_on=[src])
    def g(x):
        calls["n"] += 1
        return x

    g(1)
    src.rev = 1
    g(1)
    assert calls["n"] == 2


def test_bool_source_warns():
    # New source type name so the per-type one-shot warning isn't deduped away.
    src = BoolSource()
    c = Cash(backend=InMemoryBackend())

    @c.cache(depends_on=[src])
    def g(x):
        return x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Clear the module-level dedup so the warning can fire in this test.
        import cash.data_source as ds_mod
        ds_mod._warned_bool_token_sources.discard("BoolSource")
        g(1)
        assert any(
            issubclass(x.category, CashCacheIneffectiveWarning) and "bool" in str(x.message)
            for x in w
        ), [str(x.message) for x in w]
