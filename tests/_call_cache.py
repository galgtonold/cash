"""A ``CallCache`` outside a notebook, for unit tests of call interception.

The processor builds its ``CallCache`` with providers bound to the running
cell (lineage, loop variables, TTL, persist). Here they stand for a call at
the top level of a cell with no annotation and no lineage recorded.
"""

from __future__ import annotations

from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_unit import CallCache


def make_call_cache(cash_instance) -> CallCache:
    return CallCache(
        cash_instance,
        ctx_provider=lambda: CacheKeyContext(variable_lineage={}, user_ns={}),
        loop_vars_provider=dict,
        loop_var_digests_provider=dict,
        ttl_provider=lambda: None,
        persist_provider=lambda: False,
    )
