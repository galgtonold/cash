"""`cash inspect` names the function a call-cache entry belongs to.

Round 28, r28s1: "entries in `cash inspect` are all named `call`, so I can't
tell which function each one belongs to". Call-cache keys are `call:<sha>`,
and `inspect` names an entry by the key's first segment, which for every
intercepted call is the literal `call`. The entry now records its function.
"""

import time

from cash.__main__ import _function_of
from cash.notebook.call_interception import CallSite


def test_a_call_entry_records_its_function(call_unit_harness):
    def net_returns(k):
        time.sleep(0.01)
        return k * 2

    unit = call_unit_harness(lineage={"k": "h"}, user_ns={"k": 2, "net_returns": net_returns})
    site = CallSite(source="net_returns(k)", free_names=frozenset({"net_returns", "k"}), occurrence_index=0)
    unit.wrap(net_returns, site)(2)
    backend = unit._cash.backend
    stores = [getattr(t, "_store", {}) for t in getattr(backend, "backends", [backend])]
    metas = [meta for store in stores for key, (meta, _v) in store.items() if str(key).startswith("call:")]
    assert metas, "the call was not stored"
    assert any("net_returns" in str(m.get("function", "")) for m in metas), metas


def test_inspect_prefers_the_recorded_function():
    assert _function_of("call:abc", {"function": "__main__.net_returns"}) == "__main__.net_returns"
    assert _function_of("call:abc", {}) == "call"
    assert _function_of("stmt:abc", {}) == "(notebook statements)"
    assert _function_of("mod.f:1:2:3") == "mod.f"
