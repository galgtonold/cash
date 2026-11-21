"""Whether a value the held-function walk meets is JSON-like data alone is
told by one C pass that only reads types, not by its facts
(`_plain_data.whole_tree_facts`), which also size every level: 0.25 s for
a million records at the first statement reading them, against 0.1 s.
"""

from __future__ import annotations

import pytest

from cash import _native, _plain_data
from cash.notebook import held_functions as hf

pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")


def test_records_are_told_plain_without_their_facts(monkeypatch):
    def refused(value):
        raise AssertionError("the records' facts were made")

    monkeypatch.setattr(hf, "whole_tree_facts", refused)
    monkeypatch.setattr(_plain_data, "whole_tree_facts", refused)
    namespace: dict = {"__name__": "__main__"}
    exec("def f(x):\n    return x + 1\n", namespace)
    records = [{"i": i, "v": [i, i + 1]} for i in range(30_000)]
    held = hf.held_notebook_code(records, namespace, "lineage-told-without-facts")
    assert held.codes == () and held.complete
    records[7]["v"].append(namespace["f"])
    held = hf.held_notebook_code(records, namespace, "lineage-told-without-facts-2")
    assert held.codes == (namespace["f"].__code__,)
