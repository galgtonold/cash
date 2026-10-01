"""A tier stack whose first tier is not the RAM tier reports where entries
really went.

The configuration accepts any tier first (``[[tool.cash.tiers]] type =
"file"``). A value written there is on disk, and its ``storage`` must say
so: the notebook, ``explain()`` and the decorator's store-failed check all
read "only RAM" off it as "gone at the next restart".
"""

from __future__ import annotations

from cash.backends.factory import build_backend_from_config
from cash.config.schema import CashConfig, TierConfig


def test_a_file_first_tier_is_reported_as_disk(tmp_path):
    backend = build_backend_from_config(CashConfig(tiers=[TierConfig(type="file", cache_dir=str(tmp_path))]))
    metadata = {"execution_time": 1.0}
    backend.set("k", 42, metadata)
    assert metadata["storage"] == ["DISK"]
    backend.shutdown()


def test_a_cheap_value_in_a_file_first_stack_is_not_called_unpersisted(tmp_path):
    # Control: with the RAM tier first, the same value stays in RAM only.
    ram_first = build_backend_from_config(
        CashConfig(tiers=[TierConfig(type="memory"), TierConfig(type="file", cache_dir=str(tmp_path / "a"))])
    )
    control = {"execution_time": 0.001}
    ram_first.set("k", 42, control)
    assert control["storage"] == ["RAM"]
    assert control["persist_skipped"] == "compute"
    ram_first.shutdown()

    backend = build_backend_from_config(
        CashConfig(
            tiers=[TierConfig(type="file", cache_dir=str(tmp_path)), TierConfig(type="sqlite", cache_dir=str(tmp_path))]
        )
    )
    metadata = {"execution_time": 0.001}
    backend.set("k", 42, metadata)
    assert metadata["storage"][0] == "DISK"
    assert "persist_skipped" not in metadata
    backend.shutdown()
