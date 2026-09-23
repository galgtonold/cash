"""Restoring a variable reports what the restore saved."""

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics


@pytest.fixture
def magics(mock_shell, cash_with_file_backend):
    """CashMagics over a disk cache."""
    return CashMagics(mock_shell, cash_with_file_backend)


def test_restore_variable_returns_metrics(magics, cash_with_file_backend):
    """
    Test that Restorer.restore_variable returns metrics with saved_time.
    """
    # 1. Setup Cache Entry
    var_name = "a"
    value = 100
    execution_time = 1.23
    source_code = "a = 100"

    # Construct cache key
    # Key usually is var_sources lookup.
    cache_key = "stmt:test_hash"
    magics.tracking_state.variable_sources[var_name] = cache_key

    metadata = {
        "inputs": [],
        "outputs": [var_name],
        "execution_time": execution_time,
        "code": source_code,
        "output_lineages": {var_name: "lineage_hash"},
    }

    data = {"variables": {var_name: value}}

    cash_with_file_backend.backend.set(cache_key, data, metadata)

    # 2. Call restore_variable
    # Ensure it's NOT in user_ns
    if var_name in magics.shell.user_ns:
        del magics.shell.user_ns[var_name]

    metrics = magics._restorer.restore_variable(var_name)

    # 3. Verify
    assert var_name in magics.shell.user_ns
    assert magics.shell.user_ns[var_name] == value

    print(f"Metrics: {metrics}")

    assert len(metrics) == 1
    m = metrics[0]
    assert m["status"] == CacheStatus.RESTORED
    assert m["saved_time"] == execution_time
    assert m["code"] == source_code
    assert "a" in m["restored_vars"]
