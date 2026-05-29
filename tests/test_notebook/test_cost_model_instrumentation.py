"""Tests for cost-model prediction-input exposure on ProcessResult.

Verifies that the four cost-model fields (cost_model_size_bytes,
cost_model_restore_seconds, cost_model_type_name, cost_model_family) are
present on ProcessResult when the budget check runs and absent when the
10 ms floor short-circuits before it.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from traitlets.config.configurable import Configurable

from cash.core import Cash
from cash.backends import InMemoryBackend
from cash.notebook.statement import StatementProcessor

_COST_MODEL_KEYS = (
    'cost_model_size_bytes',
    'cost_model_restore_seconds',
    'cost_model_type_name',
    'cost_model_family',
)


class MockShell(Configurable):
    """Minimal IPython shell mock for StatementProcessor."""
    def __init__(self):
        super().__init__()
        self.user_ns: dict = {}
        self.input_transformers_cleanup: list = []
        self.ast_transformers: list = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def processor():
    """StatementProcessor backed by InMemoryBackend.

    Using process_statement directly rather than magics.cash because
    process_statement returns the ProcessResult dict, letting us assert
    on the cost-model fields without monkey-patching the call chain.
    """
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    proc = StatementProcessor(shell=shell, cash_instance=cash, debug=False)
    yield proc
    backend.clear()


class TestCostModelFieldsPopulated:
    """test_cost_model_fields_populated_on_budget_check:
    A non-trivial array statement must reach the budget check and expose
    all four cost-model fields on the returned ProcessResult.
    """

    def test_cost_model_fields_populated_on_budget_check(self, processor):
        """A large array forces the budget check and must surface all four fields."""
        # np.zeros(5_000_000) → 40 MB float64 array; execution_time will be
        # real (measured), which keeps it above the 10 ms floor only if
        # execution genuinely takes >= 10 ms.  Use a smaller computation but
        # override the floor to 0 so the budget check always runs.
        config = MagicMock()
        config.min_execution_time_to_cache_seconds = 0.0
        config.min_cache_savings_pct = 0.20
        config.min_cache_fixed_budget_seconds = 0.05
        processor.cash_instance.config = config

        result = processor.process_statement("arr = [i for i in range(100000)]")

        for key in _COST_MODEL_KEYS:
            assert key in result, (
                f"Expected cost-model field '{key}' in ProcessResult, got keys: {list(result.keys())}"
            )

        assert isinstance(result['cost_model_size_bytes'], int)
        assert result['cost_model_size_bytes'] > 0
        assert isinstance(result['cost_model_restore_seconds'], float)
        assert isinstance(result['cost_model_type_name'], str)
        assert result['cost_model_type_name'] == 'list'
        assert isinstance(result['cost_model_family'], str)
        assert result['cost_model_family'] != ''

    def test_cost_model_fields_on_large_ndarray(self, processor):
        """A numpy array statement should also surface cost-model fields with correct type."""
        config = MagicMock()
        config.min_execution_time_to_cache_seconds = 0.0
        config.min_cache_savings_pct = 0.20
        config.min_cache_fixed_budget_seconds = 0.05
        processor.cash_instance.config = config

        # Inject the array directly into the namespace to avoid import overhead
        import numpy as np  # noqa: PLC0415
        processor.shell.user_ns['np'] = np
        result = processor.process_statement("arr = np.zeros(5_000_000)")

        for key in _COST_MODEL_KEYS:
            assert key in result, (
                f"Expected cost-model field '{key}' in ProcessResult; got: {list(result.keys())}"
            )
        assert result['cost_model_type_name'] == 'ndarray'
        assert result['cost_model_family'] == 'ndarray_dense'


class TestCostModelFieldsAbsentOnFloorExit:
    """test_cost_model_fields_absent_on_floor_exit:
    A trivially cheap statement short-circuits via the 10 ms floor —
    no metadata entry is written and the four cost-model fields must
    not appear on ProcessResult.
    """

    def test_cost_model_fields_absent_on_floor_exit(self, processor):
        """a = 1 is below the 10 ms floor; backend stays empty; fields absent."""
        # Use the default floor (10 ms).  The assignment 'a = 1' executes in
        # microseconds and must not produce a cache entry.
        result = processor.process_statement("a = 1")

        for key in _COST_MODEL_KEYS:
            assert key not in result, (
                f"Cost-model field '{key}' should be absent when floor short-circuits, "
                f"but it is present with value {result.get(key)!r}"
            )

        # Confirm the backend is truly empty (no metadata-only entry written).
        backend = processor.cash_instance.backend
        assert len(backend._store) == 0, (
            f"Backend should be empty after floor exit, but has {len(backend._store)} entries"
        )
