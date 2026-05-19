import pytest

from cash.notebook.cost_model import (
    estimated_restore_time,
    estimated_serialize_time,
    _TYPE_TO_FAMILY,
    _COEFFS,
)


def test_known_type_returns_positive_prediction():
    t = estimated_restore_time("DataFrame", size_bytes=1_000_000, backend_kind="disk")
    assert t > 0


def test_serialize_and_restore_are_both_predicted():
    s = estimated_serialize_time("ndarray", size_bytes=10_000_000, backend_kind="disk")
    r = estimated_restore_time("ndarray", size_bytes=10_000_000, backend_kind="disk")
    assert s > 0 and r > 0
    # Sanity: on disk, ser and deser are within a factor of 10 of each other.
    # (If one is wildly larger, something is off.)
    assert 0.1 < s / r < 10


def test_unknown_type_falls_back_to_generic():
    """An unrecognised type name should map to the _GENERIC family."""
    t_unknown = estimated_restore_time("SomeWeirdType", size_bytes=1_000_000,
                                       backend_kind="disk")
    t_generic = estimated_restore_time("_GENERIC", size_bytes=1_000_000,
                                       backend_kind="disk")
    # We expect identical predictions (both routed to _GENERIC).
    assert t_unknown == t_generic


def test_unknown_backend_falls_back_to_disk():
    """Unknown backend_kind falls back to 'disk' (the slower path)."""
    t_unknown = estimated_restore_time("DataFrame", size_bytes=1_000_000,
                                       backend_kind="moonbeam")
    t_disk = estimated_restore_time("DataFrame", size_bytes=1_000_000,
                                    backend_kind="disk")
    assert t_unknown == t_disk


def test_linear_math_is_correct():
    """estimated_*_time = a + b × size for the (family, backend, op) tuple."""
    family = "dataframe_numeric"
    backend = "disk"
    a, b = _COEFFS[(family, backend, "deserialize")]
    size = 5_000_000
    expected = a + b * size
    # DataFrame maps to dataframe_numeric per _TYPE_TO_FAMILY.
    actual = estimated_restore_time("DataFrame", size_bytes=size, backend_kind=backend)
    assert abs(actual - expected) < 1e-9


def test_predictions_increase_with_size_within_family():
    sizes = [1_000, 10_000, 1_000_000, 100_000_000]
    times = [estimated_restore_time("DataFrame", s, "disk") for s in sizes]
    # Predictions should be monotonically increasing.
    assert times == sorted(times)


def test_type_family_map_includes_documented_types():
    expected_keys = {
        "DataFrame", "Series", "ndarray", "csr_matrix",
        "dict", "list", "bytes",
    }
    assert expected_keys.issubset(set(_TYPE_TO_FAMILY))


def test_coeffs_table_covers_all_known_families_and_backends():
    families = set(_TYPE_TO_FAMILY.values()) | {"_GENERIC"}
    for family in families:
        for backend in ("ram", "disk"):
            for op in ("serialize", "deserialize"):
                assert (family, backend, op) in _COEFFS, (
                    f"missing constant for ({family!r}, {backend!r}, {op!r})"
                )


def test_cost_model_is_called_from_statement_processor_decision(monkeypatch, tmp_path):
    """End-to-end: when _should_skip_large_object_caching evaluates a
    decision, it routes through cost_model.estimated_restore_time."""
    from unittest.mock import MagicMock
    from cash.notebook import cost_model

    calls: list[tuple] = []
    real_fn = cost_model.estimated_restore_time

    def spy(type_name, size_bytes, backend_kind):
        calls.append((type_name, size_bytes, backend_kind))
        return real_fn(type_name, size_bytes, backend_kind)

    monkeypatch.setattr(cost_model, "estimated_restore_time", spy)

    from cash.backends.backend import FileBackend
    from cash.core import Cash
    from cash.notebook.statement_processor import StatementProcessor

    backend = FileBackend(str(tmp_path))
    cash = Cash(backend=backend, register_magic=False)
    shell = MagicMock()
    shell.user_ns = {}
    proc = StatementProcessor(cash_instance=cash, shell=shell, debug=False)

    # Moderate object — just enough that the cost_model is consulted.
    # We don't assert skip=True (depends on fitted constants); the
    # crucial assertion is that the policy routed through cost_model.
    medium_obj = b"x" * 1_000_000  # 1 MB
    # _should_skip_large_object_caching(captured_vars, execution_time,
    #                                   force_persist, has_file_dependencies)
    proc._should_skip_large_object_caching(
        captured_vars={"medium": medium_obj},
        execution_time=0.1,
        force_persist=False,
        has_file_dependencies=False,
    )
    assert any(c[0] == "bytes" for c in calls), (
        f"expected estimated_restore_time call with type_name='bytes', got {calls}"
    )
