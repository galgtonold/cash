"""Tuned cost model for the cache-or-not decision.

Returns predicted seconds for serialise / deserialise of an object of
given type + size on a given backend. Constants come from the offline
measurement campaign in ``benchmarks/measure_ser_deser.py`` + fitted
by ``benchmarks/fit_cost_model.py``. Re-run those scripts and refit
to refresh the constants.

Constants source: ``benchmarks/results/ser_deser_matrix.frozen.csv``
(committed alongside this module).
"""
from __future__ import annotations

# Map runtime ``type(value).__name__`` strings to the family used in the
# fit. Anything not present here routes to the ``_GENERIC`` family
# (slowest observed in the measurement run).
_TYPE_TO_FAMILY: dict[str, str] = {
    "DataFrame": "dataframe_numeric",
    "Series": "series_numeric",
    "ndarray": "ndarray_dense",
    "csr_matrix": "sparse",
    "csc_matrix": "sparse",
    "dict": "dict_shallow",
    "list": "list_flat",
    "tuple": "list_flat",          # treat like list for cost
    "bytes": "bytes",
    "bytearray": "bytes",
}

# (family, backend_kind, operation) -> (a, b)
# Fitted from `benchmarks/fit_cost_model.py benchmarks/results/ser_deser_matrix.csv`.
# _GENERIC entries are copies of the slowest measured family per (backend, op).
_COEFFS: dict[tuple[str, str, str], tuple[float, float]] = {
    # ===== Measured families =====
    ("bytes", "disk", "deserialize"): (9.503194e-03, 4.242689e-10),  # R2=0.876
    ("bytes", "disk", "serialize"): (1.011709e-03, 6.050071e-10),  # R2=0.997
    ("bytes", "ram", "deserialize"): (2.117042e-06, -4.520247e-15),  # R2=0.052 — too fast to model
    ("bytes", "ram", "serialize"): (6.865110e-06, -1.341592e-14),  # R2=0.048 — too fast to model
    ("dataframe_numeric", "disk", "deserialize"): (8.107757e-03, 1.576640e-09),  # R2=0.995
    ("dataframe_numeric", "disk", "serialize"): (3.303260e-04, 6.463715e-10),  # R2=0.998
    ("dataframe_numeric", "ram", "deserialize"): (8.628838e-06, 1.604019e-10),  # R2=1.000
    ("dataframe_numeric", "ram", "serialize"): (2.146195e-04, 1.533608e-10),  # R2=0.999
    ("dict_shallow", "disk", "deserialize"): (1.038392e-02, 1.976465e-09),  # R2=0.989
    ("dict_shallow", "disk", "serialize"): (2.508677e-04, 1.077603e-09),  # R2=0.998
    ("dict_shallow", "ram", "deserialize"): (-1.326852e-03, 1.697479e-09),  # R2=0.999
    ("dict_shallow", "ram", "serialize"): (-3.217562e-03, 9.200596e-09),  # R2=1.000
    ("list_flat", "disk", "deserialize"): (8.612852e-03, 1.395863e-09),  # R2=0.989
    ("list_flat", "disk", "serialize"): (9.692593e-04, 5.769994e-10),  # R2=1.000
    ("list_flat", "ram", "deserialize"): (-3.290309e-04, 1.713300e-09),  # R2=1.000
    ("list_flat", "ram", "serialize"): (-9.765199e-04, 1.315996e-08),  # R2=1.000
    ("ndarray_dense", "disk", "deserialize"): (9.604519e-03, 1.547854e-09),  # R2=0.995
    ("ndarray_dense", "disk", "serialize"): (6.789494e-04, 6.642331e-10),  # R2=0.997
    ("ndarray_dense", "ram", "deserialize"): (2.794288e-05, 1.557365e-10),  # R2=1.000
    ("ndarray_dense", "ram", "serialize"): (-3.322399e-05, 1.372613e-10),  # R2=1.000
    ("series_numeric", "disk", "deserialize"): (1.278298e-02, 1.537032e-09),  # R2=0.986
    ("series_numeric", "disk", "serialize"): (1.362212e-03, 5.740690e-10),  # R2=0.996
    ("series_numeric", "ram", "deserialize"): (1.132358e-04, 1.651489e-10),  # R2=1.000
    ("series_numeric", "ram", "serialize"): (-4.913191e-06, 1.440106e-10),  # R2=1.000
    ("sparse", "disk", "deserialize"): (1.075261e-02, 1.659304e-09),  # R2=0.516 — sparse high-variance
    ("sparse", "disk", "serialize"): (1.758385e-03, 4.341337e-10),  # R2=0.871
    ("sparse", "ram", "deserialize"): (1.040403e-05, 1.678533e-10),  # R2=1.000
    ("sparse", "ram", "serialize"): (-1.553164e-05, 1.329718e-10),  # R2=0.996

    # ===== _GENERIC fallback (slowest observed per backend, op) =====
    # ram serialize: list_flat is slowest (13.16ns/byte at 100MB → 1.32s)
    ("_GENERIC", "ram", "serialize"): (-9.765199e-04, 1.315996e-08),
    # ram deserialize: list_flat is slowest (1.71ns/byte at 100MB → 0.17s)
    ("_GENERIC", "ram", "deserialize"): (-3.290309e-04, 1.713300e-09),
    # disk serialize: dict_shallow is slowest at large sizes
    ("_GENERIC", "disk", "serialize"): (2.508677e-04, 1.077603e-09),
    # disk deserialize: dict_shallow is slowest at large sizes
    ("_GENERIC", "disk", "deserialize"): (1.038392e-02, 1.976465e-09),
}

_KNOWN_BACKENDS = frozenset({"ram", "disk"})


def _resolve_family(value_type_name: str) -> str:
    return _TYPE_TO_FAMILY.get(value_type_name, "_GENERIC")


def _resolve_backend(backend_kind: str) -> str:
    return backend_kind if backend_kind in _KNOWN_BACKENDS else "disk"


def _predict(family: str, size_bytes: int, backend: str, op: str) -> float:
    a, b = _COEFFS[(family, backend, op)]
    return a + b * size_bytes


def estimated_serialize_time(
    value_type_name: str,
    size_bytes: int,
    backend_kind: str,
) -> float:
    """Predicted wall-seconds to serialise + store an object of given
    type / size on the given backend."""
    return _predict(
        _resolve_family(value_type_name),
        size_bytes,
        _resolve_backend(backend_kind),
        "serialize",
    )


def estimated_restore_time(
    value_type_name: str,
    size_bytes: int,
    backend_kind: str,
) -> float:
    """Predicted wall-seconds to load + deserialise an object of given
    type / size on the given backend."""
    return _predict(
        _resolve_family(value_type_name),
        size_bytes,
        _resolve_backend(backend_kind),
        "deserialize",
    )
