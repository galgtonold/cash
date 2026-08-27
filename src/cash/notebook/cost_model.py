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
#
# KNOWN: the disk-deserialize intercepts OVERESTIMATE small payloads by 7-25x.
# Checked against the frozen matrix these were fitted from -- its own small
# rows, not a separate experiment:
#
#     284 B sparse            measured 0.49 ms   fitted 10.75 ms   21.8x
#     942 B dict_shallow      measured 0.49 ms   fitted 10.39 ms   21.1x
#    1000 B bytes             measured 0.48 ms   fitted  9.50 ms   19.8x
#    1092 B dataframe_numeric measured 0.54 ms   fitted  8.11 ms   15.1x
#
# The cause is the fit objective, not the data: one line per family is fitted
# by ordinary least squares across 284 B to 111 MiB, and OLS minimises ABSOLUTE
# error, so the 100 MiB points dominate and the intercept absorbs their
# residual. A relative-error (or log-space) fit would price the small end
# correctly. Fixing it needs a refit plus re-validation, not an edit here --
# these numbers are generated, and hand-tuning them would desync them from the
# matrix they claim to come from.
#
# Currently harmless, which is why it is documented rather than fixed:
# `_SMART_PERSIST_COMPUTE_FLOOR_S` (100 ms) gates every decision below the
# range where the error lives, and of 504 promotion decisions above that floor
# only 2 (0.4%) flip if the intercept is corrected to the measured ~0.5 ms --
# both "promote a 100 MB write to save 250 ms", i.e. the wrong direction.
#
# It stops being harmless if that floor is ever lowered. Refit first.
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

    # ===== Redis (LAN) =====
    # Estimated, NOT benchmarked. Modelled as:
    #   a = (disk_a × 0.1) + 500us network round-trip
    #   b = max(disk_b, 20 ns/B)      ≈ 50 MB/s sustained
    # The 0.1× factor removes the disk I/O dominating disk's a; what's
    # left is the pure pickle work that's still required when serialising
    # to a network buffer.
    ("bytes",             "redis", "serialize"):   (6.011709e-04, 2.000000e-08),
    ("bytes",             "redis", "deserialize"): (1.450319e-03, 2.000000e-08),
    ("dataframe_numeric", "redis", "serialize"):   (5.330326e-04, 2.000000e-08),
    ("dataframe_numeric", "redis", "deserialize"): (1.310776e-03, 2.000000e-08),
    ("dict_shallow",      "redis", "serialize"):   (5.250868e-04, 2.000000e-08),
    ("dict_shallow",      "redis", "deserialize"): (1.538392e-03, 2.000000e-08),
    ("list_flat",         "redis", "serialize"):   (5.969259e-04, 2.000000e-08),
    ("list_flat",         "redis", "deserialize"): (1.361285e-03, 2.000000e-08),
    ("ndarray_dense",     "redis", "serialize"):   (5.678949e-04, 2.000000e-08),
    ("ndarray_dense",     "redis", "deserialize"): (1.460452e-03, 2.000000e-08),
    ("series_numeric",    "redis", "serialize"):   (6.362212e-04, 2.000000e-08),
    ("series_numeric",    "redis", "deserialize"): (1.778298e-03, 2.000000e-08),
    ("sparse",            "redis", "serialize"):   (6.758385e-04, 2.000000e-08),
    ("sparse",            "redis", "deserialize"): (1.575261e-03, 2.000000e-08),
    # Generic fallback: copy the slowest disk family with the network
    # transform applied — series_numeric deserialize is the worst case.
    ("_GENERIC",          "redis", "serialize"):   (6.758385e-04, 2.000000e-08),
    ("_GENERIC",          "redis", "deserialize"): (1.778298e-03, 2.000000e-08),

    # ===== S3 (same-region) =====
    # Estimated, NOT benchmarked. Modelled as:
    #   a = (disk_a × 0.2) + 80ms request setup (DNS/TLS/auth)
    #   b = max(disk_b, 50 ns/B)      ≈ 20 MB/s single-stream
    # Cross-region / public-internet S3 is 5–10× slower; users with
    # those topologies should override the relevant backend's
    # ``bandwidth_estimate``. The 80ms latency floor dominates objects
    # under ~2 MB.
    ("bytes",             "s3", "serialize"):   (8.020234e-02, 5.000000e-08),
    ("bytes",             "s3", "deserialize"): (8.190064e-02, 5.000000e-08),
    ("dataframe_numeric", "s3", "serialize"):   (8.006607e-02, 5.000000e-08),
    ("dataframe_numeric", "s3", "deserialize"): (8.162155e-02, 5.000000e-08),
    ("dict_shallow",      "s3", "serialize"):   (8.005017e-02, 5.000000e-08),
    ("dict_shallow",      "s3", "deserialize"): (8.207678e-02, 5.000000e-08),
    ("list_flat",         "s3", "serialize"):   (8.019385e-02, 5.000000e-08),
    ("list_flat",         "s3", "deserialize"): (8.172257e-02, 5.000000e-08),
    ("ndarray_dense",     "s3", "serialize"):   (8.013579e-02, 5.000000e-08),
    ("ndarray_dense",     "s3", "deserialize"): (8.192090e-02, 5.000000e-08),
    ("series_numeric",    "s3", "serialize"):   (8.027244e-02, 5.000000e-08),
    ("series_numeric",    "s3", "deserialize"): (8.255660e-02, 5.000000e-08),
    ("sparse",            "s3", "serialize"):   (8.035168e-02, 5.000000e-08),
    ("sparse",            "s3", "deserialize"): (8.215052e-02, 5.000000e-08),
    ("_GENERIC",          "s3", "serialize"):   (8.035168e-02, 5.000000e-08),
    ("_GENERIC",          "s3", "deserialize"): (8.255660e-02, 5.000000e-08),
}

_KNOWN_BACKENDS = frozenset({"ram", "disk", "redis", "s3"})


def resolve_family(value_type_name: str) -> str:
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
        resolve_family(value_type_name),
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
        resolve_family(value_type_name),
        size_bytes,
        _resolve_backend(backend_kind),
        "deserialize",
    )
