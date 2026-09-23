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
    "tuple": "list_flat",  # treat like list for cost
    "bytes": "bytes",
    "bytearray": "bytes",
}

# (family, backend_kind, operation) -> (a, b)
# Fitted from `benchmarks/fit_cost_model.py benchmarks/results/ser_deser_matrix.frozen.csv
# --objective relative`. Both the matrix and the objective have changed since
# the first fit, and the numbers moved enough to be worth explaining.
#
# WHAT THE OLD MATRIX MEASURED. Three faults, none of them in the fit:
#
#   * `set(k)` was followed straight by `get(k)`. A file backend hands the disk
#     write to a background worker, and a `get` of that key WAITS for it, so
#     every "deserialize" number carried part of a write -- erratically, in
#     whatever amount the worker happened to have finished. The matrix showed it
#     plainly: list_flat read 11.8 KB in 0.47 ms and 115 KB in 14.94 ms.
#   * the read went through the backend object that had just written the file.
#     That costs ~4 ms per open on Windows against ~0.1 ms through another
#     backend, and ~1.5 ms from a new process.
#   * each cell leaked a backend, and a backend is about eight threads. By a
#     dozen of them the same 0.66 ms read measured 4.70 ms, and worse the deeper
#     into the run it got.
#
# WHAT THIS ONE MEASURES. A disk read is timed in a fresh process, reading an
# entry no process has read before, with the heavy imports already done. That is
# the regime a promotion decision is about: a persistent tier exists to serve a
# LATER SESSION, because within a session the RAM tier answers first. RAM rows
# stay in-process for the same reason. Backends are shut down per cell.
#
# WHY RELATIVE. OLS minimises error in SECONDS, so across 284 B to 111 MiB the
# 100 MiB points set the intercept and the small end is priced by their
# residual. `--objective relative` weights each point by 1/y^2, minimising error
# as a RATIO, which is what a decision comparing two magnitudes needs.
#
# WHAT IT IS WORTH, scored as decisions rather than residuals, over 42 cells x 9
# body times (`benchmarks/_fit_decision_check.py`):
#
#     shipped before   37 of 378 wrong (9.8%)   37 kept out, 0 let in
#     these constants   0 of 378 wrong (0.0%)
#
# "Kept out" means persisting would have paid and cash refused -- the user loses
# a hit they should have had. Every one of the old constants' mistakes was that
# one: a 100 MB frame or array from a body of 100 ms, refused disk although
# restoring it measures 70 ms.
#
# These numbers are GENERATED. Re-measure and refit rather than editing them, or
# they desync from the matrix they claim to come from. They describe one machine
# (NVMe, Windows), where opening a small entry from a fresh process costs ~5-6 ms
# and dominates every read below about a megabyte.
_COEFFS: dict[tuple[str, str, str], tuple[float, float]] = {
    ("_GENERIC", "disk", "deserialize"): (5.478340e-03, 1.583327e-09),  # derived, not measured
    ("_GENERIC", "disk", "serialize"): (4.820535e-04, 1.497407e-09),  # derived, not measured
    ("_GENERIC", "ram", "deserialize"): (5.840887e-06, 2.720998e-09),  # derived, not measured
    ("_GENERIC", "ram", "serialize"): (1.877952e-05, 1.783824e-08),  # derived, not measured
    ("_GENERIC", "redis", "deserialize"): (1.158999e-03, 2.000000e-08),  # derived, not measured
    ("_GENERIC", "redis", "serialize"): (5.793690e-04, 2.000000e-08),  # derived, not measured
    ("_GENERIC", "s3", "deserialize"): (8.131800e-02, 5.000000e-08),  # derived, not measured
    ("_GENERIC", "s3", "serialize"): (8.015874e-02, 5.000000e-08),  # derived, not measured
    ("bytes", "disk", "deserialize"): (6.589985e-03, 6.508211e-10),  # R2=1.000 n=6 worst=1.1x
    ("bytes", "disk", "serialize"): (6.787004e-04, 1.192182e-09),  # R2=0.827 n=6 worst=1.6x
    ("bytes", "ram", "deserialize"): (5.299982e-06, 0.000000e00),  # R2=-0.364 n=6 worst=1.1x
    ("bytes", "ram", "serialize"): (1.615001e-05, 0.000000e00),  # R2=-0.028 n=6 worst=1.1x
    ("bytes", "redis", "deserialize"): (1.158999e-03, 2.000000e-08),  # derived, not measured
    ("bytes", "redis", "serialize"): (5.678700e-04, 2.000000e-08),  # derived, not measured
    ("bytes", "s3", "deserialize"): (8.131800e-02, 5.000000e-08),  # derived, not measured
    ("bytes", "s3", "serialize"): (8.013574e-02, 5.000000e-08),  # derived, not measured
    ("dataframe_numeric", "disk", "deserialize"): (6.059236e-03, 6.580528e-10),  # R2=0.999 n=6 worst=1.1x
    ("dataframe_numeric", "disk", "serialize"): (4.820535e-04, 1.497407e-09),  # R2=0.962 n=6 worst=1.7x
    ("dataframe_numeric", "ram", "deserialize"): (1.446263e-05, 2.214465e-10),  # R2=0.998 n=6 worst=1.7x
    ("dataframe_numeric", "ram", "serialize"): (3.578494e-05, 1.316268e-10),  # R2=0.903 n=6 worst=1.6x
    ("dataframe_numeric", "redis", "deserialize"): (1.105924e-03, 2.000000e-08),  # derived, not measured
    ("dataframe_numeric", "redis", "serialize"): (5.482053e-04, 2.000000e-08),  # derived, not measured
    ("dataframe_numeric", "s3", "deserialize"): (8.121185e-02, 5.000000e-08),  # derived, not measured
    ("dataframe_numeric", "s3", "serialize"): (8.009641e-02, 5.000000e-08),  # derived, not measured
    ("dict_shallow", "disk", "deserialize"): (5.478340e-03, 1.583327e-09),  # R2=0.961 n=6 worst=1.2x
    ("dict_shallow", "disk", "serialize"): (6.634816e-04, 1.424603e-09),  # R2=0.824 n=6 worst=1.6x
    ("dict_shallow", "ram", "deserialize"): (5.840887e-06, 2.720998e-09),  # R2=0.851 n=6 worst=1.5x
    ("dict_shallow", "ram", "serialize"): (1.877952e-05, 1.783824e-08),  # R2=0.957 n=6 worst=1.2x
    ("dict_shallow", "redis", "deserialize"): (1.047834e-03, 2.000000e-08),  # derived, not measured
    ("dict_shallow", "redis", "serialize"): (5.663482e-04, 2.000000e-08),  # derived, not measured
    ("dict_shallow", "s3", "deserialize"): (8.109567e-02, 5.000000e-08),  # derived, not measured
    ("dict_shallow", "s3", "serialize"): (8.013270e-02, 5.000000e-08),  # derived, not measured
    ("list_flat", "disk", "deserialize"): (5.718911e-03, 8.986306e-10),  # R2=1.000 n=6 worst=1.1x
    ("list_flat", "disk", "serialize"): (6.875598e-04, 1.094875e-09),  # R2=0.966 n=6 worst=1.2x
    ("list_flat", "ram", "deserialize"): (9.606476e-06, 9.279072e-10),  # R2=0.748 n=6 worst=1.8x
    ("list_flat", "ram", "serialize"): (2.903120e-05, 3.402422e-09),  # R2=0.978 n=6 worst=1.2x
    ("list_flat", "redis", "deserialize"): (1.071891e-03, 2.000000e-08),  # derived, not measured
    ("list_flat", "redis", "serialize"): (5.687560e-04, 2.000000e-08),  # derived, not measured
    ("list_flat", "s3", "deserialize"): (8.114378e-02, 5.000000e-08),  # derived, not measured
    ("list_flat", "s3", "serialize"): (8.013751e-02, 5.000000e-08),  # derived, not measured
    ("ndarray_dense", "disk", "deserialize"): (5.665687e-03, 6.934721e-10),  # R2=0.995 n=6 worst=1.1x
    ("ndarray_dense", "disk", "serialize"): (7.091170e-04, 1.411240e-09),  # R2=0.978 n=6 worst=1.2x
    ("ndarray_dense", "ram", "deserialize"): (8.302696e-06, 2.172459e-10),  # R2=0.995 n=6 worst=1.8x
    ("ndarray_dense", "ram", "serialize"): (1.809576e-05, 5.370905e-11),  # R2=0.391 n=6 worst=3.3x
    ("ndarray_dense", "redis", "deserialize"): (1.066569e-03, 2.000000e-08),  # derived, not measured
    ("ndarray_dense", "redis", "serialize"): (5.709117e-04, 2.000000e-08),  # derived, not measured
    ("ndarray_dense", "s3", "deserialize"): (8.113314e-02, 5.000000e-08),  # derived, not measured
    ("ndarray_dense", "s3", "serialize"): (8.014182e-02, 5.000000e-08),  # derived, not measured
    ("series_numeric", "disk", "deserialize"): (6.257127e-03, 6.480138e-10),  # R2=1.000 n=6 worst=1.1x
    ("series_numeric", "disk", "serialize"): (7.537178e-04, 1.398579e-09),  # R2=0.982 n=6 worst=1.1x
    ("series_numeric", "ram", "deserialize"): (2.595869e-05, 2.257319e-10),  # R2=0.999 n=6 worst=1.5x
    ("series_numeric", "ram", "serialize"): (5.158808e-05, 9.495364e-11),  # R2=0.694 n=6 worst=2.2x
    ("series_numeric", "redis", "deserialize"): (1.125713e-03, 2.000000e-08),  # derived, not measured
    ("series_numeric", "redis", "serialize"): (5.753718e-04, 2.000000e-08),  # derived, not measured
    ("series_numeric", "s3", "deserialize"): (8.125143e-02, 5.000000e-08),  # derived, not measured
    ("series_numeric", "s3", "serialize"): (8.015074e-02, 5.000000e-08),  # derived, not measured
    ("sparse", "disk", "deserialize"): (5.649893e-03, 9.034943e-10),  # R2=0.996 n=6 worst=1.1x
    ("sparse", "disk", "serialize"): (7.936895e-04, 1.432005e-09),  # R2=1.000 n=6 worst=1.0x
    ("sparse", "ram", "deserialize"): (1.890150e-05, 2.047055e-10),  # R2=0.960 n=6 worst=1.3x
    ("sparse", "ram", "serialize"): (3.213248e-05, 8.440021e-11),  # R2=0.562 n=6 worst=2.5x
    ("sparse", "redis", "deserialize"): (1.064989e-03, 2.000000e-08),  # derived, not measured
    ("sparse", "redis", "serialize"): (5.793690e-04, 2.000000e-08),  # derived, not measured
    ("sparse", "s3", "deserialize"): (8.112998e-02, 5.000000e-08),  # derived, not measured
    ("sparse", "s3", "serialize"): (8.015874e-02, 5.000000e-08),  # derived, not measured
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
