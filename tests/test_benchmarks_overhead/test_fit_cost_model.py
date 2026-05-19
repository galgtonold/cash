import csv
from pathlib import Path

import pytest

from benchmarks.fit_cost_model import FitRow, fit_linear, fit_all, render_python_constants


def test_fit_linear_recovers_known_coefficients():
    # Generate synthetic points on y = 0.002 + 1e-9 * x
    xs = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    a_true, b_true = 0.002, 1e-9
    ys = [a_true + b_true * x for x in xs]
    a, b, r2 = fit_linear(xs, ys)
    assert abs(a - a_true) < 1e-9
    assert abs(b - b_true) < 1e-15
    assert r2 > 0.99


def test_fit_linear_handles_noisy_data():
    import random
    random.seed(0)
    xs = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    a_true, b_true = 0.001, 2e-9
    ys = [a_true + b_true * x + random.gauss(0, 0.0001) for x in xs]
    a, b, r2 = fit_linear(xs, ys)
    assert abs(a - a_true) < 0.001
    assert abs(b - b_true) < 1e-9
    assert r2 > 0.9


def test_fit_all_returns_one_row_per_family_backend_op(tmp_path):
    csv_path = tmp_path / "in.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "family", "target_bytes", "backend_kind", "repeats",
            "actual_size_bytes", "serialize_seconds", "deserialize_seconds", "error",
        ])
        w.writeheader()
        for family in ("a", "b"):
            for backend in ("ram", "disk"):
                for size in (1000, 10000, 100000):
                    w.writerow({
                        "family": family, "target_bytes": size,
                        "backend_kind": backend, "repeats": 2,
                        "actual_size_bytes": size,
                        "serialize_seconds": 1e-6 * size + 0.0001,
                        "deserialize_seconds": 5e-7 * size + 0.00005,
                        "error": "",
                    })
    fits = fit_all(csv_path)
    # 2 families × 2 backends × 2 ops = 8
    assert len(fits) == 8
    assert all(isinstance(f, FitRow) for f in fits)
    assert all(f.r_squared > 0.9 for f in fits)


def test_render_python_constants_produces_valid_dict_literal():
    fits = [
        FitRow(family="x", backend_kind="ram", operation="serialize",
               a=0.001, b=2e-9, r_squared=0.99, n_points=4),
        FitRow(family="x", backend_kind="ram", operation="deserialize",
               a=0.0005, b=1e-9, r_squared=0.95, n_points=4),
    ]
    rendered = render_python_constants(fits)
    # Should be a parseable Python dict literal in a string.
    assert '("x", "ram", "serialize"):' in rendered
    assert '("x", "ram", "deserialize"):' in rendered
    # Smoke parse it.
    ns: dict = {}
    exec(f"COEFFS = {rendered}", ns)
    assert ("x", "ram", "serialize") in ns["COEFFS"]
