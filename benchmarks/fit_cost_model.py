"""Fit linear cost models from the measurement matrix CSV.

For each (family, backend_kind, operation) group, fits
    predicted_seconds = a + b × size_bytes
via ordinary least squares and reports a, b, R². Output is a
Python dict literal of {(family, backend, op): (a, b)} that the
production module pastes in.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class FitRow:
    family: str
    backend_kind: str
    operation: str  # "serialize" | "deserialize"
    a: float
    b: float
    r_squared: float
    n_points: int


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """OLS fit y = a + b*x. Returns (a, b, r_squared)."""
    if len(xs) < 2:
        return 0.0, 0.0, 0.0
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return sy / n, 0.0, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    # R²
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def fit_all(csv_path: Path) -> list[FitRow]:
    """Read measurement CSV, group by (family, backend, op), fit each group."""
    groups: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["error"]:
                continue
            family = row["family"]
            backend = row["backend_kind"]
            size = float(row["actual_size_bytes"])
            ser = float(row["serialize_seconds"])
            deser = float(row["deserialize_seconds"])
            groups.setdefault((family, backend, "serialize"), []).append((size, ser))
            groups.setdefault((family, backend, "deserialize"), []).append((size, deser))

    fits: list[FitRow] = []
    for (family, backend, op), pts in groups.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a, b, r2 = fit_linear(xs, ys)
        fits.append(FitRow(
            family=family, backend_kind=backend, operation=op,
            a=a, b=b, r_squared=r2, n_points=len(pts),
        ))
    return fits


def render_python_constants(fits: list[FitRow]) -> str:
    """Render a Python dict literal for paste into cost_model.py."""
    lines = ["{"]
    for f in sorted(fits, key=lambda x: (x.family, x.backend_kind, x.operation)):
        lines.append(
            f'    ("{f.family}", "{f.backend_kind}", "{f.operation}"): '
            f"({f.a:.6e}, {f.b:.6e}),  # R2={f.r_squared:.3f} n={f.n_points}"
        )
    lines.append("}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit cost model from matrix CSV")
    p.add_argument("csv_path", type=Path)
    p.add_argument("--min-r2", type=float, default=0.8,
                   help="Warn if any fit has R² below this threshold")
    args = p.parse_args(argv)

    fits = fit_all(args.csv_path)

    print("# Fit summary (sorted by R² ascending):")
    for f in sorted(fits, key=lambda x: x.r_squared):
        flag = "  [LOW R2]" if f.r_squared < args.min_r2 else ""
        print(f"  {f.family:18s} {f.backend_kind:4s} {f.operation:11s}  "
              f"a={f.a:.3e} b={f.b:.3e} R2={f.r_squared:.3f} n={f.n_points}{flag}")

    bad = [f for f in fits if f.r_squared < args.min_r2]
    if bad:
        print(f"\n[WARN] {len(bad)} fits below R2={args.min_r2} threshold -- linear model "
              "may be wrong shape for these (family, backend, op) tuples. "
              "Investigate before pasting constants.")

    print("\n# COEFFS dict (paste into src/cash/notebook/cost_model.py):")
    print(render_python_constants(fits))

    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
