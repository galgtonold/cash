"""Fit linear cost models from the measurement matrix CSV.

For each (family, backend_kind, operation) group, fits
    predicted_seconds = a + b x size_bytes
and reports a, b, R2 and the relative error. Output is a Python dict literal of
{(family, backend, op): (a, b)} that the production module pastes in.

Two objectives, because they answer different questions:

``--objective absolute`` (ordinary least squares) minimises the error in
SECONDS. Across a matrix spanning 284 B to 111 MiB the 100 MiB points dominate
the sum, so the intercept absorbs their residual and the small end is priced
7-25x too high. That is how the shipped constants were fitted, and it was
harmless only while a 100 ms compute floor gated every decision below the range
where the error lives.

``--objective relative`` (the default) minimises the error as a RATIO, by
weighting each point by 1/y^2. Every decade of size then counts the same, which
is what a promotion decision needs: "is restoring this cheaper than recomputing
it" is a comparison of magnitudes, and being 20x wrong about 0.5 ms matters
exactly as much as being 20x wrong about 5 s.
"""

from __future__ import annotations

import argparse
import csv
import statistics
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
    worst_ratio: float = 1.0


def fit_relative(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit y = a + b*x minimising RELATIVE error. Returns (a, b, r_squared).

    Weighted least squares with w = 1/y^2, which is what minimising
    sum(((a + b*x - y) / y)^2) reduces to. The reported R2 is the ordinary one,
    so it stays comparable with the absolute fit's.

    A negative intercept is clamped to the smallest measured y and b refitted
    through it: the line is used to predict costs, and a prediction below zero
    at small sizes would read as "restoring is free".
    """
    if len(xs) < 2:
        return 0.0, 0.0, 0.0
    w = [1.0 / (y * y) if y > 0 else 0.0 for y in ys]
    sw = sum(w)
    swx = sum(wi * x for wi, x in zip(w, xs))
    swy = sum(wi * y for wi, y in zip(w, ys))
    swxx = sum(wi * x * x for wi, x in zip(w, xs))
    swxy = sum(wi * x * y for wi, x, y in zip(w, xs, ys))
    denom = sw * swxx - swx * swx
    if denom == 0 or sw == 0:
        return fit_linear(xs, ys)
    b = (sw * swxy - swx * swy) / denom
    a = (swy - b * swx) / sw
    if a < 0:
        a = min(ys)
        num = sum(wi * x * (y - a) for wi, x, y in zip(w, xs, ys))
        den = sum(wi * x * x for wi, x in zip(w, xs))
        b = num / den if den else 0.0
    if b < 0:
        # Microsecond-scale families (bytes in RAM) measure as noise around
        # a constant, and a negative slope predicts that a 100 MB object
        # costs less to handle than a small one. Flat is the honest reading.
        b = 0.0
        a = statistics.median(ys)
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def worst_ratio(a: float, b: float, xs: list[float], ys: list[float]) -> float:
    """The largest factor by which (a, b) misprices any point, either way."""
    worst = 1.0
    for x, y in zip(xs, ys):
        if y <= 0:
            continue
        pred = a + b * x
        ratio = pred / y if pred > y else (y / pred if pred > 0 else float("inf"))
        worst = max(worst, ratio)
    return worst


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


def fit_all(csv_path: Path, objective: str = "relative") -> list[FitRow]:
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
        a, b, r2 = fit_relative(xs, ys) if objective == "relative" else fit_linear(xs, ys)
        fits.append(
            FitRow(
                family=family,
                backend_kind=backend,
                operation=op,
                a=a,
                b=b,
                r_squared=r2,
                n_points=len(pts),
                worst_ratio=worst_ratio(a, b, xs, ys),
            )
        )
    return fits


#: Derived (NOT measured) backends, exactly as cost_model.py documents them:
#: redis is disk with most of the I/O removed and a LAN round-trip added, s3
#: likewise with request setup. Generated here so they cannot drift from the
#: disk fit they are derived from.
_DERIVED = {
    "redis": {"a_scale": 0.1, "a_add": 500e-6, "b_floor": 20e-9},
    "s3": {"a_scale": 0.2, "a_add": 80e-3, "b_floor": 50e-9},
}


def _derive(fits: list[FitRow]) -> list[FitRow]:
    """redis/s3 rows derived from each family's disk fit."""
    out: list[FitRow] = []
    for f in fits:
        if f.backend_kind != "disk":
            continue
        for backend, rule in _DERIVED.items():
            out.append(
                FitRow(
                    family=f.family,
                    backend_kind=backend,
                    operation=f.operation,
                    a=f.a * rule["a_scale"] + rule["a_add"],
                    b=max(f.b, rule["b_floor"]),
                    r_squared=float("nan"),
                    n_points=0,
                    worst_ratio=float("nan"),
                )
            )
    return out


def _generic(fits: list[FitRow]) -> list[FitRow]:
    """_GENERIC: the slowest family per (backend, op).

    Slowest is decided by the predicted cost at 10 MB rather than by slope
    alone. The fallback is what an unknown type gets priced with, and being
    wrong there should cost a value its place on disk, not hand it one it has
    not earned.
    """
    at = 10_000_000
    slowest: dict[tuple[str, str], FitRow] = {}
    for f in fits:
        prev = slowest.get((f.backend_kind, f.operation))
        if prev is None or f.a + f.b * at > prev.a + prev.b * at:
            slowest[(f.backend_kind, f.operation)] = f
    return [
        FitRow(
            family="_GENERIC",
            backend_kind=backend,
            operation=op,
            a=f.a,
            b=f.b,
            r_squared=float("nan"),
            n_points=0,
            worst_ratio=float("nan"),
        )
        for (backend, op), f in slowest.items()
    ]


def render_python_constants(fits: list[FitRow], derived: bool = True) -> str:
    """Render a Python dict literal for paste into cost_model.py."""
    rows = list(fits)
    if derived:
        rows += _derive(fits)
        rows += _generic(rows)
    lines = ["{"]
    for f in sorted(rows, key=lambda x: (x.family, x.backend_kind, x.operation)):
        note = (
            "  # derived, not measured"
            if not f.n_points
            else f"  # R2={f.r_squared:.3f} n={f.n_points} worst={f.worst_ratio:.1f}x"
        )
        lines.append(f'    ("{f.family}", "{f.backend_kind}", "{f.operation}"): ({f.a:.6e}, {f.b:.6e}),{note}')
    lines.append("}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit cost model from matrix CSV")
    p.add_argument("csv_path", type=Path)
    p.add_argument("--min-r2", type=float, default=0.8, help="Warn if any fit has R² below this threshold")
    p.add_argument(
        "--objective",
        choices=("relative", "absolute"),
        default="relative",
        help="what the fit minimises: error as a ratio (default) or in seconds",
    )
    args = p.parse_args(argv)

    fits = fit_all(args.csv_path, args.objective)

    print("# Fit summary (sorted by R² ascending):")
    for f in sorted(fits, key=lambda x: x.r_squared):
        flag = "  [LOW R2]" if f.r_squared < args.min_r2 else ""
        print(
            f"  {f.family:18s} {f.backend_kind:4s} {f.operation:11s}  "
            f"a={f.a:.3e} b={f.b:.3e} R2={f.r_squared:.3f} n={f.n_points} "
            f"worst={f.worst_ratio:.1f}x{flag}"
        )

    bad = [f for f in fits if f.r_squared < args.min_r2]
    if bad:
        print(
            f"\n[WARN] {len(bad)} fits below R2={args.min_r2} threshold -- linear model "
            "may be wrong shape for these (family, backend, op) tuples. "
            "Investigate before pasting constants."
        )

    print("\n# COEFFS dict (paste into src/cash/notebook/cost_model.py):")
    print(render_python_constants(fits))

    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
