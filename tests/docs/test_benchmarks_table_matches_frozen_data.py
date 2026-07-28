"""The restore-cost table on benchmarks.md must match the frozen matrix.

The page was rewritten to publish the half of the speedup equation cash
actually determines — restore cost — instead of a quoted "N x faster". That
only works if the printed numbers really are the measured ones, so this reads
both and compares.

Previously this page carried tester figures (~190x, ~4-5.5x, ~1.2x) that
matched nothing in the repo, and an archived result set whose runs had errored
partway. Numbers with no traceable source are what this test exists to stop
coming back.

`ser_deser_matrix.frozen.csv` is deliberately committed while the rest of
`benchmarks/results/` is gitignored, so this comparison works from a clean
checkout.
"""
from __future__ import annotations

import csv
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "benchmarks.md"
FROZEN = REPO / "benchmarks" / "results" / "ser_deser_matrix.frozen.csv"

# (column header in the page, family, backend) for each measured column.
_COLUMNS = [
    ("DataFrame (RAM)", "dataframe_numeric", "ram"),
    ("DataFrame (disk)", "dataframe_numeric", "disk"),
    ("ndarray (disk)", "ndarray_dense", "disk"),
    ("raw bytes (disk)", "bytes", "disk"),
]
_SIZES = {"1 KB": 1_000, "1 MB": 1_000_000,
          "10 MB": 10_000_000, "100 MB": 100_000_000}


@pytest.fixture(scope="module")
def measured() -> dict[tuple[str, str, int], float]:
    assert FROZEN.exists(), (
        f"{FROZEN} is missing. It is committed on purpose (the rest of "
        "benchmarks/results/ is gitignored) precisely so this check works."
    )
    out = {}
    with FROZEN.open() as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            out[(r["family"], r["backend_kind"], int(r["target_bytes"]))] = (
                float(r["deserialize_seconds"]) * 1000
            )
    assert out, "the frozen matrix parsed to nothing"
    return out


@pytest.fixture(scope="module")
def table_rows() -> dict[str, list[str]]:
    """The restore-cost table, parsed out of the page."""
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("**Deserialise time")
    body = text[start:text.index("\n\n", text.index("| 100 MB"))]
    rows = {}
    for line in body.splitlines():
        m = re.match(r"\|\s*(\d+ [KM]B)\s*\|(.+)\|", line)
        if m:
            rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    assert len(rows) == len(_SIZES), (
        f"expected {len(_SIZES)} size rows, parsed {sorted(rows)}. If the table "
        "was restructured, update this test with it."
    )
    return rows


def _quoted_ms(cell: str) -> float:
    m = re.match(r"([\d.]+)\s*ms", cell)
    assert m, f"cell {cell!r} is not a millisecond figure"
    return float(m.group(1))


def test_every_quoted_restore_cost_matches_the_frozen_matrix(measured, table_rows):
    problems = []
    for size_label, target in _SIZES.items():
        cells = table_rows[size_label]
        for (col_label, family, backend), cell in zip(_COLUMNS, cells):
            actual = measured.get((family, backend, target))
            if actual is None:
                problems.append(f"{size_label} / {col_label}: no frozen row")
                continue
            quoted = _quoted_ms(cell)
            # 5% or 1ms, whichever is looser — the page rounds for readability.
            if abs(actual - quoted) > max(1.0, quoted * 0.05):
                problems.append(
                    f"{size_label} / {col_label}: page says {quoted} ms, "
                    f"frozen matrix says {actual:.2f} ms"
                )
    assert not problems, "benchmarks.md disagrees with the measured data:\n  " + \
        "\n  ".join(problems)


def test_the_page_does_not_quote_an_unsourced_speedup():
    """The old page led with ~190x etc. Those matched nothing in the repo."""
    text = PAGE.read_text(encoding="utf-8")
    body = text[text.index("## What a restore costs"):]
    stray = re.findall(r"~\s*\d{2,}(?:[–-]\d+)?\s*×", body)
    assert not stray, (
        f"unsourced speedup figures reappeared below the restore-cost section: "
        f"{stray}. Speedups belong to the reader's workload; publish restore "
        "cost and let them divide."
    )
