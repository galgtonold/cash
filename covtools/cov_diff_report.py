"""Compare two coverage data files: what does B cover that A does not?

For the integration-test redundancy proof:
    A = keepers.cov     (the 3 representatives we keep for a signature)
    B = redundant.cov   (a 1-per-file sample of the tests we'd drop)

If B - A is empty (no line, no branch-arc), the dropped tests exercise no
cash code path the keepers don't already cover => safe to drop.

Usage:
    python covtools/cov_diff_report.py covrun/keepers.cov covrun/redundant.cov
"""
from __future__ import annotations

import sys
from pathlib import Path

from coverage import CoverageData


def load(path: str) -> CoverageData:
    d = CoverageData(basename=path)
    d.read()
    return d


def short(f: str) -> str:
    # Trim to the cash package path for readability.
    p = f.replace("\\", "/")
    i = p.find("/src/cash/")
    return p[i + 5:] if i >= 0 else p


def main() -> None:
    a_path = sys.argv[1] if len(sys.argv) > 1 else "covrun/keepers.cov"
    b_path = sys.argv[2] if len(sys.argv) > 2 else "covrun/redundant.cov"
    a, b = load(a_path), load(b_path)

    a_files = {f for f in a.measured_files() if "cash" in f.replace("\\", "/")}
    b_files = {f for f in b.measured_files() if "cash" in f.replace("\\", "/")}

    all_files = sorted(a_files | b_files, key=short)

    total_extra_lines = 0
    total_extra_arcs = 0
    extra_by_file: list[tuple[str, list[int], list[tuple[int, int]]]] = []
    files_only_in_b: list[str] = []

    for f in all_files:
        a_lines = set(a.lines(f) or [])
        b_lines = set(b.lines(f) or [])
        a_arcs = set(a.arcs(f) or [])
        b_arcs = set(b.arcs(f) or [])

        extra_lines = sorted(b_lines - a_lines)
        # arcs can include negative sentinels for entry/exit; keep all.
        extra_arcs = sorted(b_arcs - a_arcs)

        if f not in a_files and (b_lines or b_arcs):
            files_only_in_b.append(f)

        if extra_lines or extra_arcs:
            total_extra_lines += len(extra_lines)
            total_extra_arcs += len(extra_arcs)
            extra_by_file.append((f, extra_lines, extra_arcs))

    print(f"A (keepers):   {a_path}")
    print(f"B (redundant): {b_path}")
    print(f"cash files measured: A={len(a_files)}  B={len(b_files)}")
    print()
    print(f"=== B - A  (lines/branches the dropped tests hit but keepers DON'T) ===")
    print(f"extra lines:  {total_extra_lines}")
    print(f"extra arcs:   {total_extra_arcs}")
    if files_only_in_b:
        print(f"\ncash files touched ONLY by redundant set ({len(files_only_in_b)}):")
        for f in files_only_in_b:
            print("  ", short(f))
    if extra_by_file:
        print("\nper-file extra coverage (B not in A):")
        for f, lines, arcs in extra_by_file:
            tag = "  [NEW FILE]" if f in files_only_in_b else ""
            print(f"\n  {short(f)}{tag}")
            if lines:
                print(f"    +lines ({len(lines)}): {lines[:60]}"
                      + (" ..." if len(lines) > 60 else ""))
            if arcs:
                print(f"    +arcs  ({len(arcs)}): {arcs[:40]}"
                      + (" ..." if len(arcs) > 40 else ""))
    else:
        print("\n>>> EMPTY: the dropped tests add NO line and NO branch beyond keepers.")

    # Reverse direction, for context (keepers may cover more, e.g. FileBackend).
    rev_lines = 0
    for f in all_files:
        rev_lines += len(set(a.lines(f) or []) - set(b.lines(f) or []))
    print(f"\n(for context) A - B extra lines (keepers cover, sample doesn't): {rev_lines}")


if __name__ == "__main__":
    main()
