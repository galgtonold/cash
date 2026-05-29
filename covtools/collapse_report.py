"""Report the keep/drop collapse decision per file (non-destructive).

Reads integration_inventory.json + the keep/redundant node-id lists, then
summarizes how many kernel-spinning tests each file would shed under the
K=3-per-signature policy. Pass a file substring to see the per-test verdict.

Usage:
    python covtools/collapse_report.py                # global summary
    python covtools/collapse_report.py stress_batch1  # per-test for matches
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEC_PER_TEST = 3.97  # measured: 355 redundant tests / 1409s in redundant_run.log


def nid(r: dict) -> str:
    if r.get("class"):
        return f"{r['file']}::{r['class']}::{r['test']}"
    return f"{r['file']}::{r['test']}"


def main() -> None:
    inv = json.loads((ROOT / "integration_inventory.json").read_text(encoding="utf-8"))
    recs = inv["records"]
    keep = set((ROOT / "covrun" / "keepers_all.txt").read_text(encoding="utf-8").split())
    red = set((ROOT / "covrun" / "redundant_all.txt").read_text(encoding="utf-8").split())

    needle = sys.argv[1] if len(sys.argv) > 1 else None

    if needle:
        sel = [r for r in recs if needle in r["file"]]
        for f in sorted({r["file"] for r in sel}):
            fr = [r for r in sel if r["file"] == f]
            print(f"\n{f}  ({len(fr)} tests)")
            for r in sorted(fr, key=lambda x: str(x["test"])):
                verdict = "KEEP" if nid(r) in keep else ("DROP" if nid(r) in red else "????")
                print(f"  [{verdict}] {str(r['test']):48s} {r['signature']}")
        return

    total = len(recs)
    n_keep = sum(1 for r in recs if nid(r) in keep)
    n_drop = sum(1 for r in recs if nid(r) in red)
    saved_h = n_drop * SEC_PER_TEST / 3600
    print(f"tests total = {total}")
    print(f"  KEEP = {n_keep}  ({n_keep/total:.0%})")
    print(f"  DROP = {n_drop}  ({n_drop/total:.0%})")
    print(f"  projected wall-clock saved ~= {saved_h:.1f}h "
          f"(at {SEC_PER_TEST:.1f}s/test, single-threaded)")

    # Files that shed the most tests.
    by_file_drop = Counter(r["file"] for r in recs if nid(r) in red)
    print("\nTop 15 files by tests dropped:")
    for f, n in by_file_drop.most_common(15):
        total_f = sum(1 for r in recs if r["file"] == f)
        print(f"  {n:4d}/{total_f:<4d}  {f.split('/')[-1]}")


if __name__ == "__main__":
    main()
