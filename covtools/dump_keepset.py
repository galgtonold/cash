"""Dump ALL keep-set node-ids (across every signature) and ALL redundant ones.

Replicates analyze_integration_tests.py's keep/redundant split exactly, but
emits the FULL keep set (the ~895 tests we'd retain across all signatures) so
the coverage diff can use the correct baseline: does any dropped test cover a
cash line the *entire* keep set misses?

Writes:
    covrun/keepers_all.txt     (~895)
    covrun/redundant_all.txt   (~2520)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INV = ROOT / "integration_inventory.json"
K = 3


def node_id(r: dict) -> str:
    if r.get("class"):
        return f"{r['file']}::{r['class']}::{r['test']}"
    return f"{r['file']}::{r['test']}"


def main() -> None:
    inv = json.loads(INV.read_text(encoding="utf-8"))
    records = inv["records"]
    sig_counts = Counter(r["signature"] for r in records)

    seen: Counter = Counter()
    keep: list[dict] = []
    redundant: list[dict] = []
    for r in sorted(records, key=lambda x: (x["signature"], x["file"], str(x["test"]))):
        sig = r["signature"]
        protect = (
            r["bucket"] != "generic"
            or r["flags"].get("asserts_status")
            or sig_counts[sig] <= K
        )
        if protect:
            keep.append(r)
            continue
        seen[sig] += 1
        if seen[sig] <= K:
            keep.append(r)
        else:
            redundant.append(r)

    (ROOT / "covrun" / "keepers_all.txt").write_text(
        "\n".join(node_id(r) for r in keep), encoding="utf-8"
    )
    (ROOT / "covrun" / "redundant_all.txt").write_text(
        "\n".join(node_id(r) for r in redundant), encoding="utf-8"
    )
    print(f"keep={len(keep)}  redundant={len(redundant)}  total={len(records)}")


if __name__ == "__main__":
    main()
