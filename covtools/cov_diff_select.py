"""Select keeper vs redundant pytest node-ids for a coverage-diff experiment.

Replicates the exact keep/redundant split from analyze_integration_tests.py
(sorted by signature,file,test; first K=3 of each redundant-eligible signature
are keepers, the rest redundant) and emits two node-id lists for one signature.

Usage:
    python covtools/cov_diff_select.py "none|generic"
"""
from __future__ import annotations

import json
import sys
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
    target_sig = sys.argv[1] if len(sys.argv) > 1 else "none|generic"
    inv = json.loads(INV.read_text(encoding="utf-8"))
    records = inv["records"]
    sig_counts = Counter(r["signature"] for r in records)

    seen: Counter = Counter()
    keepers: list[dict] = []
    redundant: list[dict] = []
    for r in sorted(records, key=lambda x: (x["signature"], x["file"], str(x["test"]))):
        sig = r["signature"]
        protect = (
            r["bucket"] != "generic"
            or r["flags"].get("asserts_status")
            or sig_counts[sig] <= K
        )
        if protect:
            continue
        seen[sig] += 1
        if seen[sig] <= K:
            keepers.append(r)
        else:
            redundant.append(r)

    keep_sig = [r for r in keepers if r["signature"] == target_sig]
    red_sig = [r for r in redundant if r["signature"] == target_sig]

    # Redundant sample: one test per distinct file (max snippet diversity).
    by_file: dict[str, dict] = {}
    for r in red_sig:
        by_file.setdefault(r["file"], r)
    red_sample = list(by_file.values())

    (ROOT / "covrun" / "keepers.txt").write_text(
        "\n".join(node_id(r) for r in keep_sig), encoding="utf-8"
    )
    (ROOT / "covrun" / "redundant_sample.txt").write_text(
        "\n".join(node_id(r) for r in red_sample), encoding="utf-8"
    )

    print(f"signature: {target_sig}")
    print(f"  total in signature: {sig_counts[target_sig]}")
    print(f"  keepers (K={K}): {len(keep_sig)}")
    print(f"  redundant total: {len(red_sig)}")
    print(f"  redundant sample (1/file): {len(red_sample)} across "
          f"{len(by_file)} files")
    print("\nKEEPERS:")
    for r in keep_sig:
        print("  ", node_id(r))


if __name__ == "__main__":
    main()
