"""Find tests newly protected by the refined classifier.

Computes the OLD keep set from the backed-up inventory_v1.json (pre-refinement)
and the NEW keep set from integration_inventory.json (post-refinement), then
emits the difference: node-ids that are kept now but were redundant before.

Running ONLY these and combining with the existing keepers_all.cov gives the
NEW keep coverage without re-running the 895 tests already measured.

Writes:
    covrun/newly_protected.txt
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = 3


def node_id(r: dict) -> str:
    if r.get("class"):
        return f"{r['file']}::{r['class']}::{r['test']}"
    return f"{r['file']}::{r['test']}"


def keep_set(inv_path: Path) -> set[str]:
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    records = inv["records"]
    sig_counts = Counter(r["signature"] for r in records)
    seen: Counter = Counter()
    keep: set[str] = set()
    for r in sorted(records, key=lambda x: (x["signature"], x["file"], str(x["test"]))):
        sig = r["signature"]
        protect = (
            r["bucket"] != "generic"
            or r["flags"].get("asserts_status")
            or sig_counts[sig] <= K
        )
        if protect:
            keep.add(node_id(r))
            continue
        seen[sig] += 1
        if seen[sig] <= K:
            keep.add(node_id(r))
    return keep


def main() -> None:
    old = keep_set(ROOT / "covrun" / "inventory_v1.json")
    new = keep_set(ROOT / "integration_inventory.json")
    newly = sorted(new - old)
    dropped = sorted(old - new)
    (ROOT / "covrun" / "newly_protected.txt").write_text(
        "\n".join(newly), encoding="utf-8"
    )
    print(f"old_keep={len(old)}  new_keep={len(new)}")
    print(f"newly_protected={len(newly)}  (kept now, redundant before)")
    print(f"dropped_from_keep={len(dropped)}  (kept before, redundant now)")
    if dropped:
        print("  sample dropped:")
        for d in dropped[:10]:
            print(f"    {d}")


if __name__ == "__main__":
    main()
