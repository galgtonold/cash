"""Run a list of pytest node-ids in batches (avoids OS arg-length limits).

Coverage parallel-mode data accumulates in covrun/ across every batch (each
kernel subprocess writes its own .coverage.<host>.<pid> file). Combine once
after all batches finish.

Usage:
    python covtools/cov_run_batches.py covrun/redundant_sample.txt [batch_size]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    list_file = Path(sys.argv[1])
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    node_ids = [
        ln.strip()
        for ln in list_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    batches = [node_ids[i:i + batch_size] for i in range(0, len(node_ids), batch_size)]
    print(f"{len(node_ids)} node-ids in {len(batches)} batches of <= {batch_size}")

    t0 = time.time()
    for n, batch in enumerate(batches, 1):
        cmd = [
            sys.executable, "-m", "pytest", *batch,
            "-p", "no:cacheprovider", "-q", "--continue-on-collection-errors",
            "--no-header",
        ]
        r = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True
        )
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        print(f"  batch {n}/{len(batches)} ({len(batch)} tests): {tail}")
    print(f"\nall batches done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
