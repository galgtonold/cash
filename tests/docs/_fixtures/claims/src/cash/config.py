"""Synthetic stand-in for cash/config.py, used only by clean.md's fixture pins.

clean.md used to pin its fingerprint anchors against the REAL cash/config.py
and cash/core.py. That meant editing a 163-line method in a 4,583-line real
file (Cash.cache) turned this unit test red in the hard-gate docs-parity job,
while genuine drift on real doc pages stayed advisory -- backwards severity,
and the fix (hand-editing a hash into a fixture) was documented nowhere. This
tiny synthetic tree gives clean.md something stable to pin against; real-tree
resolution is already covered by the test_resolves_* tests in
test_claims_lib.py.
"""
from dataclasses import dataclass


@dataclass
class CashConfig:
    compress: bool = False
    max_cache_size: int | None = None
