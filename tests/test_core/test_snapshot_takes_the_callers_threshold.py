"""A snapshot uses the full-hash threshold its caller passes.

The threshold used to be resolved inside the helpers only, from whichever
config was reachable -- the running call's, else the global ``Cash``'s, else
one read from disk. A caller that knows its own ``file_hash_full_max_bytes``
can now say so, and the helpers resolve one only when it does not.
"""

from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_dependencies, snapshot_file_deps, snapshot_is_fresh


def test_a_passed_threshold_decides_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "full_hash_max_bytes", lambda: 1 << 30)
    data = tmp_path / "data.bin"
    data.write_bytes(b"x" * 4096)

    assert snapshot_file_deps({str(data)}, full_hash_max=16)[str(data)]["sampled"] is True
    assert snapshot_file_deps({str(data)})[str(data)]["sampled"] is False
    assert snapshot_dependencies([str(data)], full_hash_max=16)[str(data)]["sampled"] is True


def test_the_check_uses_the_passed_threshold_too(tmp_path, monkeypatch):
    data = tmp_path / "data.bin"
    data.write_bytes(b"x" * 4096)
    snap = snapshot_file_deps({str(data)}, full_hash_max=16)

    def unresolved():
        raise AssertionError("resolved the threshold although the caller passed one")

    monkeypatch.setattr(file_dep_snapshot, "full_hash_max_bytes", unresolved)
    fresh, stale = snapshot_is_fresh(snap, full_hash_max=16)
    assert fresh, stale
