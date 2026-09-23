"""A lineage tag on a CLASS must not become every instance's identity.

The notebook's lineage store tags a value it records with
``_cash_lineage_hash``, and the decorator trusts that tag as the value's
content when it keys a call. ``from pathlib import Path`` in a cached cell
records ``Path`` -- the class -- and the tag landed on the class, so every
``Path`` instance inherited it. From then on every path argument to a
``@cash.cache`` function hashed the same, and a call on one file was served
another file's result: in the unit suite, ``test_explain_file_freshness``
got the previous test's entry whenever it shared a worker with
``test_file_tracking.py``'s ``test_pathlib_tracking`` (found while chasing that
"flaky" test). A result for a DIFFERENT file with DIFFERENT content
came back without the function running.
"""

from pathlib import Path

from cash import Cash
from cash.notebook.lineage_store import LineageStore


def test_recording_a_class_does_not_tag_it():
    store = LineageStore()
    try:
        store.record("Path", "a" * 64, value=Path)
        assert "_cash_lineage_hash" not in vars(Path)
        assert not hasattr(Path("x"), "_cash_lineage_hash")
    finally:
        # A failure here must not leave pathlib tagged for the whole process.
        for attr in ("_cash_lineage_hash", "_cash_lineage_src"):
            if attr in vars(Path):
                delattr(Path, attr)


def test_a_tag_on_a_class_is_not_an_instances_identity(tmp_path, monkeypatch):
    """Even if something tags a class, its instances keep their own identity."""
    monkeypatch.setattr(Path, "_cash_lineage_hash", "b" * 64, raising=False)
    monkeypatch.setattr(Path, "_cash_lineage_src", "statement", raising=False)
    c = Cash(cache_dir=str(tmp_path / "cache"))
    calls = []

    @c.cache
    def read(p):
        calls.append(p)
        return p.read_text(encoding="utf-8")

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("AAA", encoding="utf-8")
    b.write_text("BBB", encoding="utf-8")
    assert read(a) == "AAA"
    assert read(b) == "BBB", "a call on b.txt was served a.txt's result"
    assert len(calls) == 2
