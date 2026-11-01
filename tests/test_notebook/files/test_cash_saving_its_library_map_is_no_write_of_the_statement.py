"""Cash saving its map of installed libraries is no write of the statement
that made it look.

The first statement to call into a library made cash save which installed
distribution provides each module (``_library_map.json``) and a digest of
each one's files (``_library_records.json``), in the per-user cache folder.
The write observer took those for the statement's own writes, which a hit
would skip, and refused it (p27's ``load`` and ``design``).
"""

from __future__ import annotations

from cash.decorator import library_versions
from cash.notebook.write_observer import kept_writes, observe_writes


def test_neither_saved_file_is_a_write_of_the_block(tmp_path, monkeypatch):
    monkeypatch.setattr(library_versions, "per_user_cache_root", lambda: tmp_path)
    with observe_writes() as written:
        library_versions._save_map({"header": ["h"]}, {"pkg": [["dist", None]]})
        library_versions._save_records({"dist": "digest"})
    assert (tmp_path / "_library_map.json").exists()
    assert (tmp_path / "_library_records.json").exists()
    assert kept_writes(written) == []


def test_a_file_the_block_writes_itself_is_still_seen(tmp_path):
    target = tmp_path / "out.json"
    with observe_writes() as written:
        target.write_text("{}", encoding="utf-8")
    assert kept_writes(written) == [str(target)]
