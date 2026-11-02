"""A writer's check lists a crowded folder of reads once instead of resolving each file.

Resolving a path walks every part of it: the 24,000 documents a frame was
read from cost 276,000 ``lstat`` calls before each cell reading the frame
(13 s a cell on Windows, where resolving opens each file). The answer is
pinned in ``test_a_written_file_is_matched_to_its_reads_without_resolving_each``.
"""

from __future__ import annotations

import os

from cash.notebook.upstream.file_writers import any_resolves_to


def _crowd(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"{i}_7.txt"
        p.write_text(f"review {i}\n", encoding="utf-8")
        paths.append(str(p))
    return paths


def test_a_crowded_folder_of_plain_documents_is_not_resolved_file_by_file(tmp_path, monkeypatch):
    reads = _crowd(tmp_path / "pos", 200)
    targets = {os.path.realpath(tmp_path / "movie_data.csv")}
    resolved: list[str] = []
    real = os.path.realpath

    def counting(path, *args, **kwargs):
        resolved.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", counting)

    assert any_resolves_to(reads, targets) is False
    assert resolved == []
