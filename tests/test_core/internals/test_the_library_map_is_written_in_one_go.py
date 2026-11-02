"""The saved library map and file digests are encoded in one go.

``json.dump`` to a file encodes through ``json``'s pure-Python encoder,
chunk by chunk: the shared map file holds a map for each of 16
environments and search paths, ~1 MB, and rewriting it took ~60 ms where
``json.dumps`` takes ~14 ms. Every process saves it again after a file
appears in, or an editor's save renames into, the script's folder: each
of p25's four joblib workers did so after a code edit. What must hold:
the same file, written without the pure-Python encoder.
"""

from __future__ import annotations

import json

import pytest

from cash.decorator import library_versions


@pytest.fixture
def no_pure_python_encoder(monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("encoded chunk by chunk")

    monkeypatch.setattr(json.encoder, "_make_iterencode", refuse)


def test_the_map_is_saved_and_reads_back(tmp_path, monkeypatch, no_pure_python_encoder):
    path = tmp_path / "_library_map.json"
    monkeypatch.setattr(library_versions, "_map_path", lambda: str(path))
    header = ["v", "tag", "prefix", "base", "exe", [str(tmp_path)]]
    fingerprint = {"header": header, "paths": [["bare", 1, 2, 3, 4]]}
    by_top = {"numpy": [["numpy", None]], "pkg": [["pkg-dist", str(tmp_path / "pkg.dist-info")]]}

    library_versions._save_map(fingerprint, by_top)

    saved = json.loads(path.read_text(encoding="utf-8"))
    entry = saved[library_versions._slot(header)]
    assert entry["map"] == by_top
    assert entry["fingerprint"] == fingerprint
    assert entry["digest"] == library_versions._map_digest(by_top)


def test_the_file_digests_are_saved_and_read_back(tmp_path, monkeypatch, no_pure_python_encoder):
    path = tmp_path / "_library_records.json"
    monkeypatch.setattr(library_versions, "_records_path", lambda: str(path))
    saved = {"/site/pkg.dist-info/RECORD": [[1, 2, 3, 4], "abcd"], "": True}

    library_versions._save_records(saved)

    assert library_versions._read_records() == {"/site/pkg.dist-info/RECORD": [[1, 2, 3, 4], "abcd"]}
