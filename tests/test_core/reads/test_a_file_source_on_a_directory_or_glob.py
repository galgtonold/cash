"""A `FileDataSource` on a directory or a glob pattern tracks the files in it.

Its token was the directory's mtime -- which an edit of a file inside does
not move -- or ``"absent"`` for a pattern, which names no file: rewriting a
file of a dataset folder served the old result.
"""

from __future__ import annotations

import pytest

from cash import FileDataSource
from tests._files import rewrite


@pytest.mark.parametrize("target", ["dataset", "dataset/*.txt"])
@pytest.mark.parametrize("how", ["depends_on", "dynamic_depends_on"])
def test_an_edit_inside_recomputes(cash_instance, tmp_path, monkeypatch, target, how):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dataset").mkdir()
    part = tmp_path / "dataset" / "part.txt"
    part.write_text("v1", encoding="utf-8")
    runs: list = []
    source = FileDataSource(target)
    kwargs = {"depends_on": [source]} if how == "depends_on" else {"dynamic_depends_on": lambda: source}

    @cash_instance.cache(assume_safe=True, **kwargs)
    def load():
        runs.append(1)  # reads the folder in a way cash does not see
        return len(runs)

    load()
    load()
    assert len(runs) == 1
    rewrite(part, "v2")
    load()
    assert len(runs) == 2
    (tmp_path / "dataset" / "more.txt").write_text("new", encoding="utf-8")
    load()
    assert len(runs) == 3
    load()
    assert len(runs) == 3
