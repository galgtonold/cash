"""Editing a list or dict inside a returned frame never changes the next hit.

The RAM tier copied a frame shallowly (under copy-on-write that already
isolates the frame's own data), but neither a shallow nor a deep pandas copy
copies the Python objects in an object column. A list in a cell was one
object shared by the caller, the entry and every later hit:
``load()["tags"].iloc[0].append("X")`` and the next ``load()`` returned the
appended list. Frames holding such cells are now copied with their cells.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash import Cash

SHAPES = {
    "frame": lambda r: r,
    "series": lambda r: r["tags"],
    "frame in a dict": lambda r: {"df": r},
    "frame in a list": lambda r: [r, 1],
    "frame in a tuple": lambda r: (r, 1),
}


def _tags(result, shape):
    if shape == "series":
        return result
    if shape == "frame in a dict":
        return result["df"]["tags"]
    if shape in ("frame in a list", "frame in a tuple"):
        return result[0]["tags"]
    return result["tags"]


@pytest.fixture
def app(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("edited", ["miss", "hit"])
def test_an_edited_cell_does_not_reach_later_hits(app, shape, edited):
    @app.cache
    def load():
        frame = pd.DataFrame({"tags": [["a"], ["b"]], "rec": [{"k": 1}, {"k": 2}], "n": [1.0, 2.0]})
        return SHAPES[shape](frame)

    first = load()
    target = first if edited == "miss" else load()
    _tags(target, shape).iloc[0].append("X")
    if shape != "series":
        frame = target if shape == "frame" else (target["df"] if shape == "frame in a dict" else target[0])
        frame["rec"].iloc[1]["k"] = 99

    again = load()
    assert _tags(again, shape).tolist() == [["a"], ["b"]]
    if shape != "series":
        frame = again if shape == "frame" else (again["df"] if shape == "frame in a dict" else again[0])
        assert frame["rec"].tolist() == [{"k": 1}, {"k": 2}]
    assert load.cache_info()["hits"] >= 1
