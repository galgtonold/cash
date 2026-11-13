"""Where a text file stands cannot be read while it is iterated (``next(f)``
refuses ``tell()`` until the end): the divergence probe then tells the file
by its object. The same object the cell read before has moved; another one
was made since by the cell above (a Run All) and has not."""

from __future__ import annotations

import io

from cash.notebook.consumables import UntoldPosition, consumable_state, has_diverged


def _data(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("h1,h2\n1,2\n3,4\n", encoding="utf-8")
    return path


def test_a_file_object_made_again_has_not_moved(tmp_path):
    path = _data(tmp_path)
    with open(path, encoding="utf-8") as first, open(path, encoding="utf-8") as second:
        next(first)
        baseline = consumable_state(first)
        assert isinstance(baseline, UntoldPosition)
        assert has_diverged(first, baseline, had_baseline=True)
        next(second)
        assert not has_diverged(second, baseline, had_baseline=True)


def test_a_file_read_by_next_since_a_known_place_has_moved(tmp_path):
    with open(_data(tmp_path), encoding="utf-8") as f:
        baseline = consumable_state(f)
        assert baseline == ("tell", 0, False)
        next(f)
        assert has_diverged(f, baseline, had_baseline=True)


def test_a_buffer_refusing_tell_is_not_taken_for_a_file():
    class Untellable(io.StringIO):
        def tell(self):
            raise OSError("no position")

    assert consumable_state(Untellable("a\n")) is None
