"""A file written from native code is reported also when the frame is the
function's own.

``df = pl.Datapl.DataFrame(...); df.write_csv(p)`` got no warning at all: the static
check took a method on a frame the function built for a local change, and
polars writes from Rust, which the runtime observer (it sees ``open()``)
never sees. In a new process the call was a hit and the file not written.
"""

from __future__ import annotations

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer

pytestmark = pytest.mark.core


pl = pytest.importorskip("polars")


class Model:
    def save_model(self, path):
        return path


def export_csv(rows):
    df = pl.DataFrame({"a": rows})
    df.write_csv("out/pl.csv")
    return len(rows)


def export_parquet_chained(rows):
    pl.DataFrame({"a": rows}).write_parquet("out/pl.parquet")
    return len(rows)


def train_and_save(rows):
    m = Model()
    m.save_model("model.cbm")
    return len(rows)


def selected(rows):
    df = pl.DataFrame({"a": rows})
    return df.select("a").height


def _issues(fn):
    return [i.description for i in PurityAnalyzer().analyze(fn).issues]


@pytest.mark.parametrize("fn", [export_csv, export_parquet_chained, train_and_save])
def test_the_write_is_reported(fn):
    assert any(("write_" in d or "save_model" in d) for d in _issues(fn)), _issues(fn)


def test_other_work_on_the_frame_stays_quiet():
    assert _issues(selected) == []
