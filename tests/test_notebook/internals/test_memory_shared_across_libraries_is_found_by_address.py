"""How `borrowed_memory_names` finds the variables whose memory a frame
shares across libraries: by the address ranges of their buffers, and only
for pairs one side can write in place (numpy or pandas with polars or Arrow,
numpy with pandas). A LazyFrame shows no buffers, so its statement's inputs
stand in (`lazy_frame_sources`).
"""

from __future__ import annotations

import pytest

from cash.notebook.borrowed_memory import borrowed_memory_names, lazy_frame_sources, memory_spans

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")


def test_the_partners_found_by_address():
    arr = np.arange(100.0)
    other = np.arange(100.0)
    frame = pd.DataFrame({"a": np.arange(100.0)})
    ns = {"arr": arr, "other": other, "frame": frame, "data": {"x": other}}
    assert borrowed_memory_names([pl.Series(arr)], ns) == {"arr"}
    assert borrowed_memory_names([[pa.array(other)]], ns) == {"other", "data"}
    # Containers are looked into only when named (a statement's inputs).
    assert borrowed_memory_names([pa.array(other)], ns, containers={"arr"}) == {"other"}
    assert borrowed_memory_names([pa.array(other)], ns, containers={"data"}) == {"other", "data"}
    assert borrowed_memory_names([pl.from_pandas(frame)], ns) == {"frame"}
    assert borrowed_memory_names([frame["a"].to_numpy()], ns) == {"frame"}
    assert borrowed_memory_names([pd.Series(arr, copy=False)], ns) == {"arr"}
    # Copies, a numpy view of a numpy array (`view_bases`'s), and pandas
    # over pandas (copy-on-write keeps them apart) link nothing here.
    assert borrowed_memory_names([pl.Series(arr * 2)], ns) == set()
    assert borrowed_memory_names([pd.Series(arr)], ns) == set()
    assert borrowed_memory_names([arr[:5]], ns) == set()
    assert borrowed_memory_names([frame[["a"]]], ns) == set()
    # Two polars frames over one array share memory nothing writes past.
    assert borrowed_memory_names([pl.Series(arr)], {"t": pl.Series(arr)}) == set()


def test_a_lazy_frame_takes_its_inputs_as_sources():
    arr = np.arange(10.0)
    ns = {"arr": arr, "n": 3, "d": pl.DataFrame({"a": arr})}
    lazy = pl.LazyFrame({"a": arr})
    assert lazy_frame_sources([lazy], {"arr", "n"}, ns) == {"arr"}
    assert lazy_frame_sources([ns["d"].lazy()], {"d"}, ns) == {"d"}
    assert lazy_frame_sources([ns["d"]], {"d"}, ns) == set()


def test_the_spans_cover_every_chunk_and_skip_what_has_no_buffer():
    arr = np.arange(10.0)
    chunked = pl.concat([pl.Series(arr), pl.Series(np.arange(4.0))], rechunk=False)
    assert chunked.n_chunks() == 2
    assert len(memory_spans(chunked)) == 2
    assert memory_spans(pl.Series(["a", "b"])) == []
    assert memory_spans(pd.DataFrame({"a": [1.0]}, index=pd.RangeIndex(1))) != []
    assert memory_spans("not a frame") == []
