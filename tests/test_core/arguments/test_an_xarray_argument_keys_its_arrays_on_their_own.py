"""An xarray ``DataArray`` or ``Dataset`` argument keys its big arrays
through the content hasher, and its coordinates by their labels.

They keep their parts in slots, so the key walk pickled one whole: a 23 MB
array was copied into the key's pickle and hashed on one thread on every
hit. And a coordinate's pandas index pickled its values' state in another
order once restored from the cache than when built, so the cached call
after the one that returned the array missed on every run.
"""

from __future__ import annotations

import pickle

import pytest

from cash import Cash, FileBackend
from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT

xr = pytest.importorskip("xarray")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

#: Rows enough for the data to be keyed on its own (over a MiB).
_DAYS = 800


def _tas(sites=200, seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(280, 3, (sites, _DAYS)),
        dims=("site", "time"),
        coords={"time": pd.date_range("2000-01-01", periods=_DAYS), "site": [f"s{i}" for i in range(sites)]},
        attrs={"units": "K"},
        name="tas",
    )


def _key(value):
    return canonical_bytes(value, BUILTIN_CONTENT)


@pytest.mark.parametrize("sites", [2, 200])
def test_a_restored_array_keys_like_the_array_it_was_stored_from(tmp_path, sites):
    da = _tas(sites)
    assert _key(pickle.loads(pickle.dumps(da))) == _key(da)
    assert _key(pickle.loads(pickle.dumps(da.to_dataset()))) == _key(da.to_dataset())
    Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(_tas)(sites)
    restored = Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(_tas)(sites)
    assert restored is not da
    assert _key(restored) == _key(da)


@pytest.mark.parametrize("sites", [2, 200])
def test_reading_attrs_or_encoding_leaves_the_key(sites):
    da = _tas(sites)
    before = _key(da)
    _ = da.time.attrs, da.time.encoding, da.variable.encoding
    assert _key(da) == before


@pytest.mark.parametrize("sites", [2, 200])
@pytest.mark.parametrize("change", ["value", "attr", "name", "coordinate", "dims", "in_place", "encoding"])
def test_a_changed_array_keys_apart(sites, change):
    da = _tas(sites)
    other = da.copy(deep=True)
    if change == "value":
        other[1, 7] = other[1, 7] + 1e-9
    elif change == "attr":
        other.attrs["units"] = "degC"
    elif change == "name":
        other = other.rename("tasmax")
    elif change == "coordinate":
        other = other.assign_coords(site=[f"t{i}" for i in range(sites)])
    elif change == "dims":
        other = other.rename({"site": "station"})
    elif change == "in_place":
        before = _key(other)
        other.values[0, 0] += 1.0
        assert _key(other) != before
    elif change == "encoding":
        other.encoding["dtype"] = "float32"
    assert _key(other) != _key(da)
    if change != "in_place":
        assert _key(other.to_dataset()) != _key(da.to_dataset())


def test_a_cached_call_on_an_array_hits_and_misses_like_plain_python(tmp_path):
    calls = []

    def plain(da):
        return float(da.isel(time=slice(0, 5)).mean()), da.attrs.get("units"), str(da.name)

    def body(da):
        calls.append(1)
        return plain(da)

    f = Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(body)
    da = _tas()
    assert f(da) == plain(da)
    restored = pickle.loads(pickle.dumps(da))
    assert f(restored) == plain(da)
    assert calls == [1]
    da.values[0, 0] += 100.0
    assert f(da) == plain(da)
    da.attrs["units"] = "degC"
    assert f(da) == plain(da)
    assert calls == [1, 1, 1]
