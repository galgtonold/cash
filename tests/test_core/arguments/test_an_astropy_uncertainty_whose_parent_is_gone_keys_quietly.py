"""An astropy uncertainty whose parent ``NDData`` is gone keys, is checked
and is stored without astropy logging anything.

``NDUncertainty.__getstate__`` reads its ``parent_nddata`` property, which
logs "INFO: the associated NDData object was deleted ..." when the weak
reference to the parent is dead. specutils' arithmetic leaves such
uncertainties behind (it wraps an operand in a temporary ``NDDataArray``
that takes the uncertainty over, then drops it). cash pickles an argument
for its key and again for the mutation check, and a result for the store:
each printed that line where the plain call printed nothing. The key bytes
are the ones pickle's own reduce gives, so entries made before still hit.
"""

from __future__ import annotations

import logging
import pickle

import pytest

from cash import Cash, FileBackend, _plain_data, kept_state

np = pytest.importorskip("numpy")
nddata = pytest.importorskip("astropy.nddata")
astropy_log = pytest.importorskip("astropy").log


def _orphan():
    """A ``StdDevUncertainty`` whose parent was deleted."""
    u = nddata.StdDevUncertainty(np.array([1.0, 2.0, 3.0]))
    parent = nddata.NDData(np.array([4.0, 5.0, 6.0]), uncertainty=u)
    assert u.parent_nddata is parent
    del parent
    return u


def _total(u):
    return float(u.array.sum())


def _scaled(u):
    return nddata.StdDevUncertainty(u.array * 2.0)


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def astropy_records():
    handler = _Records()
    astropy_log.addHandler(handler)
    try:
        yield handler.messages
    finally:
        astropy_log.removeHandler(handler)


def _cache(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def _the_old_reduce(monkeypatch):
    """cash reducing the uncertainty through its ``__reduce_ex__``, as
    pickle does (and as cash did before)."""
    monkeypatch.setattr(kept_state, "_quiet_uncertainty", lambda t: False)


def test_a_call_prints_what_the_plain_call_prints(tmp_path, capsys, astropy_records):
    u = _orphan()
    plain = _total(u)
    plain_out = capsys.readouterr()
    assert astropy_records == []

    f = _cache(tmp_path).cache(_total)
    assert f(u) == plain
    assert f(u) == plain
    assert capsys.readouterr() == plain_out
    assert astropy_records == []
    info = f.cache_info()
    assert (info["hits"], info["misses"]) == (1, 1)


def test_a_result_is_stored_and_loaded_quietly(tmp_path, capsys, astropy_records):
    u = _orphan()
    plain = _scaled(u)
    plain_out = capsys.readouterr()

    f = _cache(tmp_path).cache(_scaled)
    first = f(u)
    second = _cache(tmp_path).cache(_scaled)(u)
    assert capsys.readouterr() == plain_out
    assert astropy_records == []
    for got in (first, second):
        assert type(got) is type(plain)
        np.testing.assert_array_equal(got.array, plain.array)


def test_the_key_and_the_store_bytes_are_pickle_s(monkeypatch, astropy_records):
    u = _orphan()
    quiet_key = _plain_data.key_dumps(u)
    quiet_stored = kept_state.dumps(u, 5)
    assert astropy_records == []

    with monkeypatch.context() as m:
        _the_old_reduce(m)
        assert _plain_data.key_dumps(u) == quiet_key
        assert kept_state.dumps(u, 5) == quiet_stored
    assert astropy_records  # the old reduce read the logging property
    assert quiet_stored == pickle.dumps(u, protocol=5)


def test_an_uncertainty_with_a_live_parent_keys_as_before(monkeypatch, astropy_records):
    u = nddata.StdDevUncertainty(np.array([1.0, 2.0]))
    parent = nddata.NDData(np.array([3.0, 4.0]), uncertainty=u)
    alone = nddata.StdDevUncertainty(np.array([1.0, 2.0]))
    keys = [_plain_data.key_dumps(u), _plain_data.key_dumps(alone)]
    with monkeypatch.context() as m:
        _the_old_reduce(m)
        assert [_plain_data.key_dumps(u), _plain_data.key_dumps(alone)] == keys
    assert keys[0] != keys[1]  # the parent is part of the key
    assert astropy_records == []
    assert u.parent_nddata is parent


def test_an_entry_made_before_still_hits(tmp_path, monkeypatch, astropy_records):
    u = _orphan()
    with monkeypatch.context() as m:
        _the_old_reduce(m)
        before = _cache(tmp_path).cache(_total)
        assert before(u) == 6.0
    astropy_records.clear()

    after = _cache(tmp_path).cache(_total)
    assert after(u) == 6.0
    assert after.cache_info()["hits"] == 1
    assert astropy_records == []
