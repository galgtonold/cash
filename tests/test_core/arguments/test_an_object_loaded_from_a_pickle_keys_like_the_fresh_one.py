"""An object loaded from a pickle keys like the equal object it was saved from.

A library object is pickled whole into its key. Through pickle's memo, an
object met twice was written as a back-reference by identity: a fitted
model's parameter names were the very strings keying its ``__dict__`` and
its arrays shared numpy's one ``float64`` dtype, while the model loaded from
the cache held copies. And a tree's node records keyed their padding bytes,
which hold whatever the memory held. So ``score(train(), X)`` missed once in
every new process: the model ``train()`` returned from the cache keyed
apart from the one it had fitted. Sharing that means something -- one list
held twice -- still keys apart from two equal lists.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pytest

from cash import Cash, FileBackend


@pytest.fixture
def key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    return lambda *args: c._args.hash_payload(args, {})


def _reloaded(value):
    return pickle.loads(pickle.dumps(value))


def test_strings_and_dtypes_shared_by_a_fresh_object_key_alike_after_a_reload(key):
    names = ("alpha", "beta")
    fresh = argparse.Namespace(names=names, **{n: np.arange(3.0) for n in names}, kinds=[n for n in names])
    fresh.dtypes = [fresh.alpha.dtype, fresh.beta.dtype]
    reloaded = _reloaded(fresh)
    assert fresh.alpha.dtype is fresh.beta.dtype and reloaded.alpha.dtype is not np.dtype("float64")
    assert key(fresh) == key(reloaded)


def test_a_record_arrays_padding_is_not_keyed(key):
    padded = np.dtype({"names": ["x", "flag"], "formats": ["f8", "u1"], "offsets": [0, 8], "itemsize": 16})
    raw = np.frombuffer(np.random.default_rng(0).bytes(16 * 4), dtype=padded).copy()
    raw["x"], raw["flag"] = np.arange(4.0), 1
    clean = np.zeros(4, dtype=padded)
    clean["x"], clean["flag"] = np.arange(4.0), 1
    assert raw.tobytes() != clean.tobytes()
    assert key(raw) == key(clean)
    assert key(argparse.Namespace(nodes=raw)) == key(argparse.Namespace(nodes=clean))
    changed = clean.copy()
    changed["flag"][0] = 0
    assert key(argparse.Namespace(nodes=changed)) != key(argparse.Namespace(nodes=clean))


def test_one_list_held_twice_still_keys_apart_from_two(key):
    row = [0]
    aliased = argparse.Namespace(a=row, b=row)
    separate = argparse.Namespace(a=[0], b=[0])
    assert key(aliased) != key(separate)
    assert key(aliased) == key(_reloaded(aliased))


def test_an_object_holding_itself_keys(key):
    ns = argparse.Namespace(items=[1, 2])
    ns.me = ns
    assert key(ns) == key(_reloaded(ns))


@pytest.mark.parametrize("kind", ["forest", "pipeline", "column_transformer"])
def test_a_fitted_model_keys_alike_after_a_reload(key, kind):
    pytest.importorskip("sklearn")
    pd = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, 3)), columns=list("abc"))
    X["k"] = rng.choice(["u", "v"], 200)
    y = (X.a > 0.5).astype(int)
    numbers = X[list("abc")]
    if kind == "forest":
        model = RandomForestClassifier(n_estimators=2, random_state=0).fit(numbers, y)
    elif kind == "pipeline":
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(numbers, y)
    else:
        encode = ColumnTransformer([("o", OneHotEncoder(), ["k"])], remainder="passthrough")
        model = make_pipeline(encode, LogisticRegression()).fit(X, y)
    assert key(model) == key(_reloaded(model))
