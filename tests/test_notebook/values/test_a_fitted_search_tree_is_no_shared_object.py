"""A fitted nearest-neighbours model is no object another holds.

``KNeighborsRegressor`` builds a tree over its training data whose C-level
view of the array lists no referents: the array read as held from outside,
so every statement whose output held a fitted model -- the transformer
over it, a ``ColumnTransformer`` listing that, the pipeline and the search
over it -- searched every object the garbage collector tracks for the
holder (1.8 s of a 0.01 s cell on E6), ran out of its time budget and was
not stored. A copy of the model holds its copy of the array the same way,
so the reference is the model's own.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import share_group

np = pytest.importorskip("numpy")
neighbors = pytest.importorskip("sklearn.neighbors")


def _fitted(algorithm):
    data = np.random.default_rng(0).uniform(size=(40, 2))
    return neighbors.KNeighborsRegressor(3, algorithm=algorithm).fit(data, np.arange(40.0))


@pytest.mark.parametrize("algorithm", ["kd_tree", "ball_tree"])
def test_a_fitted_search_tree_is_no_holder_outside_the_model(algorithm):
    user_ns = {"holder": [_fitted(algorithm)]}

    _found, shared = share_group(["holder"], user_ns, user_ns)

    assert not shared, shared


def test_the_training_data_bound_to_a_variable_is_still_shared():
    """The control: ``fit`` keeps the array it was given, so a variable
    bound to that array holds an object of the model."""
    data = np.random.default_rng(0).uniform(size=(40, 2))
    user_ns = {"data": data, "holder": [neighbors.KNeighborsRegressor(3).fit(data, np.arange(40.0))]}
    del data

    _found, shared = share_group(["holder"], user_ns, user_ns)

    assert shared == {"holder"}, shared
