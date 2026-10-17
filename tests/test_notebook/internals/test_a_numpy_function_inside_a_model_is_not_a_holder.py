"""A numpy function inside a model does not stop the model being stored.

``FunctionTransformer(np.log)`` in a pipeline holds ``np.log``, and so do
numpy's modules and every module that imports it by name. A ufunc is
pickled by its name, so a restore hands back the very same object: those
modules are not holders a restore must keep, and the pipeline's fit is
stored like any other.
"""

from __future__ import annotations

import copy
import pickle

import pytest

from cash.notebook.shared_objects import GaveUp, share_group

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def test_a_ufunc_comes_back_as_itself():
    assert pickle.loads(pickle.dumps(np.log)) is np.log
    assert copy.deepcopy(np.log) is np.log


def test_a_holder_of_a_ufunc_shares_nothing():
    ns = {"pipe": make_pipeline(FunctionTransformer(np.log), StandardScaler())}
    gave_up = GaveUp()
    holders, shared = share_group(["pipe"], ns, ns, gave_up=gave_up)
    assert (holders, shared) == ({}, set())
    assert not gave_up.modules


def test_the_variables_beside_it_still_join():
    """The pipeline's step is bound to a variable too: it joins the group."""
    ns = {"log_step": FunctionTransformer(np.log)}
    ns["pipe"] = make_pipeline(ns["log_step"], StandardScaler())
    holders, shared = share_group(["pipe"], {"pipe": ns["pipe"]}, ns)
    assert shared == set()
    assert set(holders) == {"log_step"}
