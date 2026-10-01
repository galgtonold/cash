"""Methods called on an object the function itself made are not side effects.

``h = hashlib.sha256(); h.update(b)`` was reported as a write, ``m =
LinearRegression(); m.fit(X, y)`` and ``r = random.Random(k); r.shuffle(xs)``
as discarded calls: the object never leaves the function, exactly like
``out = []; out.append(x)``, which was already exempt. Users learned to put
``assume_safe`` everywhere.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import random
import zipfile
from collections import deque

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer

pytestmark = pytest.mark.core


def _issues(fn):
    return [(i.kind, i.description) for i in PurityAnalyzer().analyze(fn).issues]


class Client:
    def send_metric(self, x):
        return x


def digest(s):
    h = hashlib.sha256()
    h.update(s.encode())
    return h.hexdigest()


def shuffled(k):
    r = random.Random(k)
    xs = list(range(5))
    r.shuffle(xs)
    return xs


def counted(xs):
    c = collections.Counter()
    c.update(xs)
    return c


def rest(xs):
    d = deque(xs)
    d.popleft()
    return list(d)


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--x")
    return p.parse_args(argv)


@pytest.mark.parametrize("fn", [digest, shuffled, counted, rest, parse], ids=lambda f: f.__name__)
def test_work_on_its_own_object_is_not_reported(fn):
    assert _issues(fn) == []


def test_fitting_a_new_estimator_is_not_reported():
    np = pytest.importorskip("numpy")
    linear_model = pytest.importorskip("sklearn.linear_model")

    def fit(k):
        m = linear_model.LinearRegression()
        m.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
        return m.coef_.tolist()

    assert _issues(fit) == []


def fit_the_callers(model, xs):
    model.fit(xs)
    return model.mean


def emit(x):
    client = Client()
    client.send_metric(x)
    return x


def zip_up(path, member):
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(member)
    return path


@pytest.mark.parametrize("fn", [fit_the_callers, emit, zip_up], ids=lambda f: f.__name__)
def test_the_same_shapes_on_anothers_object_or_with_an_effect_still_are(fn):
    """Controls: the caller's model, a vendor call on a new client, a write."""
    assert _issues(fn), fn.__name__
