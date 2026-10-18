"""A dict or list of a class of its own is sized by the arrays it holds as items.

scikit-learn's ``Bunch`` (``fetch_lfw_people``, ``load_digits``) is a dict
subclass that keeps its arrays as items. The size estimate read only an
object's attributes, so a Bunch of 150 MB of images came to 0: the RAM tier
copied its arrays one by one instead of keeping the pickle bytes, and the
statement that bound it paid both (0.7 s of a 2 s restore).
"""

from __future__ import annotations

import numpy as np

from cash import sizing
from cash.backends import InMemoryBackend, memory_backend

ROWS = 1 << 18


class Bunch(dict):
    """As scikit-learn's: items, also read as attributes."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


class Rows(list):
    """A list of a class of its own."""


def test_its_arrays_are_counted():
    images, target = np.ones((ROWS, 2)), np.arange(ROWS)
    assert sizing.pickled_size_estimate(Bunch(images=images, target=target)) == images.nbytes + target.nbytes
    assert sizing.pickled_size_estimate(Rows([images, target])) == images.nbytes + target.nbytes


def test_its_attributes_still_count():
    bunch = Bunch(images=np.ones(ROWS))
    bunch.__dict__["extra"] = np.ones(ROWS)
    assert sizing.pickled_size_estimate(bunch) == 2 * np.ones(ROWS).nbytes


def test_the_ram_tier_keeps_its_pickle_and_copies_no_array(monkeypatch):
    copied = []
    real = memory_backend._copy_array
    monkeypatch.setattr(memory_backend, "_copy_array", lambda a: copied.append(a) or real(a))
    ram = InMemoryBackend()
    bunch = Bunch(images=np.ones((ROWS, 2)), target=np.arange(ROWS), DESCR="faces")
    ram.set("b", bunch, {"execution_time": 60.0})
    assert type(ram._store["b"][1]) is memory_backend._Pickled
    assert not copied
    bunch["images"][0, 0] = -1.0  # after the store: not in the copy
    _meta, served = ram.get("b")
    assert type(served) is Bunch and served.DESCR == "faces"
    assert np.array_equal(served.images, np.ones((ROWS, 2))) and np.array_equal(served.target, np.arange(ROWS))
