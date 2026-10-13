"""An object of an output that a library object inside the same output
holds too is not shared with anything outside it.

A LightGBM classifier's ``_best_score`` is also its booster's
``best_score``. The share check looks into the classifier (an estimator)
but not into the booster (a library's own object), so the dict looked held
by "another variable or container", and ``model = train_lgbm(...)`` re-ran
every time with nothing else holding the model.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict

from cash.notebook.shared_objects import shared_names

#: A library's own class: not the notebook's, an estimator or a user module's.
Booster = type("Booster", (), {"__module__": "somelib.basic"})


class Classifier:
    """The estimator interface: the share check looks into its attributes."""

    def get_params(self):
        return {}

    def fit(self, X, y):
        self._Booster = Booster()
        self._Booster.best_score = defaultdict(OrderedDict)
        self._best_score = self._Booster.best_score
        return self


def _shared(namespace: dict) -> set[str]:
    return shared_names({"model": namespace["model"]}, (namespace,))


def test_a_dict_the_estimator_and_its_booster_both_hold_is_not_shared():
    namespace = {"model": Classifier().fit(None, None)}

    assert _shared(namespace) == set()


def test_the_dict_held_by_another_variable_too_is_shared():
    namespace = {"model": Classifier().fit(None, None)}
    namespace["scores"] = namespace["model"]._best_score

    assert _shared(namespace) == {"model"}


def test_a_booster_another_variable_holds_does_not_vouch_for_what_it_holds():
    namespace = {"model": Classifier().fit(None, None)}
    namespace["booster"] = namespace["model"]._Booster

    assert _shared(namespace) == {"model"}
