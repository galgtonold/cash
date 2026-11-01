"""A library object that keeps bound methods of itself in its attributes is
not shared with anything outside it.

``pyprind.ProgBar.__init__`` sets ``self._stream_out = self._no_stream``: a
bound method held by the bar, holding the bar. The share check does not look
into a library's own object, so those references counted as holders, and a
loop updating the bar re-ran on every pass ("'pbar' is a ProgBar whose own
parts hold it").
"""

from __future__ import annotations

from cash.notebook.shared_objects import shared_names


def _no_stream(self, text=None):
    return None


def _init(self):
    self.count = 0
    self._stream_out = self._no_stream
    self._stream_flush = self._no_stream


#: A library's own class: not the notebook's, an estimator or a user module's.
ProgBar = type("ProgBar", (), {"__module__": "somelib.bar", "__init__": _init, "_no_stream": _no_stream})


def _shared(namespace: dict) -> set[str]:
    return shared_names({"pbar": namespace["pbar"]}, (namespace,))


def test_a_bar_holding_its_own_bound_methods_is_not_shared():
    namespace = {"pbar": ProgBar()}

    assert _shared(namespace) == set()


def test_its_bound_method_held_elsewhere_too_makes_it_shared():
    namespace = {"pbar": ProgBar()}
    namespace["write"] = namespace["pbar"]._stream_out

    assert _shared(namespace) == {"pbar"}


def test_another_holder_of_the_bar_still_makes_it_shared():
    namespace = {"pbar": ProgBar()}
    namespace["bars"] = [namespace["pbar"]]

    assert _shared(namespace) == {"pbar"}


def test_a_bound_method_of_another_object_does_not_vouch_for_it():
    namespace = {"pbar": ProgBar()}
    other = ProgBar()
    namespace["pbar"].other_out = other._no_stream
    namespace["keep"] = other

    assert _shared(namespace) == set()


class Bar:
    """The notebook's own class: the walk reads its attributes, methods too."""

    def __init__(self):
        self.out = self.write

    def write(self, text=None):
        return None


def test_a_walked_bar_held_by_another_container_is_still_shared():
    namespace = {"pbar": Bar()}
    namespace["bars"] = [namespace["pbar"]]

    assert _shared(namespace) == {"pbar"}


def test_a_walked_bar_holding_its_own_method_is_not_shared():
    namespace = {"pbar": Bar()}

    assert _shared(namespace) == set()
