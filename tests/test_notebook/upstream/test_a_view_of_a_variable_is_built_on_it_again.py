"""An entry stores a read-only numpy view of a variable's data, held inside
an output, as where it sits there, and a restore builds it on the live data
again.

A fitted ``KNeighborsRegressor`` keeps ``y.to_numpy()`` of the labels it was
fitted on, a read-only view of the Series' memory: in plain Python a write
to the labels in place changes what the model predicts. Stored as a copy,
the restored model would not see that write, so such outputs were refused
as live aliases and re-ran on every Run All (E6's randomized search). The
entry keeps the variable as an anchor (`MEMORY_HOME`): a restore builds the
view on the live variable's data when the bytes there are the ones stored,
and misses otherwise. A view that can be written through, and a view that
is itself the output, are still re-derived.
"""

from __future__ import annotations

import pickle
import types

import pytest

from cash.notebook.anchors import MEMORY_HOME, resolve_anchors, with_anchor_refs
from cash.notebook.borrowed_memory import memory_home, rebuild_view
from cash.notebook.statement.derivation_edges import read_only_view_homes
from cash.notebook.statement.output_refusals import unrestorable_output_reason

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


def _labels():
    return pd.Series(np.arange(10.0), name="y")


def _read_only_view(series, stop=None):
    view = series.to_numpy()[:stop]
    view.flags.writeable = False
    return view


def test_a_restored_view_sits_on_the_live_labels():
    y = _labels()
    model = types.SimpleNamespace(y_=_read_only_view(y, 6))
    stored = pickle.loads(pickle.dumps(with_anchor_refs({"model": model}, {"y": y})))

    restored = resolve_anchors(stored, {"y": MEMORY_HOME}, {"y": y})["model"]

    assert restored.y_.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert not restored.y_.flags.writeable
    y.iloc[2] = 100.0
    assert restored.y_[2] == 100.0  # as the original sees it


def test_a_restore_after_a_restart_sits_on_the_labels_built_again():
    before = _labels()
    stored = pickle.dumps(with_anchor_refs({"model": types.SimpleNamespace(y_=_read_only_view(before))}, {"y": before}))
    y = _labels()  # what the notebook builds again

    restored = resolve_anchors(pickle.loads(stored), {"y": MEMORY_HOME}, {"y": y})["model"]

    assert np.shares_memory(restored.y_, y.to_numpy())


def test_labels_with_other_bytes_are_a_miss():
    y = _labels()
    stored = pickle.dumps(with_anchor_refs({"m": types.SimpleNamespace(y_=_read_only_view(y))}, {"y": y}))

    with pytest.raises(KeyError):
        resolve_anchors(pickle.loads(stored), {"y": MEMORY_HOME}, {"y": _labels() + 1})


def test_labels_that_are_gone_are_a_miss():
    y = _labels()
    stored = pickle.dumps(with_anchor_refs({"m": types.SimpleNamespace(y_=_read_only_view(y))}, {"y": y}))

    with pytest.raises(KeyError):
        resolve_anchors(pickle.loads(stored), {"y": MEMORY_HOME}, {})


def test_a_view_that_can_be_written_through_has_no_home():
    a = np.arange(10.0)
    s = pd.Series(a)

    assert memory_home(a[2:5], {"a": a}) is None
    assert memory_home(_read_only_view(s, 5), {"s": s}).name == "s"


def test_a_home_rebuilds_a_strided_view():
    a = np.arange(24.0).reshape(4, 6)
    view = a[1:, ::2]
    view.flags.writeable = False
    home = memory_home(view, {"a": a})

    rebuilt = rebuild_view(home, a)

    assert rebuilt.tolist() == view.tolist()
    assert np.shares_memory(rebuilt, a)


def _refusal(outputs, user_ns, homes):
    return unrestorable_output_reason(
        set(outputs), {name: user_ns[name] for name in outputs}, user_ns, holders={}, homes=homes
    )


def test_an_object_holding_a_read_only_view_is_no_live_alias():
    user_ns = {"y": _labels()}
    user_ns["model"] = types.SimpleNamespace(y_=_read_only_view(user_ns["y"]))
    homes: dict[int, str] = {}

    reason = _refusal(["model"], user_ns, homes)

    assert reason is None or "Live-alias" not in reason, reason
    assert set(homes.values()) == {"y"}


@pytest.mark.parametrize(
    "made",
    [
        # The output is the view itself: re-derived, as ``X_train = X[:1000]``.
        lambda ns: ns["y"].to_numpy()[:5],
        # A view that can be written through: a write to it changes ``a``.
        lambda ns: types.SimpleNamespace(part=ns["a"][2:5]),
    ],
    ids=["the output is the view", "a writable view"],
)
def test_a_view_a_restore_cannot_build_again_still_refuses(made):
    user_ns = {"y": _labels(), "a": np.arange(10.0)}
    user_ns["out"] = made(user_ns)
    homes: dict[int, str] = {}

    reason = _refusal(["out"], user_ns, homes)

    assert reason is not None and "Live-alias" in reason, reason


def test_data_held_only_through_read_only_views_is_linked_one_way():
    """A write to ``y`` reaches the model; nothing written through the
    model reaches ``y``: the edge goes one way, whether or not the
    statement is stored."""
    y = _labels()
    a = np.arange(10.0)
    user_ns = {"y": y, "a": a}

    assert read_only_view_homes(types.SimpleNamespace(y_=_read_only_view(y)), user_ns) == {"y"}
    # Also a view that can be written through: both ways, as before.
    both = types.SimpleNamespace(y_=_read_only_view(y), part=a[2:5])
    assert read_only_view_homes(both, user_ns) == {"y"}
    assert read_only_view_homes(types.SimpleNamespace(part=a[2:5]), user_ns) == set()
    # The output is the view itself.
    assert read_only_view_homes(_read_only_view(y), user_ns) == set()
