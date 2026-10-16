"""An object a library's parts of it hold is not shared, when nothing else
holds those parts.

Each trace and the layout of a plotly figure point back at the figure, as
each node of a minidom document points at the document. The share check
does not look into a library's own object, so the figure read as held from
outside, and a search of every object the garbage collector tracks ran for
each statement making one: 147 searches, 6.6 s, on E23. The parts are now
read down (`_held_within`), and a reference to the figure from a part that
nothing else holds is the figure's own.
"""

from __future__ import annotations

from xml.dom import minidom

from cash.notebook import shared_objects
from cash.notebook.shared_objects import shared_names

#: A library's own classes: not the notebook's, so the walk does not look in.
Figure = type("Figure", (), {"__module__": "somelib.figure"})
Trace = type("Trace", (), {"__module__": "somelib.figure"})


def _figure(traces: int = 3):
    fig = Figure()
    fig.data = tuple(Trace() for _ in range(traces))
    for trace in fig.data:
        trace.parent = fig
        trace.points = [1.0, 2.0]
    fig.layout = {"title": "t", "owner": fig}
    return fig


def _no_search(monkeypatch):
    searched = []
    real = shared_objects._holders_from_referrers
    monkeypatch.setattr(shared_objects, "_holders_from_referrers", lambda *a, **k: searched.append(1) or real(*a, **k))
    return searched


def test_a_figure_its_traces_point_back_at_is_not_shared(monkeypatch):
    searched = _no_search(monkeypatch)
    namespace = {"fig": _figure()}
    assert shared_names({"fig": namespace["fig"]}, (namespace,)) == set()
    assert searched == []


def test_a_trace_another_variable_holds_shares_the_figure():
    namespace = {"fig": _figure()}
    namespace["first"] = namespace["fig"].data[0]
    assert shared_names({"fig": namespace["fig"]}, (namespace,)) == {"fig"}


def test_a_list_inside_a_trace_is_not_looked_for_as_in_any_library_object():
    """What a library's object holds is not read for other holders, back
    references or not (known-limitations.md, 'Mutating an object created in
    an earlier cell'): the walk does not look into a library's object."""
    plain = Trace()
    plain.points = [1.0, 2.0]
    namespace = {"fig": _figure(), "plain": plain}
    del plain
    namespace["points"] = namespace["fig"].data[1].points
    namespace["other"] = namespace["plain"].points
    assert shared_names({"plain": namespace["plain"]}, (namespace,)) == set()
    assert shared_names({"fig": namespace["fig"]}, (namespace,)) == set()


def test_a_figure_another_container_holds_is_shared():
    namespace = {"fig": _figure()}
    namespace["figs"] = [namespace["fig"]]
    assert shared_names({"fig": namespace["fig"]}, (namespace,)) == {"fig"}


def test_a_minidom_document_is_read_down_to_its_nodes():
    namespace = {"doc": minidom.parseString("<a><b/><c>t</c></a>")}
    assert shared_names({"doc": namespace["doc"]}, (namespace,)) == set()
    namespace["node"] = namespace["doc"].documentElement.firstChild
    assert shared_names({"doc": namespace["doc"]}, (namespace,)) == {"doc"}


def test_too_many_parts_leave_it_to_the_search(monkeypatch):
    monkeypatch.setattr(shared_objects, "_WITHIN_MAX_OBJECTS", 3)
    namespace = {"fig": _figure(10)}
    assert shared_names({"fig": namespace["fig"]}, (namespace,)) == {"fig"}
