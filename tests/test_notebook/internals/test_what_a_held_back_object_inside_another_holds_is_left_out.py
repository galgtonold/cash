"""What a library object holds is not looked for when one of its parts
holds it back, also when it sits inside another library object.

A matplotlib figure is held back by its axes (``ax.figure``), so a figure
of its own is closed: its thousands of parts share the library's caches
and registries with every other figure, and none is looked for up the
heap. Inside a seaborn ``PairGrid`` it was read down part by part all the
same: E22's pair grid cell found ~1,900 parts held from outside and
climbed their referrers, 4-6 s a run against 1.7 s plain. Now such an
object inside another is left out as one of its own is, and what the
outer object holds besides it is still looked for.
"""

from __future__ import annotations

import argparse

import cash.notebook.shared_objects as shared_objects

VALUE_TYPES = shared_objects.VALUE_TYPES + shared_objects.library_value_types()


def _figure(shared):
    """A library object whose parts hold it back, as a figure's axes do."""
    figure = argparse.Namespace()
    figure.parts = [argparse.Namespace(owner=figure, data=shared) for _ in range(3)]
    return figure


def _inside(root):
    nodes = {id(root): root}
    inside, owners = shared_objects._held_from_outside(nodes, {id(root): "g"}, VALUE_TYPES)
    return inside, owners


def test_a_held_back_object_of_its_own_is_left_out():
    shared = [1, 2]
    figure = _figure(shared)
    inside, owners = _inside(figure)
    assert inside == {} and owners == set()
    assert shared  # held by this frame: outside the figure


def test_a_held_back_object_inside_another_is_left_out_too():
    shared = [1, 2]
    rows = [3, 4]
    grid = argparse.Namespace(figure=_figure(shared), rows=rows)
    inside, owners = _inside(grid)
    assert id(shared) not in inside
    assert id(grid.figure) not in inside
    assert id(rows) in inside and owners == {"g"}  # still the grid's own to look for
