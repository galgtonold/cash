"""A pyplot call that draws changes the names bound to the current figure.

``plt.plot(...)`` draws on ``plt.gca()`` with no name in the statement. The
names a variable binds to the current figure, its current axes, or a list or
array holding that axes are what the statement changes; a pyplot call that
only reads, creates or saves (``plt.gca()``, ``plt.subplots()``,
``plt.savefig()``) changes none of them.
"""

from __future__ import annotations

import ast

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


def pyplot_draw_names(tree, namespace):
    from cash.notebook.pyplot_draws import pyplot_draw_names as names

    return names(tree, namespace)


@pytest.fixture
def figure():
    plt.close("all")
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close("all")


@pytest.mark.parametrize(
    "code",
    ["plt.plot([1, 2])", "plt.title('t')", "plt.bar([1], [2])", "plt.gca().plot([1])", "plt.gcf().suptitle('s')"],
)
def test_a_drawing_call_names_the_figure_and_axes(figure, code):
    fig, ax = figure
    ns = {"plt": plt, "fig": fig, "ax": ax, "other": [1]}
    assert pyplot_draw_names(ast.parse(code), ns) == {"fig", "ax"}


@pytest.mark.parametrize("code", ["a = plt.gca()", "fig2, ax2 = plt.subplots()", "plt.savefig('x.png')", "plt.close()"])
def test_a_call_that_does_not_draw_names_nothing(figure, code):
    fig, ax = figure
    assert pyplot_draw_names(ast.parse(code), {"plt": plt, "fig": fig, "ax": ax}) == frozenset()


def test_an_array_holding_the_current_axes_is_named(figure):
    plt.close("all")
    fig, axes = plt.subplots(1, 2)
    assert pyplot_draw_names(ast.parse("plt.plot([1])"), {"plt": plt, "fig": fig, "axes": axes}) == {"fig", "axes"}


def test_with_no_figure_open_nothing_is_named():
    plt.close("all")
    assert pyplot_draw_names(ast.parse("plt.plot([1])"), {"plt": plt}) == frozenset()
