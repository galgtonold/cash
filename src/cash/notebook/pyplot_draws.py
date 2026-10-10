"""Drawing through pyplot changes the figure and axes it draws on.

``plt.plot(...)``, ``plt.title(...)`` and the other module-level pyplot
calls draw on pyplot's *current* axes and figure, which the statement does
not name: ``fig, ax = plt.subplots()`` then ``plt.plot([1, 2, 3])`` changes
``ax.lines`` and what ``fig.savefig`` writes. A statement doing so has the
names bound to the current figure, to its current axes, or to an array of
Axes holding them, for changed in place: they are read and written by it,
so their lineage moves with its code and inputs, as ``ax.plot(...)`` moves
``ax``'s. The runtime finds them on the live objects after the statement
ran and records them per statement (``TrackingState.pyplot_draw_outputs``);
the simulation, which runs nothing, reads that record.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Mapping
from typing import Any

__all__ = ["current_figure_names", "draws_through_pyplot", "pyplot_draw_names"]

#: The pyplot functions that do not draw on the current axes or figure:
#: getters, the ones that make a new figure, write or show one, and settings.
_NOT_DRAWING = frozenset(
    {
        "close",
        "colormaps",
        "color_sequences",
        "connect",
        "disconnect",
        "draw",
        "draw_if_interactive",
        "fignum_exists",
        "figure",
        "findobj",
        "gca",
        "gcf",
        "gci",
        "get",
        "get_backend",
        "get_cmap",
        "get_current_fig_manager",
        "get_figlabels",
        "get_fignums",
        "get_plot_commands",
        "getp",
        "ginput",
        "imread",
        "imsave",
        "install_repl_displayhook",
        "ioff",
        "ion",
        "isinteractive",
        "new_figure_manager",
        "pause",
        "rc",
        "rc_context",
        "rc_file",
        "rc_file_defaults",
        "rcdefaults",
        "savefig",
        "show",
        "subplot_mosaic",
        "subplots",
        "switch_backend",
        "uninstall_repl_displayhook",
        "waitforbuttonpress",
        "xkcd",
    }
)


def _is_pyplot(node: ast.AST, namespace: Mapping[str, Any]) -> bool:
    """Whether *node* names the ``matplotlib.pyplot`` module."""
    if isinstance(node, ast.Attribute):
        return node.attr == "pyplot" and isinstance(node.value, ast.Name) and node.value.id == "matplotlib"
    if isinstance(node, ast.Name):
        value = namespace.get(node.id)
        if value is not None:
            return getattr(value, "__name__", None) == "matplotlib.pyplot"
        return node.id in ("plt", "pyplot")
    return False


def draws_through_pyplot(tree: ast.AST | None, namespace: Mapping[str, Any]) -> bool:
    """Whether *tree* calls a pyplot function that draws on the current axes
    or figure (``plt.plot``, ``plt.title``, ``plt.gca().bar(...)``)."""
    if tree is None:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value
        if _is_pyplot(receiver, namespace) and node.func.attr not in _NOT_DRAWING:
            return True
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Attribute)
            and receiver.func.attr in ("gca", "gcf")
            and _is_pyplot(receiver.func.value, namespace)
        ):
            return True
    return False


def current_figure_names(namespace: Mapping[str, Any]) -> set[str]:
    """The names of *namespace* bound to pyplot's current figure, to its
    current axes, or to a list, tuple or array of Axes holding those axes.
    Empty when pyplot is not imported or holds no figure (asking it then
    would make one)."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return set()
    try:
        if not plt.get_fignums():
            return set()
        fig = plt.gcf()
        current = fig.gca() if fig.axes else None
    except Exception:  # noqa: BLE001 - a backend that cannot answer draws nothing we can name
        return set()
    names: set[str] = set()
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if value is fig or (current is not None and value is current):
            names.add(name)
        elif current is not None and _holds_axes(value, current):
            names.add(name)
    return names


def _holds_axes(value: Any, axes: Any) -> bool:
    """Whether *value* is a list, tuple or numpy array of objects holding *axes*."""
    if type(value) in (list, tuple):
        return any(item is axes for item in value)
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ndarray) and value.dtype == object:
        return any(item is axes for item in value.flat)
    return False


def pyplot_draw_names(tree: ast.AST | None, namespace: Mapping[str, Any]) -> frozenset[str]:
    """The names a statement or structure *tree* changes by drawing through
    pyplot: those of `current_figure_names` when it draws at all."""
    if not draws_through_pyplot(tree, namespace):
        return frozenset()
    return frozenset(current_figure_names(namespace))
