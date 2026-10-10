"""What pyplot's process-global figure state is, to see a statement change it.

pyplot keeps one registry of open figures and a current figure and axes in
it. A statement or a function that makes a figure (``plt.figure()``), picks
one (``plt.subplot(2, 1, 1)``, ``plt.sca(ax)``) or draws on the current one
(``sns.barplot(...)``, ``df.plot()``, a helper that calls ``plt.plot``)
changes that state without binding a name, so no key or lineage sees it. A
hit of such a statement skips the change, and the next drawing lands on
another figure than plain Python's: a date plot's axis takes the bars of a
later ``sns.barplot`` and raises.

The names cash knows (``plt.*``) are refused before they run
(:data:`cash.effects.PYPLOT_MODULE_ALIASES`). Everything else is seen here:
the state is read before and after a statement or an intercepted call runs,
and one that changed it is not stored, so it runs every time.

The reading is cheap: which figures are open and which is current, and of
the current one its axes, how many artists each holds, and their titles and
labels. It never draws, autoscales or creates anything, and reads nothing
while pyplot is not imported.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CHANGED_PYPLOT_REASON", "pyplot_state"]

#: The uncacheable reason of a statement seen changing pyplot's figure state.
CHANGED_PYPLOT_REASON = (
    "Changes pyplot's figures (makes, picks or draws on the current figure): "
    "a cache hit would not, so the statement runs every time"
)

#: What the state reads as while pyplot is not imported: no figures.
_NO_FIGURES: tuple[Any, ...] = (None, ())


def _axes_state(ax: Any) -> tuple[Any, ...]:
    children = getattr(ax, "_children", None)
    count = len(children) if children is not None else len(ax.get_children())
    return (
        id(ax),
        count,
        id(getattr(ax, "legend_", None)),
        ax.title.get_text(),
        ax.xaxis.label.get_text(),
        ax.yaxis.label.get_text(),
    )


def _figure_state(fig: Any) -> tuple[Any, ...]:
    stack = getattr(fig, "_axstack", None)
    current = stack.current() if stack is not None else None
    suptitle = getattr(fig, "_suptitle", None)
    return (
        id(fig),
        id(current),
        len(fig.texts),
        len(fig.legends),
        suptitle.get_text() if suptitle is not None else None,
        tuple(fig.get_size_inches()),
        tuple(_axes_state(ax) for ax in fig.axes),
    )


def pyplot_state() -> tuple[Any, ...] | object:
    """A value that differs whenever pyplot's figures did: which are open,
    which is current, and what the current one's axes hold.

    Only the current figure is read in detail: a call that draws without
    naming an Axes draws on it (``sns.barplot``, ``s.plot()``), and one that
    moves to another figure changes which is current. That keeps the reading
    to tens of microseconds with a hundred figures open.

    The same value for "pyplot not imported" and "no figure open", so a
    statement that merely imports pyplot changes nothing. When the registry
    cannot be read, a fresh object, which equals no other reading: a
    statement is then taken to have changed it.
    """
    if "matplotlib.pyplot" not in sys.modules:
        return _NO_FIGURES
    helpers = sys.modules.get("matplotlib._pylab_helpers")
    gcf = getattr(helpers, "Gcf", None)
    if gcf is None:
        return _NO_FIGURES
    try:
        managers = gcf.figs
        if not managers:
            return _NO_FIGURES
        active = gcf.get_active()
        current = _figure_state(active.canvas.figure) if active is not None else None
        return (tuple(map(id, managers.values())), tuple(managers), current)
    except Exception:  # noqa: BLE001 - an unreadable registry counts as changed
        logger.debug("could not read pyplot's figures", exc_info=True)
        return object()
