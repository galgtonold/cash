"""Classification of *stateful carriers* for re-execution coherence.

A **stateful carrier** is an object whose observable result depends on hidden
internal state that *using* it advances or accumulates, with **no value-level
edge** recording the dependency:

* a seeded RNG — every draw advances the bit stream, so ``rng.standard_normal(n)``
  depends on ``rng = default_rng(7)`` through the generator's position;
* an accumulating builder — ``ax.bar(...)`` adds artists to a Figure that
  ``fig.savefig(path)`` later flushes to disk.

The simulation tracks **value lineage** (which variable feeds which). A carrier's
state edge is invisible to it: the carrier's lineage never advances and its
identity never changes. So re-executing a carrier's *consumer* without the
statements that established the carrier's state yields a result that corresponds
to **no possible execution of the notebook** — cash's cardinal sin.

**Deliberately disjoint from ``consumables.py``.** Generators / queues / open file
handles are carriers too, but they are ALREADY handled — by the consumable
channel, whose producer re-execution is gated on a per-type divergence probe
(``has_diverged`` against a cell-entry baseline) so it self-disables on
``run_all``. Classifying them here as well made this pass re-derive their
producers UNCONDITIONALLY, which re-initialised cross-cell accumulators and
regressed 12 integration tests while tripling their runtime. The two channels
stay disjoint: ``consumables`` owns the drain-position carriers, this table owns
the ones nothing else classifies.

Classification is by MRO ``module.qualname`` STRING match, mirroring
``cacheability_decision._IDENTITY_COUPLED_BASES``: it imports nothing,
so a notebook without numpy/matplotlib installed pays nothing and cannot break.
We match BASE classes, not leaves, so subclasses and projections (``Axes3D``,
a user's ``class MyFig(Figure)``) are covered.

**This table is a coverage floor, not a proof**, and the known gaps are measured
rather than assumed:

* ``plt.savefig()`` (as opposed to ``fig.savefig()``) depends on the current
  figure through pyplot's process-global ``Gcf`` registry, so the consuming
  statement's only input is the MODULE ``plt`` and there is no variable edge for
  the planner to follow at all. Not reachable from here.
* A generic accumulate-then-flush builder (``wb.save``, ``csv.writer``) was
  probed and does NOT exhibit the defect: being cacheable, its producer is
  RESTORED rather than re-derived, so its history never goes incoherent. That is
  why this table stops at the identity-coupled types instead of guessing at a
  long list of builders.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["stateful_carrier_kind"]

_CARRIER_BASES: Mapping[str, str] = {
    # Seeded RNGs: each draw advances the bit stream. ``default_rng``
    # has been numpy's recommended API since 1.17.
    'numpy.random._generator.Generator': 'numpy Generator',
    'numpy.random.mtrand.RandomState': 'numpy RandomState',
    'numpy.random.bit_generator.BitGenerator': 'numpy BitGenerator',
    'random.Random': 'random.Random',
    # Accumulate-then-flush builders: content is added by mutation and only
    # later written out, so a re-derived-but-unfilled builder writes a blank
    # artifact over a good one. These two are here rather than in a
    # longer list of plausible builders because they are the ones that PROVABLY
    # break: CAS-144 refuses to cache them (they are identity-coupled to
    # pyplot's globals), so the plan RE-EXECUTES the ``plt.subplots()`` that
    # produces them while the ``ax.bar(...)`` that fills them merely restores.
    # An ordinary builder (a user's ``Report``, an openpyxl ``Workbook``) is
    # cacheable, so its producer is restored rather than re-derived and the
    # incoherence never arises -- measured, not assumed. Adding speculative
    # entries here is not free: a table hit FORCES a producer re-execution.
    'matplotlib.figure.FigureBase': 'matplotlib Figure',      # Figure, SubFigure
    'matplotlib.axes._base._AxesBase': 'matplotlib Axes',     # Axes + projections
}

# Cheap pre-filter so the MRO walk only runs for plausibly-relevant objects.
# Must list the top-level package of every key above.
_CARRIER_MODULE_PREFIXES = (
    'numpy', 'random', '_random', 'matplotlib', 'mpl_toolkits',
)


def _mro_kind(value: Any) -> str | None:
    """Return the carrier kind for *value* by MRO string match, or ``None``."""
    try:
        mro = type(value).__mro__
    except AttributeError:  # pragma: no cover - exotic metaclass
        return None
    for base in mro:
        module = getattr(base, '__module__', '') or ''
        if not module.startswith(_CARRIER_MODULE_PREFIXES):
            continue
        kind = _CARRIER_BASES.get(f"{module}.{getattr(base, '__qualname__', '')}")
        if kind is not None:
            return kind
    return None


def stateful_carrier_kind(value: Any) -> str | None:
    """Return a human-readable carrier kind for *value*, or ``None``.

    ``None`` means "no evidence this object carries hidden state a consumer
    depends on" — the planner then leaves its producer alone, exactly as before.
    """
    if value is None:
        return None
    return _mro_kind(value)
