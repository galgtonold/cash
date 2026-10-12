"""Library functions that only read their arguments.

Shared by the call units (which skip hashing such a call's arguments before
and after it) and the statement analysis (which does not fingerprint the
arguments of a bare call to one). Each is matched by identity against the
function its module holds, so a user's function of the same name never
passes.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache

__all__ = ["NOT_A_LITERAL", "reads_its_arguments_only", "turns_on_a_write", "write_parameter"]

#: Stands for an argument whose value is not known (an expression in a
#: statement's text): it may be anything, so it may ask for a write.
NOT_A_LITERAL = object()


#: Library functions that read their arguments and never write into them:
#: each builds a new object, or a number, from what it is given. By module
#: and name, resolved against the module the process already imported.
_READ_ONLY_CALLEES: dict[str, tuple[str, ...]] = {
    "pandas": ("concat", "merge", "to_datetime", "to_numeric", "crosstab", "pivot_table", "get_dummies"),
    "numpy": (
        "concatenate",
        "stack",
        "hstack",
        "vstack",
        "quantile",
        "percentile",
        "nanquantile",
        "nanpercentile",
        "median",
        "mean",
        "average",
        "std",
        "var",
        "histogram",
        "unique",
        "sort",
        "argsort",
        "array",
        "asarray",
    ),
    # Drawing reads the data it plots: ``plt.hist(deltas)`` over 1.7 million
    # values was fingerprinted before and after, 4.4 s plain against 25.7 s.
    "matplotlib.pyplot": ("hist", "plot", "scatter", "bar", "barh", "boxplot", "pie", "step", "errorbar", "imshow"),
    # seaborn's figure-level and axes-level plots: ``sns.countplot(x=..., data=temptastic)``
    # over 1.9 million rows was fingerprinted before and after, 2.4 s plain against 6.2 s.
    "seaborn": (
        "countplot",
        "lineplot",
        "barplot",
        "scatterplot",
        "histplot",
        "boxplot",
        "violinplot",
        "stripplot",
        "swarmplot",
        "pointplot",
        "kdeplot",
        "ecdfplot",
        "rugplot",
        "heatmap",
        "regplot",
        "displot",
        "catplot",
        "relplot",
        "lmplot",
        "pairplot",
        "jointplot",
    ),
}

_ALL_NAMES = frozenset(name for names in _READ_ONLY_CALLEES.values() for name in names)


def write_parameter(name: str) -> bool:
    """``out=``, ``inplace=`` and ``overwrite_input=`` (any ``overwrite_*``):
    the parameters through which a library function writes into what it is
    handed. One rule over names, so a function added to the list above is
    covered without a table of its own."""
    return name in ("out", "inplace") or name.startswith("overwrite_")


def _is_off(value) -> bool:
    """Only the literal ``False`` or ``None`` leaves a write parameter off."""
    return value is None or value is False


def turns_on_a_write(name: str, value) -> bool:
    """Is ``name=value`` a write parameter (`write_parameter`) left on?"""
    return write_parameter(name) and not _is_off(value)


@lru_cache(maxsize=256)
def _write_positions(fn) -> tuple[int, ...] | None:
    """The positions at which *fn* takes a write parameter positionally,
    or None when its signature cannot be read (a C function on an older
    numpy): then any argument after the first may be one."""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return tuple(index for index, p in enumerate(parameters) if p.kind in positional and write_parameter(p.name))


def _the_call_writes(fn, args: Sequence, kwargs: Mapping[str, object], open_ended: bool) -> bool:
    """Does this call of *fn* pass a write parameter that is on?

    *args* and *kwargs* are the values passed (`NOT_A_LITERAL` where only an
    expression is known); *open_ended* says ``*args`` or ``**kwargs`` passed
    more than can be seen.
    """
    if any(turns_on_a_write(name, value) for name, value in kwargs.items()):
        return True
    positions = _write_positions(fn)
    if positions is None:
        return open_ended or len(args) > 1
    if open_ended:
        return bool(positions)
    return any(index < len(args) and not _is_off(args[index]) for index in positions)


def reads_its_arguments_only(
    fn,
    args: Sequence = (),
    kwargs: Mapping[str, object] | None = None,
    *,
    open_ended: bool = False,
) -> bool:
    """Is *fn* one of `_READ_ONLY_CALLEES`, the very function its module
    holds under that name, called without asking it to write?

    Its arguments need no hash before and after it to see whether it wrote
    into them: ``df = pd.concat(df_array)`` over 87,000 small frames hashed
    every one twice, 100 s on top of a loop plain Python ran in 87 s.

    The call's *args* and *kwargs* decide the rest: ``np.median(a,
    overwrite_input=True)`` sorts ``a`` in place and ``np.mean(a, out=b)``
    writes ``b``, so a call that turns on a write parameter is not read-only.
    """
    name = getattr(fn, "__name__", None)
    if not isinstance(name, str) or name not in _ALL_NAMES:
        return False
    for module_name, names in _READ_ONLY_CALLEES.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if name in names and getattr(module, name, None) is fn:
            return not _the_call_writes(fn, args, kwargs or {}, open_ended)
    return False
