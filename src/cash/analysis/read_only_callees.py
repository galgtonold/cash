"""Library functions that only read their arguments.

Shared by the call units (which skip hashing such a call's arguments before
and after it) and the statement analysis (which does not fingerprint the
arguments of a bare call to one). Each is matched by identity against the
function its module holds, so a user's function of the same name never
passes.
"""

from __future__ import annotations

import sys

__all__ = ["reads_its_arguments_only"]


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
    "matplotlib.pyplot": ("hist", "plot", "scatter", "bar", "boxplot"),
}


def reads_its_arguments_only(fn) -> bool:
    """Is *fn* one of `_READ_ONLY_CALLEES`, the very function its module
    holds under that name?

    Its arguments need no hash before and after it to see whether it wrote
    into them: ``df = pd.concat(df_array)`` over 87,000 small frames hashed
    every one twice, 100 s on top of a loop plain Python ran in 87 s.
    """
    for module_name, names in _READ_ONLY_CALLEES.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        name = getattr(fn, "__name__", None)
        if name in names and getattr(module, name, None) is fn:
            return True
    return False
