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

__all__ = [
    "ESTIMATOR_READING_METHODS",
    "NOT_A_LITERAL",
    "estimator_reads_its_arguments",
    "reads_its_arguments_only",
    "turns_on_a_write",
    "write_parameter",
]

#: Stands for an argument whose value is not known (an expression in a
#: statement's text): it may be anything, so it may ask for a write.
NOT_A_LITERAL = object()


#: Library functions that read their arguments and never write into them:
#: each builds a new object, or a number, from what it is given. By module
#: and name, resolved against the module the process already imported.
_READ_ONLY_CALLEES: dict[str, tuple[str, ...]] = {
    # The constructors too: ``for day, counts in day_counts.items():
    # weekday_df[day] = pd.Series(counts)`` hashed a Series of a million
    # dates in lists before and after the loop, 7 s plain against 34 s.
    "pandas": (
        "concat",
        "merge",
        "to_datetime",
        "to_numeric",
        "crosstab",
        "pivot_table",
        "get_dummies",
        "Series",
        "DataFrame",
        "Index",
    ),
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


# -- scikit-learn estimators ---------------------------------------------------

#: Methods of a scikit-learn estimator that leave what they are handed (``X``,
#: ``y``, ``sample_weight``) as it was. An assumption about one popular
#: library, stated here once: cash does not read sklearn's source to prove
#: it. It does not hold when the estimator is asked to work in place: a
#: ``copy=False`` scaler (``StandardScaler``, ``MinMaxScaler``,
#: ``Normalizer``, ``Binarizer``) scales ``X`` itself, and a linear model
#: with ``copy_X=False`` may centre it in place, so any such parameter set to
#: ``False``, on the estimator or on one it holds, voids the rule
#: (`estimator_reads_its_arguments`).
ESTIMATOR_READING_METHODS = frozenset(
    {
        "fit",
        "partial_fit",
        "predict",
        "predict_proba",
        "decision_function",
        "score",
        "transform",
        "fit_transform",
        "inverse_transform",
    }
)

#: The parameters through which an estimator is asked to work in place.
_IN_PLACE_PARAMETERS = frozenset({"copy", "copy_X"})

#: Modules whose objects an estimator may hold without running code of
#: anyone else's on what it is handed (a ``RandomState``, a scoring
#: function of sklearn's, an array of class weights).
_TRUSTED_ROOTS = frozenset({"sklearn", "numpy", "scipy", "builtins"})

_PLAIN = (type(None), bool, int, float, complex, str, bytes)


def _root_module(obj) -> str:
    module = getattr(obj, "__module__", None) if callable(obj) and not hasattr(obj, "get_params") else None
    if not isinstance(module, str):
        module = type(obj).__module__
    return module.partition(".")[0]


def _held_value_is_known(value, depth: int) -> bool:
    """Whether *value*, a parameter of a scikit-learn estimator, is known
    to do nothing to what the estimator is handed: plain data, sklearn's,
    numpy's or scipy's own objects, and containers of those. An estimator
    it holds must pass `estimator_reads_its_arguments`' test itself; a
    function or splitter of the user's (``FunctionTransformer(func=f)``,
    ``cv=MySplit()``) is handed the data, and is not known."""
    if depth > 20:
        return False
    if isinstance(value, _PLAIN):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_held_value_is_known(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(_held_value_is_known(k, depth + 1) and _held_value_is_known(v, depth + 1) for k, v in value.items())
    if _root_module(value) not in _TRUSTED_ROOTS:
        return False
    if callable(getattr(value, "get_params", None)) and not isinstance(value, type):
        return _estimator_is_known(value, depth + 1)
    return True


def _estimator_is_known(estimator, depth: int) -> bool:
    if type(estimator).__module__.partition(".")[0] != "sklearn":
        return False
    try:
        params = estimator.get_params(deep=True)
    except Exception:  # noqa: BLE001 - parameters not read: nothing is assumed
        return False
    if not isinstance(params, dict):
        return False
    for key, value in params.items():
        if not isinstance(key, str):
            return False
        if key.rpartition("__")[2] in _IN_PLACE_PARAMETERS and value is False:
            return False
        if not _held_value_is_known(value, depth):
            return False
    return True


def estimator_reads_its_arguments(estimator, method: str) -> bool:
    """Whether ``estimator.method(X, y)`` leaves ``X`` and ``y`` as they were,
    by the one assumption cash makes about scikit-learn
    (`ESTIMATOR_READING_METHODS`): *estimator* is scikit-learn's own (not a
    subclass of the user's), *method* is one of those methods, and no
    ``copy`` or ``copy_X`` parameter is ``False``, on it or on any estimator
    it holds (a ``Pipeline``'s steps, a ``GridSearchCV``'s estimator), read
    from ``get_params(deep=True)`` now. Everything an estimator holds must
    be known (`_held_value_is_known`); one that is not, or parameters that
    cannot be read, answer no."""
    if method not in ESTIMATOR_READING_METHODS:
        return False
    try:
        return _estimator_is_known(estimator, 0)
    except Exception:  # noqa: BLE001 - an estimator that cannot be inspected is not known
        return False
