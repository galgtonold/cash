"""Estimators that use their ``random_state`` only with some settings.

``random_state=None`` on an estimator was read as "its fit draws unseeded".
Many do; some use the seed only with particular settings, and say so in their
documentation: ``Ridge()`` only with ``solver='sag'`` or ``'saga'``,
``LogisticRegression()`` only with ``'sag'``, ``'saga'`` or ``'liblinear'``,
``Lasso()`` only with ``selection='random'``. With the defaults those fits are
deterministic, and RANDOM-UNSEEDED on them was a false claim.

The table below holds each such class, the setting its documentation names
and the values with which it draws. A setting cash can read -- a literal in the
call, the default of the installed class, the value on a live estimator, or
the value a fitted one reports it used (``Ridge.solver_``) -- decides. A value
whose draw depends on the data (``PCA(svd_solver='auto')``) or that cash cannot
read leaves it undecided, and the warning says what it depends on instead of
claiming a draw. A class not in the table keeps the old reading.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, NamedTuple

__all__ = ["call_draws", "draw_condition", "estimator_draws"]


class _Rule(NamedTuple):
    #: The setting the class's ``random_state`` documentation names.
    setting: str
    #: Values of *setting* with which the fit draws.
    draws: frozenset
    #: Values with which it may draw, depending on the data.
    undecided: frozenset = frozenset()
    #: Attributes of a fitted estimator holding the value actually used.
    fitted: tuple[str, ...] = ()
    #: How the documentation puts it, for the warning.
    when: str = ""


_SAG = _Rule("solver", frozenset({"sag", "saga"}), fitted=("solver_",), when="only with solver='sag' or 'saga'")
_LOGISTIC = _Rule(
    "solver", frozenset({"sag", "saga", "liblinear"}), when="only with solver='sag', 'saga' or 'liblinear'"
)
_SELECTION = _Rule("selection", frozenset({"random"}), when="only with selection='random'")
_PROBABILITY = _Rule("probability", frozenset({True}), when="only with probability=True")

#: scikit-learn class name -> when its fit draws (each from the class's own
#: ``random_state`` documentation).
_RULES: dict[str, _Rule] = {
    "Ridge": _SAG,
    "RidgeClassifier": _SAG,
    "LogisticRegression": _LOGISTIC,
    "LogisticRegressionCV": _LOGISTIC,
    **dict.fromkeys(
        (
            "Lasso",
            "ElasticNet",
            "LassoCV",
            "ElasticNetCV",
            "MultiTaskLasso",
            "MultiTaskElasticNet",
            "MultiTaskLassoCV",
            "MultiTaskElasticNetCV",
        ),
        _SELECTION,
    ),
    "SVC": _PROBABILITY,
    "NuSVC": _PROBABILITY,
    "PCA": _Rule(
        "svd_solver",
        frozenset({"arpack", "randomized"}),
        undecided=frozenset({"auto"}),
        fitted=("_fit_svd_solver",),
        when="only with svd_solver='arpack' or 'randomized' ('auto' picks one from the data's shape)",
    ),
}

_UNREAD = object()


def _rule(cls: Any) -> _Rule | None:
    """The rule for scikit-learn's own class *cls*; None for any other class,
    a subclass of one included (its ``fit`` may draw)."""
    if not isinstance(cls, type) or not (getattr(cls, "__module__", "") or "").startswith("sklearn."):
        return None
    return _RULES.get(cls.__name__)


def _verdict(rule: _Rule, value: Any) -> bool | None:
    if value is _UNREAD:
        return None
    try:
        if value in rule.undecided:
            return None
        return value in rule.draws
    except TypeError:  # an unhashable value: not one the documentation names
        return None


def estimator_draws(estimator: Any) -> bool | None:
    """Does this live estimator's fit draw at random, given its settings?

    True for a class not in the table (the reading ``random_state=None``
    always had), False when its own settings rule a draw out, None when that
    is undecided.
    """
    rule = _rule(type(estimator))
    if rule is None:
        return True
    value: Any = _UNREAD
    for name in rule.fitted:
        try:
            value = inspect.getattr_static(estimator, name)
        except AttributeError:
            continue
        if isinstance(value, str):
            break
        value = _UNREAD
    if value is _UNREAD:
        try:
            value = estimator.get_params(deep=False).get(rule.setting, _UNREAD)
        except Exception:  # noqa: BLE001 - an advisory never breaks a call
            return None
    return _verdict(rule, value)


def call_draws(cls: Any, call: ast.Call) -> bool | None:
    """Does ``cls(...)`` as written in *call* build an estimator whose fit
    draws? Read from a literal in the call or the installed class's default;
    anything else is undecided (None). True for a class not in the table."""
    rule = _rule(cls)
    if rule is None:
        return True
    value: Any = _UNREAD
    for kw in call.keywords:
        if kw.arg == rule.setting:
            return _verdict(rule, kw.value.value if isinstance(kw.value, ast.Constant) else _UNREAD)
    try:
        params = list(inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        return None
    for i, p in enumerate(params):
        if p.name != rule.setting:
            continue
        positional = p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        if positional and len(call.args) > i:
            arg = call.args[i]
            value = arg.value if isinstance(arg, ast.Constant) else _UNREAD
        elif p.default is not inspect.Parameter.empty:
            value = p.default
    return _verdict(rule, value)


def draw_condition(cls: Any) -> str | None:
    """``"Ridge uses random_state only with solver='sag' or 'saga'"``, or None
    for a class not in the table."""
    rule = _rule(cls)
    if rule is None:
        return None
    return f"{cls.__name__} uses random_state {rule.when}"
