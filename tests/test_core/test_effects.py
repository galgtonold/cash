"""The shared effect vocabulary (`cash.effects`) and the two policies over it.

What a call IS is decided once, in `classify_call`; what each path DOES about
it is a table next to that path's code. These tests pin the vocabulary and
make sure neither table can fall behind it.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.file_effects import NOTEBOOK_POLICY
from cash.analysis.purity_analyzer import DECORATOR_POLICY
from cash.effects import Action, EffectKind, classify_call


def _call(src: str) -> ast.Call:
    node = ast.parse(src, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


@pytest.mark.parametrize("policy", [NOTEBOOK_POLICY, DECORATOR_POLICY], ids=["notebook", "decorator"])
def test_each_policy_decides_every_kind(policy):
    """Adding a kind without deciding it in BOTH paths is how the two name
    lists drifted apart; this is the check that stops it."""
    assert set(policy) == set(EffectKind)
    assert all(isinstance(action, Action) for action in policy.values())


@pytest.mark.parametrize(
    ("src", "kind"),
    [
        ("open(p, 'w')", EffectKind.FILE_WRITE),
        ("open(p)", EffectKind.FILE_READ),
        ("open(p, mode)", EffectKind.FILE_READ),
        ("df.to_csv(p)", EffectKind.FILE_WRITE),
        ("shutil.copytree(a, b)", EffectKind.FILE_WRITE),
        ("requests.get(u)", EffectKind.NETWORK_READ),
        ("requests.request('GET', u)", EffectKind.NETWORK_READ),
        ("requests.request('POST', u)", EffectKind.NETWORK_WRITE),
        ("requests.request(method, u)", EffectKind.NETWORK),
        ("urllib.request.urlopen(u)", EffectKind.NETWORK_READ),
        ("urllib.request.urlopen(u, data)", EffectKind.NETWORK_WRITE),
        ("session.post(u)", EffectKind.NETWORK_WRITE),
        ("cur.execute('SELECT 1')", EffectKind.DB_READ),
        ("cur.execute(sql)", EffectKind.DB_WRITE),
        ("df.to_sql('t', con)", EffectKind.DB_WRITE),
        ("pd.read_sql(q, con)", EffectKind.DB_READ),
        ("os.popen('ls')", EffectKind.SUBPROCESS),
        ("time.time()", EffectKind.CLOCK),
        ("time.strftime('%Y')", EffectKind.CLOCK),
        ("os.getenv('X')", EffectKind.ENVIRONMENT),
        ("print(x)", EffectKind.CONSOLE),
        ("sys.stderr.write(x)", EffectKind.CONSOLE),
        ("os.write(2, b)", EffectKind.CONSOLE),
        ("plt.plot(x)", EffectKind.DISPLAY),
        ("input()", EffectKind.INTERACTIVE),
    ],
)
def test_classify_call(src, kind):
    effect = classify_call(_call(src))
    assert effect is not None and effect.kind is kind


@pytest.mark.parametrize(
    "src",
    [
        "time.strftime('%Y', t)",  # formats t, reads no clock
        "time.localtime(ts)",  # converts ts
        "d.get(k)",  # `get` is not a verb: dict.get
        "s.replace('a', 'b')",  # nor is `replace`: str.replace
        "lst.append(x)",  # a mutation, not an effect
        "plt.figure()",  # creates a figure: judged as the object it returns
    ],
)
def test_not_an_effect(src):
    assert classify_call(_call(src)) is None


def test_savefig_on_pyplot_is_a_file_write_not_a_display():
    effect = classify_call(_call("plt.savefig(p)"))
    assert effect is not None and effect.kind is EffectKind.FILE_WRITE


def test_a_namespace_resolves_an_alias():
    import time as _t

    effect = classify_call(_call("now()"), {"now": _t.time})
    assert effect is not None and (effect.kind, effect.name) == (EffectKind.CLOCK, "time.time")


def test_a_shadowed_builtin_is_not_the_builtin():
    assert classify_call(_call("input()"), {"input": lambda: "stub"}) is None
