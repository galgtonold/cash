"""A call with an effect is the same call under an alias or an imported name.

``subprocess.run``, ``requests.post``, ``os.system`` and ``os.makedirs``
were matched by their dotted text alone: only the clock, the environment
and the keyboard were looked up through what a name is bound to. So
``from subprocess import run; run(cmd)`` and ``import subprocess as sp;
sp.check_output(cmd)`` were no effect at all, for the notebook (which
stored the statement) and for the decorator (which said nothing).
"""

from __future__ import annotations

import ast
import os
import subprocess
import urllib.request

import pytest
import requests

from cash.analysis.cacheability import analyze_statement
from cash.effects import EffectKind, classify_call


def canonical_spellings(code, namespace):
    from cash.effects import canonical_spellings as read

    return read(code, namespace)


def _call(src: str) -> ast.Call:
    node = ast.parse(src, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


NAMESPACE = {
    "subprocess": subprocess,
    "sp": subprocess,
    "run": subprocess.run,
    "launch": subprocess.Popen,
    "system": os.system,
    "makedirs": os.makedirs,
    "o": os,
    "requests": requests,
    "rq": requests,
    "post": requests.post,
    "fetch": requests.get,
    "request": urllib.request,
    "urlopen": urllib.request.urlopen,
    "Request": urllib.request.Request,
}

ALIASED = [
    ("run(cmd)", EffectKind.SUBPROCESS, "subprocess.run"),
    ("sp.check_call(cmd)", EffectKind.SUBPROCESS, "subprocess.check_call"),
    ("sp.check_output(cmd)", EffectKind.SUBPROCESS, "subprocess.check_output"),
    ("launch(cmd)", EffectKind.SUBPROCESS, "subprocess.Popen"),
    ("system(cmd)", EffectKind.SUBPROCESS, "os.system"),
    ("o.system(cmd)", EffectKind.SUBPROCESS, "os.system"),
    ("makedirs(path)", EffectKind.FILE_WRITE, "os.makedirs"),
    ("post(url)", EffectKind.NETWORK_WRITE, "requests.post"),
    ("rq.put(url)", EffectKind.NETWORK_WRITE, "requests.put"),
    ("rq.request('DELETE', url)", EffectKind.NETWORK_WRITE, "requests.request"),
    ("rq.request('GET', url)", EffectKind.NETWORK_READ, "requests.request"),
    ("fetch(url)", EffectKind.NETWORK_READ, "requests.get"),
    ("request.urlopen(url, body)", EffectKind.NETWORK_WRITE, "urllib.request.urlopen"),
]


@pytest.mark.parametrize(("src", "kind", "name"), ALIASED)
def test_a_namespace_resolves_every_kind(src, kind, name):
    effect = classify_call(_call(src), NAMESPACE)
    assert effect is not None and (effect.kind, effect.name) == (kind, name)


@pytest.mark.parametrize(("src", "kind", "name"), ALIASED)
def test_the_spelling_alone_does_not_name_it(src, kind, name):
    """The control: without the namespace none of these is recognised as
    that call (``put`` is a verb on any receiver), so the test above is
    about the lookup."""
    effect = classify_call(_call(src))
    assert effect is None or effect.method


@pytest.mark.parametrize(("src", "kind", "name"), ALIASED)
def test_the_names_read_ahead_classify_as_the_namespace_does(src, kind, name):
    aliases = canonical_spellings(src, NAMESPACE)
    effect = classify_call(_call(src), None, aliases)
    assert effect is not None and (effect.kind, effect.name) == (kind, name)


@pytest.mark.parametrize("src", ["sp.PIPE", "o.path.join(a, b)", "rows.append(run_id)", "undefined(cmd)", "len(cmd)"])
def test_a_name_bound_to_no_known_call_is_no_effect(src):
    assert canonical_spellings(src, {**NAMESPACE, "rows": [], "run_id": 3, "len": len}) == {}


def test_a_name_rebound_to_the_users_own_function_is_not_the_library_call():
    def run(cmd):
        return cmd

    assert classify_call(_call("run(cmd)"), {"run": run}) is None
    assert canonical_spellings("run(cmd)", {"run": run}) == {}


@pytest.mark.parametrize(
    ("src", "kind"),
    [
        ("urlopen(Request(url, data=body, method='POST'))", EffectKind.NETWORK_WRITE),
        ("urlopen(Request(url, body))", EffectKind.NETWORK_WRITE),
        ("urlopen(Request(url, method='DELETE'))", EffectKind.NETWORK_WRITE),
        ("urlopen(urllib.request.Request(url, data=body))", EffectKind.NETWORK_WRITE),
        ("urlopen(Request(url, method=verb))", EffectKind.NETWORK),
        ("urlopen(Request(url))", EffectKind.NETWORK_READ),
        ("urlopen(Request(url, None))", EffectKind.NETWORK_READ),
        ("urlopen(Request(url, data=None, method=None))", EffectKind.NETWORK_READ),
        ("urlopen(Request(url, method='head'))", EffectKind.NETWORK_READ),
        ("urlopen(Request(url, headers=h))", EffectKind.NETWORK_READ),
        ("urlopen(url)", EffectKind.NETWORK_READ),
    ],
)
def test_urlopen_is_judged_by_the_request_written_in_it(src, kind):
    effect = classify_call(_call(src))
    assert effect is not None and effect.kind is kind


@pytest.mark.parametrize(
    ("code", "label", "what"),
    [
        ("run(['make'])", "system", "subprocess.run()"),
        ("out = sp.check_output(['git', 'rev-parse', 'HEAD'])", "system", "subprocess.check_output()"),
        ("system('make')", "system", "os.system()"),
        ("r = post(url, data=b'x')", "network", "requests.post()"),
        ("makedirs('out')", "file_write", "os.makedirs()"),
        ("r = urlopen(Request(url, data=b'x', method='POST'))", "network", "urlopen()"),
    ],
)
def test_a_notebook_statement_names_the_side_effect(code, label, what):
    effects = analyze_statement(code, None, NAMESPACE).side_effects
    assert [(e.kind, e.description) for e in effects] == [(label, what)]
    assert analyze_statement(code, None, NAMESPACE).skip_reasons(set()) == [f"Side effect: {what} ({label})"]


def test_the_statement_analysis_is_not_shared_between_two_bindings_of_a_name():
    """The analysis is memoised on the statement's text: what its names are
    bound to is part of the question."""

    def run(cmd):
        return cmd

    assert analyze_statement("run(['make'])", None, {"run": subprocess.run}).side_effects
    assert not analyze_statement("run(['make'])", None, {"run": run}).side_effects
    assert analyze_statement("run(['make'])", None, {"run": subprocess.run}).side_effects


def test_a_fetch_through_an_imported_name_is_still_no_side_effect():
    assert analyze_statement("page = fetch(url)", None, NAMESPACE).side_effects == ()
    assert analyze_statement("page = urlopen(Request(url))", None, NAMESPACE).side_effects == ()


def _warnings_of(func, *args):
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        func(*args)
    return [str(w.message) for w in seen if "cash" in type(w.message).__module__]


def test_a_cached_function_posting_through_an_imported_name_is_told_so(disk_cash, monkeypatch):
    """The decorator reads the same vocabulary: ``post(url)`` is a request a
    hit would not send, not a read to put a ``ttl=`` on."""
    from requests import post

    sent = []
    monkeypatch.setattr(requests.sessions.Session, "request", lambda self, method, url, **kw: sent.append(method))

    @disk_cash.cache
    def notify(url):
        post(url, data=b"x")
        return 1

    said = "\n".join(_warnings_of(notify, "http://127.0.0.1:9/hook"))
    assert sent == ["post"]
    assert "post" in said and "KEY-NETWORK-READ" not in said, said
