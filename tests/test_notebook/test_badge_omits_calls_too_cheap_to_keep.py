"""The badge does not list an intercepted call that was too cheap to keep.

Round 29, r29s5: the badge read ``join() [intercepted]: 0/1 cached`` for
``os.path.join(...)`` -- a call cash wrapped, found under the cost floor and
did not store. Listed like that it reads as a cache that failed.
"""
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view_builder import build_interactive_badge


def _metrics(*events):
    return [{"status": "COMPUTED", "code": "p = f(os.path.join(a, b))", "execution_time": 0.5,
             "total_time": 0.5, "decorator_calls": list(events)}]


def _event(name, **kw):
    return {"func_name": name, "cache_hit": False, "execution_time": 0.0001,
            "intercepted": True, "cache_key": "call:x", "call_source": name + "(a)",
            "occurrence_index": 0, **kw}


def test_a_call_that_was_not_kept_is_not_listed():
    out = render_text(build_interactive_badge(_metrics(_event("posixpath.join", stored=False))))
    assert "join()" not in out, out


def test_a_stored_call_is_still_listed():
    out = render_text(build_interactive_badge(_metrics(
        _event("posixpath.join", stored=False), _event("mylib.fit", stored=True, execution_time=2.0))))
    assert "fit()" in out and "join()" not in out, out
