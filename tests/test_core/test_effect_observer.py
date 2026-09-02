"""Side effects the STATIC analyzer cannot see, caught at runtime instead.

The analyzer stops at library boundaries by design, so an effect inside an
installed package is reachable only by the method's NAME. Measured before this
existed: of 24 planted side effects, the only misses were all inside a library,
and a widened name list still could not reach two of them -- a network read
through a client object (``get`` collides with ``dict.get``) and a vendor
function whose return value was used.

These tests pin both halves of the answer:

* the widened name list (``notebook.purity._WRITE_METHODS``), which catches
  ``session.post`` / ``cur.execute`` statically, and
* :mod:`cash.effect_observer`, which watches the first call for a file write,
  an outbound connection, or a spawned process.

The controls matter as much as the cases. A detector that warns about
everything is not a detector, so a pure function must stay silent, and cash's
own cache write must never be reported as the user's effect.
"""
from __future__ import annotations

import socket
import threading
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpurityWarning


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _impurity_warnings(record):
    return [w for w in record if issubclass(type(w.message), CashImpurityWarning)]


def _call_capturing(c, fn, *args):
    """Call ``fn`` once through the cache, returning (result, warnings)."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = c.cache(fn)(*args)
    return result, _impurity_warnings(record)


# ---------------------------------------------------------------------------
# Controls -- without these, every assertion below is worthless
# ---------------------------------------------------------------------------

def test_a_pure_function_is_silent(tmp_path):
    """The detector has to discriminate, or "it warned" means nothing."""
    c = _cash(tmp_path)

    def pure(n):
        return n * 2

    result, warned = _call_capturing(c, pure, 3)
    assert result == 6
    assert not warned, f"a pure function warned: {[str(w.message) for w in warned]}"


def test_mutating_a_fresh_local_is_still_pure(tmp_path):
    """The fresh-allocation guard must survive the widened name list.

    ``out = []; out.append(x)`` uses a name in the write set, but the receiver
    is a local the function just allocated, so it cannot reach caller state.
    """
    c = _cash(tmp_path)

    def builds_a_list(n):
        out = []
        for i in range(n):
            out.append(i)
        return sum(out)

    result, warned = _call_capturing(c, builds_a_list, 4)
    assert result == 6
    assert not warned, f"mutating a fresh local warned: {[str(w.message) for w in warned]}"


def test_cash_own_cache_write_is_not_reported_as_the_users_effect(tmp_path):
    """Storing the entry is a file write inside the observed window.

    Without the cache-directory exclusion every cached function on a file
    backend would report itself as impure -- which is exactly the "warns about
    everything" failure the controls exist to catch.
    """
    c = _cash(tmp_path)

    def pure_but_stored(n):
        return list(range(n))

    _result, warned = _call_capturing(c, pure_but_stored, 5)
    assert not warned, (
        "cash's own cache write was attributed to the user's function: "
        f"{[str(w.message) for w in warned]}"
    )


# ---------------------------------------------------------------------------
# Tier 1 -- the widened name list, statically
# ---------------------------------------------------------------------------

class _Client:
    """Stands in for a requests Session / db cursor / bus producer."""

    def post(self, payload):
        return {"ok": True}

    def put(self, payload):
        return {"ok": True}

    def patch(self, payload):
        return {"ok": True}

    def execute(self, payload):
        return {"ok": True}

    def commit(self, payload):
        return {"ok": True}

    def publish(self, payload):
        return {"ok": True}


_CLIENT = _Client()


# Each verb is a LITERAL attribute call: `getattr(client, name)(...)` would be
# dynamic dispatch, which the analyzer raises on for an unrelated reason and
# would make this test pass without exercising the name list at all.
def _uses_post(payload):
    return _CLIENT.post(payload)["ok"]


def _uses_put(payload):
    return _CLIENT.put(payload)["ok"]


def _uses_patch(payload):
    return _CLIENT.patch(payload)["ok"]


def _uses_execute(payload):
    return _CLIENT.execute(payload)["ok"]


def _uses_commit(payload):
    return _CLIENT.commit(payload)["ok"]


def _uses_publish(payload):
    return _CLIENT.publish(payload)["ok"]


@pytest.mark.parametrize("verb, fn", [
    ("post", _uses_post), ("put", _uses_put), ("patch", _uses_patch),
    ("execute", _uses_execute), ("commit", _uses_commit), ("publish", _uses_publish),
])
def test_effect_shaped_method_names_are_flagged_on_any_receiver(tmp_path, verb, fn):
    """A client object's write verb is the only static handle on a library effect.

    The receiver's type is unknowable from source, so the name is all there is.
    Before these names were added, ``r = session.post(u); return r.json()``
    was silent while ``requests.post(u)`` warned -- the same operation, and the
    silent spelling is the one production code uses.
    """
    c = _cash(tmp_path)
    result, warned = _call_capturing(c, fn, {"a": 1})
    assert result is True
    assert warned, f"a call to .{verb}() on a client object was not flagged"
    assert verb in str(warned[0].message)


# ---------------------------------------------------------------------------
# Tier 3 -- observed at runtime, where no name could have reached
# ---------------------------------------------------------------------------

def test_a_file_write_inside_an_unnamed_call_is_observed(tmp_path):
    """The effect is real, the callee is not name-matched, the return is used.

    That combination is precisely what slipped through: the discarded-call rule
    did not apply and no write-method name appeared.
    """
    c = _cash(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("effect")
    target = tmp_path / "written.txt"

    # shutil.copyfile is a real write; it is stdlib (so not walked into) and it
    # is NOT in _IMPURE_MODULE_CALLS, which lists copy/copy2/move/rmtree but
    # not copyfile. No name matches, the return value is used -- exactly the
    # combination that was silent.
    def uses_stdlib_writer():
        import shutil
        return str(shutil.copyfile(source, target))

    result, warned = _call_capturing(c, uses_stdlib_writer)
    assert target.exists() and str(target) in result
    assert warned, "a file written during the first call was not observed"
    message = str(warned[0].message)
    assert "file write" in message and "written.txt" in message


def test_an_outbound_connection_is_observed(tmp_path):
    """A network READ through a client object -- unreachable by name.

    ``get`` cannot go in the write-method list because ``dict.get`` would
    match, so this class only closes by watching the call actually happen.
    """
    c = _cash(tmp_path)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    threading.Thread(target=lambda: server.accept(), daemon=True).start()

    def fetches():
        # `with` rather than an explicit conn.close(): a discarded call is
        # flagged by the STATIC pass, which would suppress the observer and
        # make this test pass for the wrong reason.
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            return "fetched"

    try:
        result, warned = _call_capturing(c, fetches)
        assert result == "fetched"
        assert warned, "an outbound connection during the first call was not observed"
        assert "network" in str(warned[0].message)
    finally:
        server.close()


def test_observed_effects_are_silenced_by_assume_safe(tmp_path):
    """``assume_safe=True`` means the user audited it; a second opinion is noise."""
    c = _cash(tmp_path)
    target = tmp_path / "audited.txt"

    def writes():
        with open(target, "w") as fh:
            fh.write("x")
        return 1

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        c.cache(assume_safe=True)(writes)()
    assert not _impurity_warnings(record), "assume_safe did not silence the observed effect"
    assert target.exists(), "assume_safe must not change what the function does"


def test_no_second_warning_when_the_static_pass_already_flagged(tmp_path):
    """One function, one warning. The static message is the more useful one:
    it names the line, which a runtime observation cannot."""
    c = _cash(tmp_path)
    target = tmp_path / "both.txt"

    def writes_by_name():
        # `write_text` IS in the static name list, so the analyzer flags this
        # AND the observer sees the write. The user should hear it once.
        import pathlib
        pathlib.Path(target).write_text("x")
        return 1

    _result, warned = _call_capturing(c, writes_by_name)
    assert len(warned) == 1, (
        f"expected exactly one warning, got {len(warned)}: "
        f"{[str(w.message) for w in warned]}"
    )


def test_a_cache_hit_does_not_re_warn_or_repeat_the_effect(tmp_path):
    """The point of the warning: the second call does neither the effect nor
    the warning, and returns the stored value."""
    c = _cash(tmp_path)
    target = tmp_path / "once.txt"

    def appends():
        with open(target, "a") as fh:
            fh.write("x")
        return "done"

    cached = c.cache(appends)
    with warnings.catch_warnings(record=True) as first_record:
        warnings.simplefilter("always")
        assert cached() == "done"
    with warnings.catch_warnings(record=True) as second_record:
        warnings.simplefilter("always")
        assert cached() == "done"

    assert _impurity_warnings(first_record), "the first call should warn"
    assert not _impurity_warnings(second_record), "the warning is once per function"
    assert target.read_text() == "x", (
        "the effect repeated on a cache hit -- if this ever holds, the warning "
        "is describing a hazard that does not exist"
    )
