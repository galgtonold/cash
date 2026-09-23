"""One identity for a function's code, whoever asks.

The decorator's helper walk, its registration and pins, and the notebook's
function tracker each read a function's source and hashed it their own way.
Only the helper walk read a ``functools.wraps`` wrapper's OWN code; the other
two let ``inspect.getsource`` unwrap it and hashed the wrapped function's text
instead, so the wrapper's body was invisible to them: editing what a decorator
does around the function kept the key. They now all use
``cash.source_norm.callable_identity``, which reads the wrapper's own code and
folds in the identity of what it wraps.
"""

from __future__ import annotations

import textwrap

import pytest

from cash import Cash
from cash.source_norm import callable_identity
from cash.tracking.function_tracker import FunctionTracker

MODULE = textwrap.dedent("""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs) {suffix}
        return wrapper

    @deco
    def area(r):
        return r * r {body}

    @deco
    def perimeter(r):
        return 4 * r
""")


def _load(tmp_path, name, suffix="", body=""):
    """Define the module's functions from a real file, so their source reads."""
    path = tmp_path / f"{name}.py"
    path.write_text(MODULE.format(suffix=suffix, body=body), encoding="utf-8")
    ns: dict = {"__name__": name}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns


@pytest.fixture
def identities():
    """Each way cash takes a function's identity."""
    tracker = FunctionTracker()
    return {
        "callable_identity": callable_identity,
        "decorator helper": Cash._hash_callable_source,
        "notebook tracker": tracker.get_function_source_hash,
    }


def test_every_path_agrees_on_a_wrapper(tmp_path, identities):
    ns = _load(tmp_path, "shapes")
    digests = {name: fn(ns["area"]) for name, fn in identities.items()}
    assert len(set(digests.values())) == 1, digests


def test_registration_keys_a_wrapper_as_the_helper_walk_does(tmp_path):
    """The cached function's own pin and a helper's digest are one rule."""
    ns = _load(tmp_path, "shapes_reg")
    c = Cash()
    c.cache(ns["area"])
    assert c.source_hashes[c.get_func_key(ns["area"])] == Cash._hash_callable_source(ns["area"])


@pytest.mark.parametrize("path", ["callable_identity", "decorator helper", "notebook tracker"])
def test_editing_the_decorator_body_moves_the_identity(tmp_path, identities, path):
    """``return fn(...)`` became ``return fn(...) + 1``: a different result."""
    before = identities[path](_load(tmp_path, "v1")["area"])
    after = identities[path](_load(tmp_path, "v2", suffix="+ 1")["area"])
    assert before != after


@pytest.mark.parametrize("path", ["callable_identity", "decorator helper", "notebook tracker"])
def test_editing_the_wrapped_body_moves_the_identity(tmp_path, identities, path):
    before = identities[path](_load(tmp_path, "w1")["area"])
    after = identities[path](_load(tmp_path, "w2", body="+ 1")["area"])
    assert before != after


@pytest.mark.parametrize("path", ["callable_identity", "decorator helper", "notebook tracker"])
def test_two_functions_one_decorator_do_not_share_an_identity(tmp_path, identities, path):
    ns = _load(tmp_path, "pair")
    assert identities[path](ns["area"]) != identities[path](ns["perimeter"])


@pytest.mark.parametrize("path", ["callable_identity", "decorator helper", "notebook tracker"])
def test_the_same_code_keeps_its_identity(tmp_path, identities, path):
    """The control: a fresh definition of identical code is the same identity."""
    assert identities[path](_load(tmp_path, "same_a")["area"]) == identities[path](_load(tmp_path, "same_b")["area"])
