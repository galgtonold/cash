"""The decorator's own arguments must not decide a cached entry's identity.

``inspect.getsource`` hands back the ``@...`` lines along with the function,
so every argument passed to ``@cash.cache`` used to land in the function's
source digest and therefore in its cache key. Changing one threw the entry
away.

Nothing about that was a decision. The purity analyzer already drops
``decorator_list`` from the same source before analyzing it; the identity
hash simply never got the same treatment. The tell is that adding an empty
``()`` -- which cannot mean anything -- invalidated too.

It made cash's own advice self-defeating: ``CashImpurityWarning`` tells the
user to add ``assume_safe=True`` after auditing, and taking that advice
recomputed everything, on exactly the expensive functions the warning fires
for.

This file is the matrix that pins which changes move the key and which do
not, at three levels:

* the digest itself, where the rule lives (fast, no caching involved);
* an in-process hit/miss matrix, because a digest that collapses correctly
  proves nothing if some other channel re-introduces the difference -- or,
  worse, if a dependency that MUST move the key was travelling through the
  decorator's text all along;
* one arm across a real process boundary, since "I restarted and it
  recomputed everything" is the complaint, and every other level here shares
  a process with the code it is testing.

Two arms are load-bearing controls rather than claims.
``test_matrix_must_recompute[body edited]`` fails if the harness ever stops
seeing an edit, which would turn every "restored" verdict into a false pass;
``test_a_restart_restores_across_a_real_process_boundary[]`` is the
empty-decorator control for the subprocess pair.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
import types

import pytest

import cash
from cash.backends import InMemoryBackend
from cash.source_norm import (
    _CACHE_DECORATOR_PARAMS,
    source_identity_digest,
    strip_cache_decorator,
)

# ---------------------------------------------------------------------------
# Level 1: the digest rule itself
# ---------------------------------------------------------------------------


def _src(decorators=("@_c.cache",), body="return n + 1") -> str:
    return "\n".join([*decorators, "def work(n):", f"    {body}"]) + "\n"


BARE = source_identity_digest(_src())

# Spellings of the SAME function. Cash's decorator carries configuration,
# never behaviour, so each of these has to digest identically to ``@_c.cache``.
SAME_IDENTITY = {
    "empty parens": "@_c.cache()",
    "assume_safe": "@_c.cache(assume_safe=True)",
    "strict": "@_c.cache(strict=True)",
    "allow_random": "@_c.cache(allow_random=True)",
    "ttl": "@_c.cache(ttl=3600)",
    "cache_if": "@_c.cache(cache_if=lambda r: r is not None)",
    "chunking": "@_c.cache(chunk_max_items=10, chunk_max_bytes=99)",
    "several at once": "@_c.cache(ttl=60, assume_safe=True, strict=False)",
    # The receiver is a runtime value, so the rule keys on the trailing
    # attribute; these are all real spellings from the docs and tests.
    "module singleton": "@cash.cache(assume_safe=True)",
    "factory call": "@get_cash().cache(assume_safe=True)",
    "attribute chain": "@self.cash.cache(assume_safe=True)",
}


@pytest.mark.parametrize("label", sorted(SAME_IDENTITY))
def test_a_cache_decorator_argument_never_moves_the_identity(label):
    assert source_identity_digest(_src((SAME_IDENTITY[label],))) == BARE


def test_a_multi_line_decorator_is_stripped_whole():
    """A ``black``-formatted decorator spans lines; all of them must go."""
    multiline = _src(("@_c.cache(", "    ttl=60,", "    assume_safe=True,", ")"))
    assert source_identity_digest(multiline) == BARE


def test_an_indented_decorator_is_stripped():
    """Methods arrive from ``inspect.getsource`` still indented."""
    indented = "    @_c.cache(assume_safe=True)\n    def work(n):\n        return n + 1\n"
    assert source_identity_digest(indented) == source_identity_digest(
        "    @_c.cache\n    def work(n):\n        return n + 1\n")


def test_async_functions_follow_the_same_rule():
    """``@cash.cache`` wraps coroutines too, and they are a separate AST node."""
    bare = "@_c.cache\nasync def work(n):\n    return n + 1\n"
    configured = "@_c.cache(assume_safe=True)\nasync def work(n):\n    return n + 1\n"
    assert source_identity_digest(bare) == source_identity_digest(configured)
    assert source_identity_digest(bare) != BARE, "async is not the sync function"


def test_a_stacked_decorator_survives_while_ours_is_stripped():
    """``@staticmethod`` above ``@cash.cache`` must still be part of identity."""
    bare = "@staticmethod\n@_c.cache\ndef work(n):\n    return n + 1\n"
    configured = "@staticmethod\n@_c.cache(ttl=60)\ndef work(n):\n    return n + 1\n"
    assert source_identity_digest(bare) == source_identity_digest(configured)
    assert source_identity_digest(bare) != BARE


def test_the_decorator_leaves_no_trace_at_all():
    """The endpoint of the rule, stated once so it cannot drift.

    A cached function digests exactly as the same function would undecorated.
    Safe because the digest is one component of a key that also carries
    ``module.qualname`` -- two different functions never meet here.
    """
    assert BARE == source_identity_digest("def work(n):\n    return n + 1\n")


# Changes that MUST still move the digest. The foreign-decorator arms are the
# reason the rule is scoped to cash's own decorator rather than to
# ``decorator_list``: ``@inject(db=prod)`` can absolutely change what the
# function returns, and nothing here can tell that it does not.
DIFFERENT_IDENTITY = {
    "body edited": _src(body="return n + 2"),
    "third-party decorator added": _src(("@marker(1)", "@_c.cache")),
    "third-party decorator argument": _src(("@marker(2)", "@_c.cache")),
    "foreign .cache decorator": _src(("@requests_cache.cache(expire_after=60)",)),
    "positional argument": _src(("@thing.cache(60)",)),
    "keyword cash does not accept": _src(("@thing.cache(expire_after=60)",)),
    "**kwargs splat": _src(("@thing.cache(**opts)",)),
}


@pytest.mark.parametrize("label", sorted(DIFFERENT_IDENTITY))
def test_a_real_change_still_moves_the_identity(label):
    assert source_identity_digest(DIFFERENT_IDENTITY[label]) != BARE


def test_the_foreign_decorators_stay_distinct_from_each_other():
    """Not just "different from bare" -- they must not collapse together."""
    digests = {source_identity_digest(src) for src in DIFFERENT_IDENTITY.values()}
    assert len(digests) == len(DIFFERENT_IDENTITY)


def test_class_decorators_are_never_stripped():
    """``@dataclass(frozen=True)`` is not cosmetic, and a class is not ours."""
    frozen = "@dataclass(frozen=True)\nclass K:\n    x: int\n"
    plain = "@dataclass()\nclass K:\n    x: int\n"
    assert source_identity_digest(frozen) != source_identity_digest(plain)


def test_source_that_will_not_parse_falls_back_to_itself():
    """A hasher must never raise; coarse beats broken."""
    broken = "@_c.cache(\ndef work(n):"
    assert strip_cache_decorator(broken) == broken


def test_an_undecorated_function_is_returned_untouched():
    """The common case must not pay for a parse."""
    plain = "def helper(n):\n    return n + 1\n"
    assert strip_cache_decorator(plain) is plain


def test_the_parameter_whitelist_matches_the_real_signature():
    """A new ``cache()`` parameter must be added to the whitelist too.

    Forgetting only reverts that parameter to the old over-invalidating
    behaviour -- the safe direction -- but it reverts it silently, so pin it.
    """
    declared = set(inspect.signature(cash.Cash.cache).parameters) - {"self", "func"}
    assert declared == set(_CACHE_DECORATOR_PARAMS)


# ---------------------------------------------------------------------------
# Level 2: the end-to-end matrix
# ---------------------------------------------------------------------------

MODULE = '''\
def marker(tag):
    def deco(fn):
        return fn
    return deco


def helper(n):
    return n + {helper_const}


def sidecar(n):
    # Never called by `work`, so naming it in `depends_on` is a NEW edge.
    # Naming a callee `work` already has would prove nothing: the analyzer
    # folds that one in either way.
    return n + 1


{decorators}
def work(n):
    return helper(n) * {body_const}
'''

# Both versions of ``work`` must report the same ``__module__``: the function
# key is ``module.qualname``, so a differing module name would separate them
# before the cache key ever mattered.
ARM_MODULE = "cash_decorator_arm"


class _Arm:
    """Two spellings of one function, sharing a backend. Does the second hit?

    Deliberately NOT a reload of one module. The first version of this test
    rewrote a module in place and re-imported it, which two independent
    caches keyed on ``(path, mtime, size)`` and ``id(func)`` are entitled to
    serve stale -- and did: it disagreed with itself between ``-n 16`` and
    ``-n0``, and both disagreed with a subprocess. Here the two versions are
    separate files, exec'd into separate namespaces, both alive at once, so
    neither cache can confuse them and nothing depends on filesystem
    timestamps.

    Separate ``Cash`` instances over ONE backend object stand in for two
    sessions against one cache directory. The subprocess test below is what
    checks that this stand-in is honest.
    """

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.backend = InMemoryBackend()
        self.datafile = tmp_path / "tracked.csv"
        self.datafile.write_text("a,b\n1,2\n", encoding="utf-8")
        self._alive = []
        self._version = 0

    def run(self, decorators=("@_c.cache",), helper_const=1, body_const=2):
        """Define this spelling of ``work``, call it, and report hit or miss."""
        self._version += 1
        source = MODULE.format(
            decorators="\n".join(decorators),
            helper_const=helper_const,
            body_const=body_const,
        )
        # A fresh filename per version: ``linecache`` keys on path plus stat,
        # and an edit that keeps the byte count identical (``+ 1`` -> ``+ 9``)
        # within one mtime tick is invisible to it.
        path = self.tmp_path / f"arm_v{self._version}.py"
        path.write_text(source, encoding="utf-8")
        instance = cash.Cash(backend=self.backend)

        # A real module in ``sys.modules``, not a bare dict. Cash re-resolves
        # each helper from ``sys.modules`` on every call and re-hashes its
        # source, so a namespace that is not registered there fails that
        # lookup and falls back to the hashes snapshotted when the parent was
        # analyzed -- which made an edited helper look unchanged. Registering
        # is also what a redefinition really does.
        module = types.ModuleType(ARM_MODULE)
        module.__file__ = str(path)
        module.__dict__["_c"] = instance
        sys.modules[ARM_MODULE] = module
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102

        work = module.work
        # Keep every version reachable for the life of the arm. Cash memoizes
        # source digests against ``id(...)``, and a collected function can
        # hand its id to its own replacement.
        self._alive.append((module, instance))
        before = work.cache_info()["hits"]
        work(21)
        return work.cache_info()["hits"] > before


@pytest.fixture
def arm(tmp_path):
    previous = sys.modules.get(ARM_MODULE)
    yield _Arm(tmp_path)
    if previous is None:
        sys.modules.pop(ARM_MODULE, None)
    else:
        sys.modules[ARM_MODULE] = previous


def _restorable(arm_, first, second):
    assert not arm_.run(**first), "the priming run should have been a miss"
    return arm_.run(**second)


# Changes that must leave the entry reachable. ``no change at all`` is the
# harness's positive control: without it, a harness that could never restore
# anything would pass the recompute half of the matrix and fail this half for
# the wrong reason.
MUST_RESTORE = {
    "no change at all": {},
    "empty parens added": {"decorators": ("@_c.cache()",)},
    "assume_safe added": {"decorators": ("@_c.cache(assume_safe=True)",)},
    "strict added": {"decorators": ("@_c.cache(strict=True)",)},
    "allow_random added": {"decorators": ("@_c.cache(allow_random=True)",)},
    "ttl added": {"decorators": ("@_c.cache(ttl=3600)",)},
    "chunk sizes changed": {"decorators": ("@_c.cache(chunk_max_items=7)",)},
}


@pytest.mark.parametrize("label", sorted(MUST_RESTORE))
def test_matrix_must_restore(arm, label):
    assert _restorable(arm, {}, MUST_RESTORE[label]), (
        f"{label!r} recomputed a value cash already had; the decorator's "
        f"arguments are back in the cache key"
    )


def test_assume_safe_can_also_be_removed(arm):
    """The direction a user hits second, after deciding the audit was wrong."""
    assert _restorable(arm, {"decorators": ("@_c.cache(assume_safe=True)",)}, {})


# Changes that must invalidate. ``body edited`` is the harness control: it is
# the one arm that would still pass if the reload silently served run one's
# source to run two, so a failure there indicts the harness, not the fix.
MUST_RECOMPUTE = {
    "body edited": {"body_const": 3},
    "helper edited": {"helper_const": 9},
    "depends_on added": {"decorators": ("@_c.cache(depends_on=[sidecar])",)},
    "third-party decorator argument changed": {
        "decorators": ("@marker(2)", "@_c.cache")},
}


@pytest.mark.parametrize("label", sorted(MUST_RECOMPUTE))
def test_matrix_must_recompute(arm, label):
    first = {"decorators": ("@marker(1)", "@_c.cache")} \
        if label == "third-party decorator argument changed" else {}
    assert not _restorable(arm, first, MUST_RECOMPUTE[label]), (
        f"{label!r} restored a stale value; that change used to move the key "
        f"only because the decorator's text was in the digest"
    )


SUBPROCESS = '''\
import sys
import cash
from cash.backends import FileBackend

# An explicit backend keeps the tiered cost model's ~100ms persist floor out
# of this: otherwise the body would need a sleep to clear it, and a
# wall-clock threshold is the one thing this must not depend on.
_c = cash.Cash(backend=FileBackend(sys.argv[1]))

RAN = []


@_c.cache{decorator}
def work(n):
    RAN.append(n)
    return n * 2


work(21)
print("RAN" if RAN else "RESTORED")
'''


@pytest.mark.parametrize("second", ["", "(assume_safe=True)"])
def test_a_restart_restores_across_a_real_process_boundary(tmp_path, second):
    """The user-visible claim, on the only oracle that cannot be fooled.

    Everything above shares a process with the code under test. This does
    not: it primes in one interpreter and reads in another, which is what
    "I added assume_safe and had to recompute everything" actually means.
    The empty-decorator arm is the control -- if it ever reports RAN, the
    cache is not surviving the restart for some unrelated reason and the
    ``assume_safe`` arm proves nothing.
    """
    script = tmp_path / "arm.py"
    cache_dir = tmp_path / "cache"

    def run(decorator):
        script.write_text(SUBPROCESS.format(decorator=decorator), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(script), str(cache_dir)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert done.returncode == 0, textwrap.indent(done.stderr or "", "    ")
        return done.stdout.strip()

    assert run("") == "RAN", "the priming run should have computed"
    assert run(second) == "RESTORED", (
        f"a fresh process with @_c.cache{second} recomputed a value already "
        f"on disk"
    )


def test_file_depends_on_still_invalidates_on_a_file_edit(arm):
    """The channel that carries ``file_depends_on`` is the dependency graph.

    Worth its own test rather than a matrix row: an earlier probe passed a
    RELATIVE path, which resolved against the process cwd instead of the
    fixture directory, and cash dutifully tracked a file that did not exist.
    The arm looked like a stale hit and was nearly reported as one.
    """
    decorators = (f"@_c.cache(file_depends_on={str(arm.datafile)!r})",)
    assert not arm.run(decorators=decorators)
    assert arm.run(decorators=decorators), "an untouched file should still restore"
    arm.datafile.write_text("a,b\n9,9\n", encoding="utf-8")
    assert not arm.run(decorators=decorators), (
        "editing a tracked file did not invalidate; declaring the dependency "
        "reached the key only through the decorator's text"
    )
