"""A cached call that binds, rebinds or deletes attributes of an instance an
argument already held before the call -- one level down, ``est.kernel.memory_
= ...`` -- is stored with them, and a hit replays them on the instance the
argument it is handed holds there.

NiMARE's estimators fit with ``self.kernel_transformer`` gaining ``memory_``
and ``_shelving`` (nilearn's ``CacheMixin._fit_cache``): a change of the
argument ``self`` a hit could not make, so the fit ran every time. The rules
are those of the top-level replay: an instance of a plain class, each value
bound a new object of the call's own, an argument, the instance itself, a
module global, or an object the instances held before and still hold, bound
again by identity. Anything changed in place inside a held value, or deeper
than one level, keeps the refusal.

Each case runs the function plainly once, then cached on fresh objects, and
checks every cached call leaves exactly the plain run's state: the same
attributes in the same order on both levels, the same objects where the run
bound an argument or an object already held, the held instance still the
very object the caller had (so a holder outside the call sees the change).
"""

from __future__ import annotations

import textwrap
import warnings

import numpy as np
import pytest

from cash import rewrite_in_place
from tests._scripts import run_python

DEFAULT_MEMORY = ["a default the class shares"]


class Kernel:
    def __init__(self, memory=DEFAULT_MEMORY):
        self.memory = memory
        self.k = 2
        self.history = []

    def _fit_cache(self):
        # nilearn's CacheMixin._fit_cache, as NiMARE's kernels run it.
        self.memory_ = self.memory if self.memory is not None else {"location": None}
        if getattr(self, "_shelving", None) is None:
            self._shelving = False


class Inner:
    def __init__(self):
        self.k = 1


class Est:
    def __init__(self):
        self.memory = DEFAULT_MEMORY
        self.kernel = Kernel()
        self.kernel.inner = Inner()
        self.alpha = 1.0


def _content(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.tolist())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_content(v) for v in value])
    if isinstance(value, dict):
        return ("dict", [(k, _content(v)) for k, v in value.items()])
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return (type(value).__name__, sorted(vars(value)))
    return value


def _state(est, kernel, result, named):
    """What a run left on *est* and on what it holds, one level down: each
    attribute in order, by content and by which object it is -- *est*, the
    kernel it held before the call, a named outside object, or the n-th
    distinct object met (so aliasing across both levels shows)."""
    seen: dict[int, int] = {}
    known = {"est": est, "kernel before": kernel, **named}

    def ident(value):
        for name, held in known.items():
            if value is held:
                return name
        if type(value) in (int, float, str, bool, type(None)):
            return "atom"
        return seen.setdefault(id(value), len(seen))

    rows = []
    for key, value in vars(est).items():
        rows.append((key, ident(value), _content(value)))
        if isinstance(value, (Kernel, Inner)):
            rows.extend((f"{key}.{sub}", ident(held), _content(held)) for sub, held in vars(value).items())
    return rows, ident(result), _content(result)


def _check(disk_cash, method, calls=3, make=Est):
    """Run *method* plainly on a fresh *make*, then cached *calls* times on
    fresh ones, each with a holder outside the call that also holds its
    kernel: ``(runs of the cached body, whether every cached call left the
    plain run's state)``."""
    X = np.arange(6.0).reshape(3, 2)
    est = make()
    kernel = est.kernel
    outside = [kernel]
    plain = _state(est, kernel, method(est, X), {"X": X, "DEFAULT_MEMORY": DEFAULT_MEMORY})
    plain_outside = (_content(outside[0]), outside[0] is est.kernel)
    runs = []

    def body(est, X):
        runs.append(1)
        return method(est, X)

    body.__qualname__ = body.__name__ = method.__name__
    cached = disk_cash.cache(body)
    same = True
    for _ in range(calls):
        X = np.arange(6.0).reshape(3, 2)
        est = make()
        kernel = est.kernel
        outside = [kernel]
        result = cached(est, X)
        same = same and _state(est, kernel, result, {"X": X, "DEFAULT_MEMORY": DEFAULT_MEMORY}) == plain
        # The holder outside sees what a plain run leaves it: the hit bound
        # on that very object.
        same = same and (_content(outside[0]), outside[0] is est.kernel) == plain_outside
    return len(runs), same


def fit_like_nimare(self, X):
    self.kernel._fit_cache()
    self.coef_ = X.sum(0) * self.kernel.k
    return self


def fit_held_fresh(self, X):
    self.kernel.memory_ = {"location": None}
    self.kernel._shelving = False
    self.coef_ = X.sum(0)
    return 1


def fit_held_only(self, X):
    # Nothing bound on the argument itself.
    self.kernel.seen_ = X.shape
    return 1


def fit_held_rebind(self, X):
    self.kernel.k = self.kernel.k + 1
    self.k_used_ = self.kernel.k
    return 1


def fit_held_delete(self, X):
    del self.kernel.k
    self.n_ = 1
    return 1


def fit_held_kept(self, X):
    # An object the kernel held before and still holds.
    self.kernel.memory_ = self.kernel.memory
    self.kernel.history_ = self.kernel.history
    return 1


def fit_held_relinks(self, X):
    self.kernel.X_ = X
    self.kernel.owner_ = self
    self.kernel.default_ = DEFAULT_MEMORY
    return 1


def fit_held_alias(self, X):
    shared = [float(X.sum())]
    self.kernel.a_ = shared
    self.a_ = shared
    self.kernel.b_ = {"of": shared}
    return shared


def fit_held_and_top_kept(self, X):
    # The kernel itself, bound again at the top level.
    self.kernel_ = self.kernel
    self.kernel.fitted_ = True
    return 1


def fit_return_held(self, X):
    # A plain run hands back the kernel the argument holds.
    self.kernel.fitted_ = True
    return self.kernel


@pytest.mark.parametrize(
    "method",
    [
        fit_return_held,
        fit_like_nimare,
        fit_held_fresh,
        fit_held_only,
        fit_held_rebind,
        fit_held_delete,
        fit_held_kept,
        fit_held_relinks,
        fit_held_alias,
        fit_held_and_top_kept,
    ],
)
def test_attributes_bound_on_a_held_instance_are_replayed(disk_cash, method):
    assert _check(disk_cash, method) == (1, True)


def test_a_hit_binds_objects_of_its_own_on_the_held_instance(disk_cash):
    fit = disk_cash.cache(fit_held_fresh)
    kept = []
    for _ in range(3):
        est = Est()
        fit(est, np.ones((2, 2)))
        kept.append(est)
    assert fit.cache_info()["hits"] == 2
    assert kept[1].kernel.memory_ == {"location": None}
    assert kept[1].kernel.memory_ is not kept[2].kernel.memory_


class SharedMemory(Est):
    """The argument and its kernel hold one memory object of their own, as
    NiMARE's estimator and kernel hold joblib's default ``Memory``."""

    def __init__(self):
        super().__init__()
        self.memory = ["a memory of its own"]
        self.kernel.memory = self.memory


class EqualMemories(SharedMemory):
    """Two equal memory objects where `SharedMemory` has one."""

    def __init__(self):
        super().__init__()
        self.kernel.memory = list(self.memory)


@pytest.mark.parametrize("make", [SharedMemory, EqualMemories])
def test_an_object_held_at_several_places_is_bound_again_by_identity(disk_cash, make):
    """A hit binds the object the call bound, which the argument and its
    kernel both held, while every place still holds that one object."""
    assert _check(disk_cash, fit_like_nimare, make=make) == (1, True)
    fit = disk_cash.cache(fit_like_nimare)
    est = make()
    fit(est, np.arange(6.0).reshape(3, 2))
    assert est.kernel.memory_ is est.kernel.memory
    assert (est.kernel.memory is est.memory) == (make is SharedMemory)


def fit_held_append(self, X):
    self.kernel.history.append(1)
    self.coef_ = X.sum(0)
    return 1


def fit_two_levels_down(self, X):
    self.kernel.inner.k = 9
    self.coef_ = X.sum(0)
    return 1


def fit_held_then_replaced(self, X):
    self.kernel.memory_ = {}
    self.kernel = Kernel()
    return 1


def fit_held_bound_to_what_was_there(self, X):
    # An object the kernel held before, moved to another name.
    self.kernel.held_ = self.kernel.inner
    del self.kernel.inner
    return 1


@pytest.mark.parametrize(
    "method", [fit_held_append, fit_two_levels_down, fit_held_then_replaced, fit_held_bound_to_what_was_there]
)
def test_what_a_hit_cannot_replay_on_a_held_instance_runs_every_time(disk_cash, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _check(disk_cash, method) == (3, True)


def test_a_held_instance_rebound_after_its_change_says_why(disk_cash):
    @disk_cash.cache
    def fit(est, n):
        est.kernel.memory_ = {}
        est.kernel = Kernel()
        return n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit(Est(), 1)
        fit(Est(), 1)
    why = fit.explain(Est(), 1).details["why"]
    assert "also rebound or deleted est.kernel" in why, why


class Twice(Est):
    """Holds its kernel at two places: which one the call reached it
    through is not known."""

    def __init__(self):
        super().__init__()
        self.kernel_too = self.kernel


def fit_through_the_other(self, X):
    self.kernel_too.memory_ = {}
    return 1


def test_an_instance_held_at_two_places_runs_every_time(disk_cash):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _check(disk_cash, fit_through_the_other, make=Twice) == (3, True)


GLOBAL_KERNEL = Kernel()


class HoldsTheGlobal(Est):
    def __init__(self):
        super().__init__()
        self.kernel = GLOBAL_KERNEL


def test_a_held_instance_the_call_reaches_through_a_global_runs_every_time(disk_cash):
    """The body writes through ``GLOBAL_KERNEL``, which the argument holds
    when it is stored: a hit would bind on what the argument it is handed
    holds, where a plain run writes on whatever the global holds."""
    runs = []

    @disk_cash.cache
    def fit(est, n):
        runs.append(1)
        GLOBAL_KERNEL.seen_ = n
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            est = HoldsTheGlobal()
            fit(est, 3)
            assert est.kernel.seen_ == 3
    assert len(runs) == 2


class SlottedKernel:
    __slots__ = ("k", "memory_")

    def __init__(self):
        self.k = 2


class HoldsSlotted(Est):
    def __init__(self):
        super().__init__()
        self.kernel = SlottedKernel()


def test_a_held_instance_with_slots_runs_every_time(disk_cash):
    runs = []

    @disk_cash.cache
    def fit(est, n):
        runs.append(1)
        est.kernel.memory_ = n
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            est = HoldsSlotted()
            fit(est, 3)
            assert est.kernel.memory_ == 3
    assert len(runs) == 2


def test_without_the_c_watcher_held_attributes_are_compared(disk_cash, monkeypatch):
    """Below Python 3.12, or without the C extension, each held instance's
    ``__dict__`` is copied before the body and compared after it."""
    monkeypatch.setattr(rewrite_in_place, "cwatch", None)
    for method in (fit_like_nimare, fit_held_delete, fit_held_alias):
        assert _check(disk_cash, method) == (1, True), method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _check(disk_cash, fit_held_append) == (3, True)


def test_a_write_into_an_attribute_the_call_bound_fresh_is_not_named_as_the_cause(disk_cash):
    """``self.null_ = {}`` then ``self.null_[k] = v`` changes nothing the
    caller had: when the replay is refused, the refusal names what was
    changed in place, not that write."""

    @disk_cash.cache
    def fit(est, n):
        est.null_ = {}
        est.null_["values"] = n
        est.kernel.history.append(n)
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit(Est(), 1)
        fit(Est(), 1)
    why = fit.explain(Est(), 1).details["why"]
    assert "a hit cannot replay it" in why, why
    assert "est.kernel.history.append" in why, why
    assert "est.null_" not in why, why


def test_a_write_into_a_fresh_attribute_is_not_named_when_nothing_else_is_found(disk_cash):
    """ALESubtraction's shape: the only finding reading the source sees is
    the write into the dict the call bound itself; what changed is not in
    the body's own source."""

    def grow(kernel):
        kernel.history.append(1)

    @disk_cash.cache
    def fit(est, n):
        est.null_ = {}
        est.null_["values"] = n
        grow(est.kernel)
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit(Est(), 1)
        fit(Est(), 1)
    why = fit.explain(Est(), 1).details["why"]
    assert "est.null_" not in why, why


@pytest.mark.timeout(120)
def test_a_second_process_replays_what_the_first_stored_on_a_held_instance(tmp_path):
    script = tmp_path / "fit.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import sys
            import numpy as np
            import cash

            c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)
            LOG = sys.argv[2]
            DEFAULT = ["shared default"]

            class Kernel:
                def __init__(self):
                    self.memory = DEFAULT
                    self.k = 3

                def _fit_cache(self):
                    self.memory_ = self.memory
                    self._shelving = False

            class Est:
                def __init__(self):
                    self.memory = DEFAULT
                    self.kernel = Kernel()

            @c.cache
            def fit(est, X):
                fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
                os.write(fd, b"ran\\n")
                os.close(fd)
                est.kernel._fit_cache()
                est.kernel.X_ = X
                est.kernel.w_ = X * est.kernel.k
                del est.kernel.k
                est.coef_ = est.kernel.w_
                return est

            for _ in range(2):
                X = np.arange(4.0)
                est = Est()
                held = est.kernel
                assert fit(est, X) is est
                assert est.kernel is held
                assert list(vars(held)) == ["memory", "memory_", "_shelving", "X_", "w_"], vars(held)
                assert held.memory_ is held.memory is DEFAULT and held.X_ is X
                assert held.w_.tolist() == [0.0, 3.0, 6.0, 9.0] and est.coef_ is held.w_
            print("ok")
            """
        ),
        encoding="utf-8",
    )
    log = tmp_path / "runs.log"
    for _ in range(2):
        out = run_python(script, tmp_path / ".cash", log, cwd=tmp_path)
        assert out.stdout.strip() == "ok", out.stderr
    assert log.read_text(encoding="utf-8").count("ran") == 1
