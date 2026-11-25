"""A cached call that binds a new ``joblib.Memory`` on an instance its
argument held -- nilearn's ``CacheMixin._fit_cache`` giving NiMARE's
``self.kernel_transformer`` a ``memory_`` built from ``memory=None`` or a
cache folder -- is stored, and a hit binds an equal ``Memory``.

joblib's ``Memory.__getstate__`` hands back every attribute but
``timestamp``, which it sets to None: the time the object was built
(``time.time()`` in ``__init__``), left out so a ``Memory``'s hash does not
depend on it. Nothing else reads it but ``Memory.cache``, which hands it to
the ``MemorizedFunc`` it builds for its log lines. So the call was refused
("whose __getstate__ leaves out or changes some of its attributes"). A
stored ``Memory`` now comes back with its other attributes as pickled and
``timestamp`` read from the clock as it is rebuilt -- what a plain run
building it at that moment binds. Its location, backend, mmap mode,
compression, verbosity, backend options and store backend are the ones the
call built.
"""

from __future__ import annotations

import pickle
import textwrap
import time

import pytest

from cash import kept_state
from tests._scripts import run_python

joblib = pytest.importorskip("joblib")
Memory = joblib.Memory


class Kernel:
    def __init__(self, memory=None, verbose=0):
        self.memory = memory
        self.verbose = verbose

    def _fit_cache(self):
        # nilearn's CacheMixin._fit_cache with its check_memory: None and a
        # folder each give a Memory built here; a Memory is kept as it is.
        memory = self.memory
        if memory is None or isinstance(memory, str):
            memory = Memory(location=memory, verbose=self.verbose)
        self.memory_ = memory
        if getattr(self, "_shelving", None) is None:
            self._shelving = False


class Est:
    def __init__(self, memory=None):
        self.kernel_transformer = Kernel(memory)
        self.alpha = 0.05


def fit(est, n):
    est.kernel_transformer._fit_cache()
    est.result_ = n * 2
    return est


def _settings(memory):
    """What a ``Memory`` holds and does, but the moment it was built."""
    backend = memory.store_backend
    return (
        type(memory),
        list(vars(memory)),
        {k: v for k, v in vars(memory).items() if k not in ("timestamp", "store_backend")},
        type(memory.timestamp),
        repr(memory),
        None if backend is None else (type(backend), vars(backend)),
    )


def _state(est):
    kernel = est.kernel_transformer
    return (
        list(vars(est)),
        list(vars(kernel)),
        est.result_,
        kernel._shelving,
        type(kernel.memory_),
        _settings(kernel.memory_),
    )


@pytest.mark.parametrize("located", [False, True])
def test_a_new_memory_bound_on_a_held_instance_is_stored(disk_cash, tmp_path, located):
    location = str(tmp_path / "joblib-cache") if located else None
    plain = Est(location)
    fit(plain, 3)
    runs = []

    def body(est, n):
        runs.append(1)
        return fit(est, n)

    body.__qualname__ = body.__name__ = "fit"
    cached = disk_cash.cache(body)
    for _ in range(3):
        est = Est(location)
        kernel = est.kernel_transformer
        before = time.time()
        assert cached(est, 3) is est
        assert est.kernel_transformer is kernel
        assert _state(est) == _state(plain)
        # Read from the clock as it was built, as a plain run's is.
        assert before <= kernel.memory_.timestamp <= time.time()
    assert len(runs) == 1
    assert cached.cache_info()["hits"] == 2


def test_the_shared_memory_is_bound_again_by_identity(disk_cash):
    """A ``Memory`` the kernel held before (NiMARE's default) is the very
    object again on a hit, not a copy."""
    cached = disk_cash.cache(fit)
    for _ in range(2):
        shared = Memory(location=None, verbose=0)
        est = Est(shared)
        cached(est, 3)
        assert est.kernel_transformer.memory_ is shared
    assert cached.cache_info()["hits"] == 1


def test_a_stored_memory_is_rebuilt_with_its_settings(tmp_path):
    memory = Memory(location=str(tmp_path), mmap_mode="r", verbose=3, backend_options={"x": 1})
    back = pickle.loads(kept_state.dumps(memory))
    assert _settings(back) == _settings(memory)
    assert back.timestamp is not None
    # The same bytes whenever it was built: keyed and stored, the clock is
    # left out.
    later = Memory(location=str(tmp_path), mmap_mode="r", verbose=3, backend_options={"x": 1})
    later.timestamp += 100.0
    assert kept_state.dumps(later) == kept_state.dumps(memory)
    assert kept_state.dumps(later, keyed=True) == kept_state.dumps(memory, keyed=True)


class OwnMemory(Memory):
    pass


def test_a_subclass_keeps_the_refusal(disk_cash):
    """Only joblib's own class, whose ``__getstate__`` is read above: a
    subclass may pickle otherwise, and runs every time."""

    def fit_own(est, n):
        est.kernel_transformer.memory_ = OwnMemory(location=None, verbose=0)
        return n

    cached = disk_cash.cache(fit_own)
    with pytest.warns(Warning):
        for _ in range(2):
            cached(Est(), 1)
    assert cached.cache_info()["hits"] == 0


def test_a_memory_without_a_timestamp_comes_back_without_one(disk_cash):
    """A ``Memory`` whose ``timestamp`` is None already pickles whole: a
    hit gives it back None, not a clock read."""

    @disk_cash.cache
    def fit_stamped_none(est, n):
        memory = Memory(location=None, verbose=0)
        memory.timestamp = None
        est.kernel_transformer.memory_ = memory
        return n

    for _ in range(2):
        est = Est()
        fit_stamped_none(est, 1)
        assert est.kernel_transformer.memory_.timestamp is None
    assert fit_stamped_none.cache_info()["hits"] == 1


def test_a_memory_bound_in_one_process_is_served_in_the_next(tmp_path):
    script = tmp_path / "main.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import sys
            import time

            import cash
            from joblib import Memory

            c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)
            LOG = sys.argv[2]
            LOCATION = sys.argv[3]


            class Kernel:
                def __init__(self, memory):
                    self.memory = memory

                def _fit_cache(self):
                    memory = self.memory
                    if memory is None or isinstance(memory, str):
                        memory = Memory(location=memory, verbose=0)
                    self.memory_ = memory
                    if getattr(self, "_shelving", None) is None:
                        self._shelving = False


            class Est:
                def __init__(self, memory):
                    self.kernel_transformer = Kernel(memory)


            @c.cache
            def fit(est, n):
                fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
                os.write(fd, b"ran\\n")
                os.close(fd)
                est.kernel_transformer._fit_cache()
                est.result_ = n
                return n


            for location in (None, LOCATION):
                plain = Kernel(location)
                plain._fit_cache()
                est = Est(location)
                before = time.time()
                fit(est, 1 if location is None else 2)
                kernel = est.kernel_transformer
                assert list(vars(kernel)) == list(vars(plain)), vars(kernel)
                got, want = kernel.memory_, plain.memory_
                assert type(got) is Memory and type(got.timestamp) is float
                assert before <= got.timestamp <= time.time()
                for name in ("location", "backend", "mmap_mode", "compress", "_verbose", "backend_options"):
                    assert getattr(got, name) == getattr(want, name), name
                assert repr(got) == repr(want)
                if location is None:
                    assert got.store_backend is None
                else:
                    assert vars(got.store_backend) == vars(want.store_backend)
            print("ok", fit.cache_info()["hits"])
            """
        ),
        encoding="utf-8",
    )
    log = tmp_path / "runs.log"
    location = tmp_path / "joblib-cache"
    outs = []
    for _ in range(2):
        out = run_python(script, tmp_path / ".cash", log, location, cwd=tmp_path)
        outs.append(out.stdout.strip())
    assert outs == ["ok 0", "ok 2"], outs
    assert log.read_text(encoding="utf-8").count("ran") == 2
