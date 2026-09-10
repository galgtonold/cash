"""Correctness of argument hashing for large numpy/pyarrow values.

Large arrays/tables used to be hashed by a SAMPLE (numpy: first/last 128
elements; pyarrow >=10MB: schema + row count only), so two different values
that matched the sample collided into a wrong cache hit - a silent
data-corruption bug for exactly the large arrays a data/ML cache targets.
These tests pin full-content hashing.
"""
from __future__ import annotations

import numpy as np
import pytest

from cash import Cash, FileBackend


def _counter_fn(c):
    calls = {"n": 0}

    @c.cache
    def f(arr):
        calls["n"] += 1
        return float(np.asarray(arr).sum())

    return f, calls


def test_large_numpy_middle_difference_does_not_collide(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f, calls = _counter_fn(c)
    base = np.ones(4_000_000)              # 32 MB -> old sampling path
    a = f(base)
    other = base.copy()
    other[2_000_000] = 999_999.0           # differs only in the middle
    b = f(other)
    assert calls["n"] == 2, "different arrays must not share a cache entry"
    assert a == base.sum()
    assert b == other.sum()
    assert a != b


def test_numpy_reshape_does_not_collide(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def shape_of(a):
        return tuple(a.shape)

    assert shape_of(np.ones(4)) == (4,)
    assert shape_of(np.ones((2, 2))) == (2, 2)   # same bytes, different shape


def test_equal_numpy_arrays_still_hit(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    f, calls = _counter_fn(c)
    arr = np.arange(1000.0)
    f(arr)
    f(arr.copy())                          # equal value, fresh object -> must hit
    assert calls["n"] == 1


def test_numpy_layout_is_part_of_the_key(tmp_path):
    """Layout DISCRIMINATES; it used to be invariant, and that was a wrong answer.

    This test previously asserted the opposite — that a C-ordered array and an
    F-ordered array of the same values share a cache key — because the
    ``tobytes()`` fallback normalises to C-order. That is right for value
    equality and wrong for a key: a round-15 tester showed a layout-sensitive
    kernel being served the other layout's result, with
    ``np.ravel(x, order='A')`` returning ``[0, 1, 2, …]`` for an F-ordered input
    whose true answer is ``[0, 4, 8, 1, …]``, 5/5 across separate processes.
    ``order='A'``, ``reshape``, ``.flags`` and any compiled callee expecting a
    layout all read what the normalisation erased.

    Changed deliberately, by owner decision: correctness over the hit rate. The
    cost is that same-values-different-layout inputs now miss instead of hit,
    and existing ndarray entries invalidate once on upgrade. See the
    layout-insensitive sibling below, which is the control that this did not
    simply stop caching.
    """
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def s(a):
        return float(a.sum())

    A = np.arange(20.0).reshape(4, 5)
    non_contig = np.ascontiguousarray(A.T).T   # same values, non-C-contiguous
    assert np.array_equal(A, non_contig)
    assert s.explain(A).cache_key != s.explain(non_contig).cache_key


def test_same_layout_same_values_still_share_a_key(tmp_path):
    """The control for the above: discrimination must not mean never hitting."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def s(a):
        return float(a.sum())

    A = np.arange(20.0).reshape(4, 5)
    assert s.explain(A).cache_key == s.explain(A.copy()).cache_key


def test_large_pyarrow_different_data_does_not_collide(tmp_path):
    pa = pytest.importorskip("pyarrow")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    calls = {"n": 0}

    @c.cache
    def rows(t):
        calls["n"] += 1
        return t.num_rows

    n = 2_000_000
    t1 = pa.table({"a": np.zeros(n), "b": np.ones(n)})        # >10 MB
    t2 = pa.table({"a": np.zeros(n), "b": np.full(n, 2.0)})   # same schema/rows, diff data
    rows(t1)
    rows(t2)
    assert calls["n"] == 2, "tables with different data must not collide"


# -- CAS-123: the layout that keys is memory ORDER, not stride size ----------

def _key(c, value):
    @c.cache
    def s(a):
        return float(a.sum())
    return s.explain(value).cache_key


@pytest.mark.parametrize("make_view", [
    lambda a: a[:, 0],                      # a column: 1-D, strided
    lambda a: a[::2, :],                    # every other row
    lambda a: a[:, ::-1],                   # reversed columns
    lambda a: np.broadcast_to(a[0], a.shape),   # zero-stride axis
], ids=["column", "row-step", "reversed", "broadcast"])
def test_a_view_and_its_restored_copy_share_a_key(tmp_path, make_view):
    """A cached function returning a view hands its caller a view once and a
    contiguous copy on every restore. Keying on raw strides made the caller's
    key differ between the two: its expensive step ran twice after every
    upstream edit (round 17, 1 then 1 then 0 executions)."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    view = make_view(np.arange(20.0).reshape(4, 5))
    import pickle
    restored = pickle.loads(pickle.dumps(view))
    assert np.array_equal(view, restored)
    assert _key(c, view) == _key(c, restored)


def test_a_permuted_layout_still_keys_apart(tmp_path):
    """The control: a genuinely different memory order reads differently
    (`ravel(order='K')`), so it must not share the C-ordered entry."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    a = np.arange(24.0).reshape(2, 3, 4)
    permuted = np.ascontiguousarray(a.transpose(1, 0, 2)).transpose(1, 0, 2)
    assert np.array_equal(a, permuted)
    assert not np.array_equal(np.ravel(a, order="K"), np.ravel(permuted, order="K"))
    assert _key(c, a) != _key(c, permuted)


def _order_readings(x):
    """Everything an order-reading callee can see: `ravel` and `tobytes` in each
    order. Two arrays may share a key only if all of these agree."""
    return (
        tuple(np.ravel(x, order=o).tolist() for o in "CFAK"),
        tuple(x.tobytes(order=o) for o in "CFA"),
    )


def _forms():
    rng = np.random.default_rng(0)
    v = rng.random((6, 8))
    vt = v.T
    s = rng.random((4, 5, 6))
    return {
        "C": v,
        "C strided": v[::2, ::2],
        "C strided copy": v[::2, ::2].copy(),
        "reversed": v[:, ::-1],
        "reversed copy": v[:, ::-1].copy(),
        "F view": vt,
        "F view C-copy": np.ascontiguousarray(vt),
        "F view F-copy": np.asfortranarray(vt),
        "F-like strided": vt[::2],
        "F-like strided C-copy": np.ascontiguousarray(vt[::2]),
        "F-like strided F-copy": np.asfortranarray(vt[::2]),
        "permuted": np.transpose(s, (1, 2, 0)),
        "permuted C-copy": np.ascontiguousarray(np.transpose(s, (1, 2, 0))),
        "permuted F-copy": np.asfortranarray(np.transpose(s, (1, 2, 0))),
        "row": v[:1, :],
        "row F": np.asfortranarray(v[:1, :]),
        "col": v[:, :1],
        "col copy": v[:, :1].copy(),
        "broadcast": np.broadcast_to(v[0], (8, 8)),
        "broadcast T": np.broadcast_to(v[0], (8, 8)).T,
        "broadcast copy": np.broadcast_to(v[0], (8, 8)).copy(),
    }


def test_arrays_share_a_key_only_when_every_order_reading_agrees():
    """The exact oracle for the layout part of the key, over every pair of forms
    holding equal values (round 18, r18s2's probe_forms). A key that merges two
    forms some callee reads differently serves one the other's result:
    `np.ravel(x, order='A')` reads in Fortran order only for an F-CONTIGUOUS
    array, so an F-like strided view and its F-contiguous copy -- which the
    memory-order key of 0fd2cb5's successor merged -- returned [0, 24, ...]
    for the copy (right answer [0, 2, ...])."""
    forms = _forms()
    names = sorted(forms)
    for i, a_name in enumerate(names):
        a = forms[a_name]
        for b_name in names[i + 1:]:
            b = forms[b_name]
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                continue
            if Cash._try_hash_numpy(a) == Cash._try_hash_numpy(b):
                assert _order_readings(a) == _order_readings(b), (
                    f"{a_name} and {b_name} share a key but a callee reads them differently")


@pytest.mark.parametrize("pair", [
    ("C", "C strided copy", lambda f: (f["C strided"], f["C strided copy"])),
    ("reversed", lambda f: (f["reversed"], f["reversed copy"])),
    ("F view", lambda f: (f["F view"], f["F view F-copy"])),
    ("column", lambda f: (f["col"], f["col copy"])),
    ("broadcast", lambda f: (f["broadcast"], f["broadcast copy"])),
], ids=lambda p: p[0])
def test_forms_no_callee_can_tell_apart_still_share_a_key(pair):
    """The other direction, which the fix above must not undo (CAS-123): a view
    and its copy that every order reading agrees on share one entry, or a
    function returning a view makes its caller run twice after every edit."""
    a, b = pair[-1](_forms())
    assert _order_readings(a) == _order_readings(b)
    assert Cash._try_hash_numpy(a) == Cash._try_hash_numpy(b)


def test_a_returned_view_does_not_rerun_its_caller(tmp_path):
    """r17s4's shape, across three fresh processes: 1, 0, 0 executions."""
    import os
    import subprocess
    import sys
    import textwrap
    (tmp_path / "job.py").write_text(textwrap.dedent("""
        import sys, time
        import numpy as np
        import cash

        @cash.cache
        def upstream(n):
            time.sleep(0.2)                       # @cash:assume-safe
            arr = np.arange(2 * n, dtype=float).reshape(n, 2)
            return arr[:, 0], arr[:, 1]

        @cash.cache
        def downstream(t, v):
            print("RAN", file=sys.stderr)        # @cash:assume-safe
            time.sleep(0.2)                       # @cash:assume-safe
            return float(t @ v)

        downstream(*upstream(1000))
    """), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / ".cash")
    runs = []
    for _ in range(3):
        out = subprocess.run([sys.executable, "-W", "ignore", str(tmp_path / "job.py")],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        runs.append(out.stderr.count("RAN"))
    assert runs == [1, 0, 0], runs
