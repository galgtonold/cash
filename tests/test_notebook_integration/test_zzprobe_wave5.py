"""Adversarial probes, wave 5 (2026-07-02): randomness, annotations, cell
magics, stdlib accumulators, cycles, decorator persistence across restart.

 1. test_unseeded_random_not_served_stale    — ADJUDICATED (CAS-230): freezing
        is intended; asserts the # @cash:no-cache opt-out redraws instead.
 2. test_seeded_random_reproducible          — seeded cell: identical output
        across run_alls (cached or re-executed, both deterministic).
 3. test_allow_random_annotation_caches      — # @cash:allow-random suppresses
        the warning; the value stays frozen: second run_all reprints the SAME.
 4. test_ttl_annotation_expiry               — # @cash:ttl=1: immediate rerun
        cached (no miss), post-expiry rerun recomputes (miss marker).
 5. test_capture_cellmagic_graceful          — %%capture cell: correct values,
        edit propagation through the captured cell.
 6. test_writefile_cellmagic_chain           — %%writefile edited -> reader
        cell must see the new content on run_all.
 7. test_counter_defaultdict_rerun_idempotent — Counter.update / defaultdict
        loop accumulators: isolated re-run idempotence.
 8. test_cyclic_structure_cells_restart      — self-referential list crossing
        cells + rerun + restart/persist: no crash, correct.
 9. test_decorator_disk_hit_survives_restart — @cash.cache (>1s) in a
        notebook: after kernel restart an edited call cell re-executes but the
        decorator serves the disk entry (side-effect log stays at 1 line).
"""

import uuid

import pytest

pytestmark = [pytest.mark.timeout(150)]

MISS = "Executing (cache miss)"


def _p(path) -> str:
    return str(path).replace("\\", "/")


def _val(out, tag):
    return out.split(tag)[-1].strip().splitlines()[0]


def test_unseeded_random_not_served_stale(nb_runner):
    """ADJUDICATED (CAS-230): freezing is intended; the opt-out is the contract.

    This probe asserted plain-kernel semantics — a fresh draw each run. That is
    NOT what cash does, and deliberately so: unseeded randomness is pervasive in
    the notebooks cash targets, and redrawing every run would cascade-invalidate
    everything cached downstream of it.

    The probe did surface a real bug, just not the one it named. Freezing is
    defensible only because you can opt out, and the opt-out was broken: the
    warning names `# @cash:no-cache` as the way to "re-run it every time", but
    it only switched off caching — which was never what froze the value. The
    freeze comes from the RNG rewind, so the statement re-executed and redrew
    the identical number. Fixed by making `no-cache` skip the rewind too.

    Rewritten to assert the opt-out rather than the default. The full contract
    lives in `test_rng_unseeded_contract.py`.
    """
    nb_runner.create_notebook([
        "import random",
        "# @cash:no-cache\nr = random.random()\nprint('r=', r)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    v1 = _val(nb_runner.get_output(2), "r=")
    nb_runner.run_all()
    v2 = _val(nb_runner.get_output(2), "r=")
    assert v1 != v2, (
        f"# @cash:no-cache must redraw each run: {v1} == {v2} "
        f"(it has to switch off the RNG rewind, not just caching)"
    )


def test_seeded_random_reproducible(nb_runner):
    nb_runner.create_notebook([
        "import random\nrandom.seed(42)\nvals = [random.random() for _ in range(3)]",
        "print('v0=', round(vals[0], 6))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    v1 = _val(nb_runner.get_output(2), "v0=")
    nb_runner.run_all()
    v2 = _val(nb_runner.get_output(2), "v0=")
    assert v1 == v2, f"seeded random not reproducible across run_alls: {v1} vs {v2}"


def test_allow_random_annotation_caches(nb_runner):
    nb_runner.create_notebook([
        "import random",
        "# @cash:allow-random\nrv = random.random()\nprint('rv=', rv)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    v1 = _val(nb_runner.get_output(2), "rv=")
    nb_runner.run_all()
    v2 = _val(nb_runner.get_output(2), "rv=")
    assert v1 == v2, (
        f"# @cash:allow-random ignored: annotated unseeded cell re-drew on an "
        f"unchanged rerun ({v1} vs {v2})"
    )


def test_ttl_annotation_within_ttl_cached(nb_runner):
    # ttl=60 comfortably exceeds kernel round-trip latency, so the second
    # run_all is unambiguously within the ttl.
    # print lives in its own cell: it sits below the 10ms cost floor and
    # legitimately re-executes every run, so it must not pollute the
    # miss-attribution for the annotated statement.
    nb_runner.create_notebook([
        "# @cash:ttl=60\nbig5 = sum(i * i for i in range(3000000))",
        "print('big=', big5)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "big=" in nb_runner.get_output(2)

    nb_runner.run_all()
    raw = nb_runner.get_raw_output(1)
    assert MISS not in raw, (
        f"# @cash:ttl=60 statement recomputed within ttl: "
        f"{[l for l in raw.splitlines() if MISS in l]}"
    )


def test_ttl_annotation_expiry(nb_runner):
    import time as _time
    nb_runner.create_notebook([
        "# @cash:ttl=1\nbig6 = sum(i * i for i in range(3000000))",
        "print('big=', big6)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "big=" in nb_runner.get_output(2)

    _time.sleep(1.5)
    nb_runner.run_all()  # expired -> must recompute
    raw = nb_runner.get_raw_output(1)
    assert MISS in raw, (
        "# @cash:ttl=1 did not expire: no recompute 1.5s after the entry was "
        "written (annotation ignored?)"
    )


def test_capture_cellmagic_graceful(nb_runner):
    nb_runner.create_notebook([
        "x5 = 5",
        "%%capture cap5\nprint('hidden')\ny5 = x5 * 2",
        "print('y5=', y5)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y5= 10" in nb_runner.get_output(3)
    nb_runner.run_all()
    assert "y5= 10" in nb_runner.get_output(3)

    nb_runner.set_cell_source(1, "x5 = 7")
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "y5= 14" in out, (
        f"edit did not propagate through a %%capture cell: {out!r}"
    )


def test_writefile_cellmagic_chain(nb_runner, tmp_path):
    snip = _p(tmp_path / "snippet5.txt")
    nb_runner.create_notebook([
        f"%%writefile {snip}\nhello v1",
        f"body5 = open('{snip}').read()\nprint('b=', body5.strip())",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "b= hello v1" in nb_runner.get_output(2)

    nb_runner.set_cell_source(1, f"%%writefile {snip}\nhello v2")
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "b= hello v2" in out, (
        f"reader served stale content after the %%writefile cell was edited "
        f"(run_all!): {out!r}"
    )


def test_counter_defaultdict_rerun_idempotent(nb_runner):
    nb_runner.create_notebook([
        "from collections import Counter, defaultdict\nwords5 = ['a', 'b', 'a']",
        "counts5 = Counter()\ncounts5.update(words5)\n"
        "dd5 = defaultdict(int)\nfor w5 in words5:\n    dd5[w5] += 1\n"
        "print('c=', dict(counts5), 'd=', dict(dd5))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    expected = "c= {'a': 2, 'b': 1} d= {'a': 2, 'b': 1}"
    assert expected in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert expected in out, (
        f"Counter/defaultdict accumulators not idempotent on isolated re-run: {out!r}"
    )


def test_cyclic_structure_cells_restart(nb_runner):
    nb_runner.create_notebook([
        "cyc5 = [1, 2]\ncyc5.append(cyc5)",
        "print('n=', len(cyc5), 'self=', cyc5[2] is cyc5)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "n= 3 self= True" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    assert "n= 3 self= True" in nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "n= 3 self= True" in out, (
        f"cyclic structure broke after restart+persist: {out!r}"
    )


def test_decorator_disk_hit_survives_restart(nb_runner, tmp_path):
    log = _p(tmp_path / "slowlog5.txt")
    salt = uuid.uuid4().hex[:8]
    fn_cell = (
        "import cash, time\n"
        "@cash.cache\n"
        f"def slow_add5(x):  # salt {salt}\n"
        "    time.sleep(1.2)\n"
        f"    with open('{log}', 'a') as f:\n"
        "        f.write('call\\n')\n"
        "    return x + 1"
    )
    nb_runner.create_notebook([
        fn_cell,
        "res5 = slow_add5(5)\nprint('res=', res5)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "res= 6" in nb_runner.get_output(2)
    with open(log.replace("/", "\\")) as f:
        assert f.read().count("call") == 1

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    # Force real execution of the call cell (new code) so the DECORATOR path
    # is exercised post-restart, not the notebook statement cache.
    nb_runner.set_cell_source(2, "res5 = slow_add5(5)\nprint('res5b=', res5)")
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "res5b= 6" in out, out
    with open(log.replace("/", "\\")) as f:
        n = f.read().count("call")
    assert n == 1, (
        f"decorator did not serve the disk entry after kernel restart: "
        f"function body ran {n} times total (expected 1)"
    )
