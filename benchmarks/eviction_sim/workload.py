"""Synthetic notebook-project workload with lineage-keyed entries.

A project holds several notebooks sharing one cache directory.  A notebook
is a list of cells; a cell is a list of statements; a statement reads a few
earlier statements' outputs.  A statement's cache key is
hash(slot, code version, input keys) -- exactly cash's lineage rule -- so
editing a statement changes its key AND every key downstream of it.

The session driver issues logical operations (restart, run cell, edit,
undo).  The *engine* turns them into cache requests the way cash does:

* running a cell               -> every statement is looked up, even when the
                                  kernel already holds its current value (a
                                  re-run restores from the cache)
* an input absent or stale     -> looked up too; an input the kernel holds at
                                  its current key is used as is
* miss                         -> materialise the inputs first (recursive: an
                                  upstream miss after a restart cascades),
                                  then compute and offer the result to the
                                  cache.

The session ops are generated independently of the policy under test, so
every policy sees the same user behaviour for a given seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Stmt:
    nb: int
    cell: int
    idx: int
    inputs: list  # global stmt ids (earlier)
    compute: float  # seconds
    size: int  # serialized bytes
    loop_iters: int = 0  # >0: a loop statement, one entry per iteration
    version: int = 0
    history: list = field(default_factory=lambda: [0])


@dataclass
class Params:
    n_notebooks: int = 4
    cells: tuple = (15, 35)
    stmts_per_cell: tuple = (1, 3)
    # Calibrated with calibrate.py on the cold runs of an overhead sweep
    # (_rerun_sweep.py over ref_notebooks.txt), 2026-09-13: 9 notebooks, 553
    # computed statements. Compute p50 15 ms, p90 0.82 s, p99 8 s. Sizes
    # are bimodal (scalars/small objects vs frames): p50 13 KB, p75 40 MB,
    # p90 490 MB. The two are coupled through the MIX, not within a mode:
    # ~20% of cheap statements but ~58% of disk-eligible (>= 0.1 s) ones
    # produce a large value, while within the eligible set log-corr(size,
    # compute) is only 0.02. So the chance of a large value rises with
    # compute (a logistic in log-compute from p_large to p_large_hi, centred
    # on p_large_pivot). The first published sweep used an uncoupled v1
    # model, recoverable with: compute_median=0.012, compute_sigma=2.6,
    # size_small_median=2e3, size_small_sigma=2.0, size_large_median=60e6,
    # size_large_sigma=1.3, p_large=0.3, p_large_hi=0.3. Its expensive
    # statements were mostly tiny (eligible p50 5.4 KB vs 16 MB real).
    # Edit behaviour below (edits_per_day, p_undo, ...) is NOT calibrated --
    # it is an informed guess.
    compute_median: float = 0.015
    compute_sigma: float = 2.8
    size_small_median: float = 1e3
    size_small_sigma: float = 2.5
    size_large_median: float = 150e6
    size_large_sigma: float = 1.2
    p_large: float = 0.2
    p_large_hi: float = 0.65
    p_large_pivot: float = 0.1
    size_scale: float = 1.0  # multiply every size (workload archetypes)
    loop_cell_frac: float = 0.08
    loop_iters: tuple = (20, 200)
    days: int = 40
    edits_per_day: tuple = (15, 60)
    p_undo: float = 0.15
    p_run_below: float = 0.35
    p_midday_restart: float = 0.04
    p_switch_notebook: float = 0.3
    p_morning_runall: float = 0.6
    restore_bps: float = 400e6  # deserialize throughput
    restore_floor: float = 0.002
    persist_floor: float = 0.1  # cash's compute floor for persisting
    seed: int = 0


class Project:
    def __init__(self, p: Params):
        self.p = p
        rng = self.rng = random.Random(p.seed)
        self.stmts: list[Stmt] = []
        self.nb_stmts: list[list[int]] = []
        self.nb_cells: list[list[list[int]]] = []
        for nb in range(p.n_notebooks):
            ids, cells = [], []
            for c in range(rng.randint(*p.cells)):
                cell = []
                is_loop = rng.random() < p.loop_cell_frac and c > 1
                for s in range(1 if is_loop else rng.randint(*p.stmts_per_cell)):
                    sid = len(self.stmts)
                    k = min(len(ids), rng.choice([0, 1, 1, 1, 2, 2, 3]))
                    # prefer recent producers, sometimes reach far back
                    inputs = []
                    for _ in range(k):
                        j = len(ids) - 1 - min(len(ids) - 1, int(rng.expovariate(1 / 3)))
                        inputs.append(ids[j])
                    compute = min(1800.0, max(0.0002, p.compute_median * math.exp(p.compute_sigma * rng.gauss(0, 1))))
                    steep = 1 / (1 + math.exp(-2.0 * (math.log(compute) - math.log(p.p_large_pivot))))
                    if rng.random() < p.p_large + (p.p_large_hi - p.p_large) * steep:
                        size = p.size_large_median * math.exp(p.size_large_sigma * rng.gauss(0, 1))
                    else:
                        size = p.size_small_median * math.exp(p.size_small_sigma * rng.gauss(0, 1))
                    size = int(min(4e9, max(32, size * p.size_scale)))
                    iters = rng.randint(*p.loop_iters) if is_loop else 0
                    if iters:
                        compute = compute / 4  # per iteration
                        size = max(200, size // 50)  # per iteration
                    st = Stmt(nb, c, s, sorted(set(inputs)), compute, size, iters)
                    self.stmts.append(st)
                    ids.append(sid)
                    cell.append(sid)
                cells.append(cell)
            self.nb_stmts.append(ids)
            self.nb_cells.append(cells)
        self._key_cache: dict[int, tuple] = {}

    # -- keys ----------------------------------------------------------------
    def invalidate_keys(self):
        self._key_cache.clear()

    def key(self, sid):
        k = self._key_cache.get(sid)
        if k is None:
            st = self.stmts[sid]
            k = hash((sid, st.version, tuple(self.key(i) for i in st.inputs)))
            self._key_cache[sid] = k
        return k

    def entries(self, sid):
        """(key, size, compute, slot) of every cache entry a statement produces."""
        st = self.stmts[sid]
        base = self.key(sid)
        if st.loop_iters:
            return [(hash((base, i)), st.size, st.compute, (sid, i)) for i in range(st.loop_iters)]
        return [(base, st.size, st.compute, (sid,))]

    def live_keys(self, frontier):
        live = set()
        for nb, ids in enumerate(self.nb_stmts):
            for sid in ids:
                if self.stmts[sid].cell < frontier[nb]:
                    for e in self.entries(sid):
                        live.add(e[0])
        return live


class Engine:
    """Runs logical ops against a RAM tier + a disk tier, the way cash's
    default TieredBackend wires them:

    * every request goes RAM -> disk; a disk hit is promoted into RAM
      (whatever its size -- read-repair has no size gate);
    * a computed value is always offered to RAM (if it clears the 10 ms
      caching floor) and to disk only past the cost-model gate (0.1 s floor,
      restore must save >= 20% of compute);
    * a restart empties RAM and the kernel; disk persists.

    ``ram`` may be None (disk-only experiments) and so may ``disk``.
    """

    def __init__(self, proj: Project, disk, ram=None):
        self.proj = proj
        self.disk = disk
        self.ram = ram
        self.p = proj.p
        self.mem: dict[int, int] = {}  # sid -> key materialised in the kernel
        self.time = 0.0  # user-visible seconds
        self.compute_s = 0.0
        self.restore_s = 0.0
        self.requests = 0
        self.hits = 0
        self.ram_hits = 0
        self.disk_hits = 0

    def restore_cost(self, size):
        return self.p.restore_floor + size / self.p.restore_bps

    def ram_cost(self, size):
        # the RAM tier hands back a deep copy
        return 0.0003 + size / 2e9

    def admitted(self, size, compute):
        return compute >= self.p.persist_floor and compute - self.restore_cost(size) > 0.2 * compute

    def _lookup(self, key, size, compute, slot):
        self.requests += 1
        meta = {"slot": slot, "nb": self.proj.stmts[slot[0]].nb, "loop": len(slot) > 1}
        if self.ram is not None and self.ram.access(key, size, compute - self.ram_cost(size), meta):
            self.ram_hits += 1
            self.hits += 1
            c = self.ram_cost(size)
            self.time += c
            self.restore_s += c
            # keep the disk policy's view of recency in step: cash's TieredBackend
            # does NOT touch the disk tier on a RAM hit, so neither do we.
            return True
        if self.disk is not None and self.disk.access(key, size, compute - self.restore_cost(size), meta):
            self.disk_hits += 1
            self.hits += 1
            c = self.restore_cost(size)
            self.time += c
            self.restore_s += c
            if self.ram is not None:
                self.ram.insert(key, size, compute - self.ram_cost(size), meta)
            return True
        return False

    def _store(self, key, size, compute, slot):
        meta = {"slot": slot, "nb": self.proj.stmts[slot[0]].nb, "loop": len(slot) > 1}
        # RAM admission = Gate 0 (10 ms floor) + Gate A against the RAM copy cost
        if self.ram is not None and compute >= 0.01 and self.ram_cost(size) <= max(0.05, 0.8 * compute):
            self.ram.insert(key, size, compute - self.ram_cost(size), meta)
        if self.disk is not None and self.admitted(size, compute):
            self.disk.insert(key, size, compute - self.restore_cost(size), meta)

    def need(self, sid, force_lookup=False):
        """Make statement ``sid``'s current value present in the kernel.

        ``force_lookup``: the user ran this statement's cell, so cash looks it
        up even if the kernel already holds the current value (it does -- a
        re-run restores from the cache rather than trusting the namespace).
        Inputs are only fetched when absent or stale.
        """
        proj = self.proj
        k = proj.key(sid)
        if not force_lookup and self.mem.get(sid) == k:
            return
        ents = proj.entries(sid)
        all_hit = True
        for key, size, compute, slot in ents:
            if not self._lookup(key, size, compute, slot):
                all_hit = False
                break
        if not all_hit:
            for i in proj.stmts[sid].inputs:
                self.need(i)
            for key, size, compute, slot in ents:
                # modelled as: the whole loop reruns on any miss (equally
                # pessimistic for every policy)
                self.time += compute
                self.compute_s += compute
                self._store(key, size, compute, slot)
        self.mem[sid] = k

    def run_cell(self, nb, c):
        for sid in self.proj.nb_cells[nb][c]:
            self.need(sid, force_lookup=True)

    def restart(self):
        self.mem.clear()
        if self.ram is not None:
            self.ram.clear_all()


def session_ops(proj: Project):
    """Yield logical ops.  Deterministic for a given Params.seed, and
    independent of the policy under test (so every policy sees the same
    user behaviour)."""
    p = proj.p
    rng = random.Random(p.seed + 1)
    n_cells = [len(c) for c in proj.nb_cells]
    frontier = [max(3, n // 3) for n in n_cells]
    active = 0
    for day in range(p.days):
        if rng.random() < p.p_switch_notebook:
            active = rng.randrange(p.n_notebooks)
        yield ("restart",)
        yield ("frontier", list(frontier))
        if rng.random() < p.p_morning_runall:
            for c in range(frontier[active]):
                yield ("run", active, c)
        else:
            yield ("run", active, frontier[active] - 1)
        for _ in range(rng.randint(*p.edits_per_day)):
            # work mostly near the frontier, sometimes revisit upstream
            back = int(rng.expovariate(1 / 3))
            c = max(0, frontier[active] - 1 - back)
            cell = proj.nb_cells[active][c]
            sid = rng.choice(cell)
            st = proj.stmts[sid]
            if rng.random() < p.p_undo and len(st.history) > 1:
                st.version = st.history[-2]
                st.history.append(st.version)
            else:
                st.version = max(st.history) + 1
                st.history.append(st.version)
            proj.invalidate_keys()
            yield ("edited",)
            yield ("run", active, c)
            if rng.random() < p.p_run_below:
                for cc in range(c + 1, frontier[active]):
                    yield ("run", active, cc)
            else:
                for cc in range(c + 1, min(frontier[active], c + 1 + rng.randint(0, 2))):
                    yield ("run", active, cc)
            if rng.random() < p.p_midday_restart:
                yield ("restart",)
                yield ("run", active, frontier[active] - 1)
            # the notebook grows
            if rng.random() < 0.08 and frontier[active] < n_cells[active]:
                frontier[active] += 1
                yield ("frontier", list(frontier))
                yield ("run", active, frontier[active] - 1)
