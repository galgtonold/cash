"""Byte-capacity cache policies for trace-driven simulation.

Every policy sees the same interface:

    access(key, size, cost, meta) -> bool      # True = hit
    insert(key, size, cost, meta)              # after a miss, if admitted
    hint(kind, **kw)                           # optional side-channel (supersede, live set)

``cost`` is the NET benefit of a hit in seconds (recompute - restore).
``meta`` carries the entry's slot: the statement position that produced it.
Capacities are in bytes; entries larger than the cap are refused.

``CashCurrent`` and ``CashRAM`` model the byte-cap eviction ``FileBackend``
and ``InMemoryBackend`` shipped with before GDSF ranking (2026-09-13): LRU,
plus the disk tier's size split. They are kept as the baseline the redesign
was measured against; the shipped ranking is now ``GDSF`` (quantized, see
``cash.backends._base.gdsf_value``).
"""

from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict, defaultdict, deque


class Policy:
    name = "base"

    def __init__(self, cap: int):
        self.cap = cap
        self.used = 0
        self.size: dict = {}
        self.now = 0  # logical clock (request count)
        self.evictions = 0
        self.bytes_written = 0

    # --- interface -------------------------------------------------------
    def access(self, key, size, cost, meta) -> bool:
        self.now += 1
        if key in self.size:
            self._on_hit(key, size, cost, meta)
            return True
        self._on_miss(key, size, cost, meta)
        return False

    def insert(self, key, size, cost, meta):
        if size > self.cap or key in self.size:
            return
        self.size[key] = size
        self.used += size
        self.bytes_written += size
        self._on_insert(key, size, cost, meta)
        self._evict_to_fit(protect=key)

    def hint(self, kind, **kw):
        pass

    def remove(self, key):
        """External removal (TTL, clear). Not counted as an eviction."""
        if key in self.size:
            self._on_remove(key)
            self.used -= self.size.pop(key)

    def clear_all(self):
        for k in list(self.size):
            self.remove(k)

    # --- hooks -----------------------------------------------------------
    def _on_hit(self, key, size, cost, meta): ...
    def _on_miss(self, key, size, cost, meta): ...
    def _on_insert(self, key, size, cost, meta): ...
    def _on_remove(self, key): ...
    def _victim(self, protect):
        raise NotImplementedError

    def _evict_to_fit(self, protect=None):
        while self.used > self.cap:
            v = self._victim(protect)
            if v is None:
                break
            self._drop(v)

    def _drop(self, key):
        self.used -= self.size.pop(key)
        self.evictions += 1
        self._on_remove(key)


# ---------------------------------------------------------------------------
# Recency / frequency classics
# ---------------------------------------------------------------------------
class LRU(Policy):
    name = "LRU"

    def __init__(self, cap):
        super().__init__(cap)
        self.od = OrderedDict()

    def _on_hit(self, key, size, cost, meta):
        self.od.move_to_end(key)

    def _on_insert(self, key, size, cost, meta):
        self.od[key] = None

    def _on_remove(self, key):
        self.od.pop(key, None)

    def _victim(self, protect):
        for k in self.od:
            if k != protect:
                return k
        return None


class CashCurrent(Policy):
    """``FileBackend._check_and_evict``: LRU, evicting down to a 90% low
    watermark once over the cap. Entries are split into 'crumbs' (< 0.1% of
    the cap, ``_EVICT_CRUMB_FRACTION``) and the rest, and crumbs are taken
    first whenever they alone can close the gap (``_evict_order``).

    ``crumbs=False`` and ``target=1.0`` are the two ablations."""

    name = "cash-now"
    CRUMB = 0.001

    def __init__(self, cap, target=0.9, crumbs=True):
        super().__init__(cap)
        self.TARGET = target
        self.crumb = max(1, int(cap * self.CRUMB)) if crumbs else 0
        self.small = OrderedDict()
        self.big = OrderedDict()
        self.small_bytes = 0

    def _q(self, key):
        return self.small if self.size[key] < self.crumb else self.big

    def _on_hit(self, key, size, cost, meta):
        self._q(key).move_to_end(key)

    def _on_insert(self, key, size, cost, meta):
        self._q(key)[key] = None
        if self.size[key] < self.crumb:
            self.small_bytes += size

    def _on_remove(self, key):
        if key in self.small:
            self.small.pop(key)
            self.small_bytes -= self._sz_removed
        self.big.pop(key, None)

    def _drop(self, key):
        self._sz_removed = self.size[key]
        super()._drop(key)

    def remove(self, key):
        if key in self.size:
            self._sz_removed = self.size[key]
        super().remove(key)

    def _evict_to_fit(self, protect=None):
        if self.used <= self.cap:
            return
        target = self.cap * self.TARGET
        while self.used > target:
            need = self.used - target
            sb = self.small_bytes - (self.size[protect] if protect in self.small else 0)
            order = (self.small, self.big) if sb >= need else (self.big, self.small)
            victim = None
            for q in order:
                for k in q:
                    if k != protect:
                        victim = k
                        break
                if victim is not None:
                    break
            if victim is None:
                break
            self._drop(victim)


class CashRAM(LRU):
    """``InMemoryBackend._evict_to_byte_cap``: LRU over ``last_access``,
    evicting down to 90% of the cap. The NEW entry is in the candidate list,
    so a value bigger than 0.9 x cap first flushes everything older, then
    itself. Every write is accepted (no size gate). The psutil pressure path
    (``_check_and_evict``) is not modelled: it depends on the host."""

    name = "cash-RAM-now"

    def insert(self, key, size, cost, meta):
        if key in self.size:
            return
        self.size[key] = size
        self.used += size
        self.bytes_written += size
        self._on_insert(key, size, cost, meta)
        if self.used > self.cap:
            target = self.cap * 0.9
            for k in list(self.od):
                if self.used <= target:
                    break
                self._drop(k)


class FIFO(LRU):
    name = "FIFO"

    def _on_hit(self, key, size, cost, meta):
        pass


class LFU(Policy):
    """LFU with LRU tie-break (and in-cache frequency only)."""

    name = "LFU"

    def __init__(self, cap):
        super().__init__(cap)
        self.freq = {}
        self.last = {}
        self.heap = []

    def _push(self, key):
        heapq.heappush(self.heap, (self.freq[key], self.last[key], key))

    def _on_hit(self, key, size, cost, meta):
        self.freq[key] += 1
        self.last[key] = self.now
        self._push(key)

    def _on_insert(self, key, size, cost, meta):
        self.freq[key] = 1
        self.last[key] = self.now
        self._push(key)

    def _on_remove(self, key):
        self.freq.pop(key, None)
        self.last.pop(key, None)

    def _victim(self, protect):
        skipped = []
        while self.heap:
            f, t, k = heapq.heappop(self.heap)
            if k not in self.size or self.freq.get(k) != f or self.last.get(k) != t:
                continue
            if k == protect:
                skipped.append((f, t, k))
                continue
            for s in skipped:
                heapq.heappush(self.heap, s)
            return k
        for s in skipped:
            heapq.heappush(self.heap, s)
        return None


class SLRU(Policy):
    """Segmented LRU: probation (new) + protected (hit at least twice).
    The protected segment holds up to 80% of the bytes."""

    name = "SLRU"
    PROT = 0.8

    def __init__(self, cap):
        super().__init__(cap)
        self.prob = OrderedDict()
        self.prot = OrderedDict()
        self.prot_used = 0

    def _on_hit(self, key, size, cost, meta):
        if key in self.prot:
            self.prot.move_to_end(key)
            return
        self.prob.pop(key)
        self.prot[key] = None
        self.prot_used += self.size[key]
        while self.prot_used > self.cap * self.PROT and len(self.prot) > 1:
            k, _ = self.prot.popitem(last=False)
            self.prot_used -= self.size[k]
            self.prob[k] = None  # demote to probation MRU

    def _on_insert(self, key, size, cost, meta):
        self.prob[key] = None

    def _on_remove(self, key):
        if key in self.prot:
            self.prot.pop(key)
            self.prot_used -= self.size.get(key, 0) if key in self.size else 0
        self.prob.pop(key, None)

    def _drop(self, key):
        sz = self.size[key]
        if key in self.prot:
            self.prot_used -= sz
        super()._drop(key)

    def _victim(self, protect):
        for seg in (self.prob, self.prot):
            for k in seg:
                if k != protect:
                    return k
        return None


class ARC(Policy):
    """Byte-weighted ARC (Megiddo & Modha), ghost lists sized in bytes."""

    name = "ARC"

    def __init__(self, cap):
        super().__init__(cap)
        self.t1, self.t2 = OrderedDict(), OrderedDict()
        self.b1, self.b2 = OrderedDict(), OrderedDict()
        self.b1_bytes = self.b2_bytes = 0
        self.t1_bytes = 0
        self.p = 0.0
        self.gsize = {}

    def _on_hit(self, key, size, cost, meta):
        if key in self.t1:
            self.t1.pop(key)
            self.t1_bytes -= self.size[key]
        else:
            self.t2.pop(key)
        self.t2[key] = None

    def _on_miss(self, key, size, cost, meta):
        if key in self.b1:
            d = max(1.0, self.b2_bytes / max(1, self.b1_bytes)) * size
            self.p = min(self.cap, self.p + d)
        elif key in self.b2:
            d = max(1.0, self.b1_bytes / max(1, self.b2_bytes)) * size
            self.p = max(0.0, self.p - d)

    def _ghost_del(self, key):
        if key in self.b1:
            self.b1.pop(key)
            self.b1_bytes -= self.gsize.pop(key)
        elif key in self.b2:
            self.b2.pop(key)
            self.b2_bytes -= self.gsize.pop(key)

    def _on_insert(self, key, size, cost, meta):
        if key in self.b1 or key in self.b2:
            self._ghost_del(key)
            self.t2[key] = None
        else:
            self.t1[key] = None
            self.t1_bytes += size

    def _on_remove(self, key):
        if key in self.t1:
            self.t1.pop(key)
            self.t1_bytes -= self.size.get(key, 0)
        self.t2.pop(key, None)

    def _drop(self, key):
        sz = self.size[key]
        in_t1 = key in self.t1
        if in_t1:
            self.t1_bytes -= sz
        super()._drop(key)
        g, gb = (self.b1, "b1_bytes") if in_t1 else (self.b2, "b2_bytes")
        g[key] = None
        self.gsize[key] = sz
        setattr(self, gb, getattr(self, gb) + sz)
        for gl, attr in ((self.b1, "b1_bytes"), (self.b2, "b2_bytes")):
            while getattr(self, attr) > self.cap and gl:
                k, _ = gl.popitem(last=False)
                setattr(self, attr, getattr(self, attr) - self.gsize.pop(k))

    def _victim(self, protect):
        prefer_t1 = self.t1_bytes > self.p
        order = (self.t1, self.t2) if prefer_t1 else (self.t2, self.t1)
        for seg in order:
            for k in seg:
                if k != protect:
                    return k
        return None


class S3FIFO(Policy):
    """S3-FIFO (Yang et al., SOSP'23): small FIFO (10%) + main FIFO with
    2-bit frequency + ghost FIFO. Byte-weighted."""

    name = "S3-FIFO"
    SMALL = 0.1

    def __init__(self, cap):
        super().__init__(cap)
        self.S = deque()
        self.M = deque()
        self.where = {}
        self.freq = defaultdict(int)
        self.ghost = OrderedDict()
        self.ghost_bytes = 0
        self.s_bytes = 0

    def _on_hit(self, key, size, cost, meta):
        self.freq[key] = min(3, self.freq[key] + 1)

    def _on_insert(self, key, size, cost, meta):
        self.freq[key] = 0
        if key in self.ghost:
            self.ghost_bytes -= self.ghost.pop(key)
            self.M.append(key)
            self.where[key] = "M"
        else:
            self.S.append(key)
            self.where[key] = "S"
            self.s_bytes += size

    def _on_remove(self, key):
        w = self.where.pop(key, None)
        if w == "S":
            self.s_bytes -= self.size.get(key, 0)
        self.freq.pop(key, None)

    def _drop(self, key):
        if self.where.get(key) == "S":
            self.s_bytes -= self.size[key]
        sz = self.size[key]
        super()._drop(key)
        return sz

    def _victim(self, protect):
        # Lazy deletion: entries may linger in the deques after removal.
        guard = 0
        while guard < 4 * (len(self.S) + len(self.M)) + 8:
            guard += 1
            if self.s_bytes > self.cap * self.SMALL and self.S:
                k = self.S.popleft()
                if self.where.get(k) != "S":
                    continue
                if k == protect:
                    self.S.append(k)
                    continue
                if self.freq[k] > 0:
                    self.s_bytes -= self.size[k]
                    self.M.append(k)
                    self.where[k] = "M"
                    self.freq[k] = 0
                    continue
                self.ghost[k] = self.size[k]
                self.ghost_bytes += self.size[k]
                while self.ghost_bytes > self.cap and self.ghost:
                    _, gs = self.ghost.popitem(last=False)
                    self.ghost_bytes -= gs
                return k
            if self.M:
                k = self.M.popleft()
                if self.where.get(k) != "M":
                    continue
                if k == protect:
                    self.M.append(k)
                    continue
                if self.freq[k] > 0:
                    self.freq[k] -= 1
                    self.M.append(k)
                    continue
                return k
            if self.S:
                self.s_bytes = self.cap  # force S path next loop
                continue
            return None
        return None


class SIEVE(Policy):
    """SIEVE (Zhang et al., NSDI'24): FIFO queue + visited bit + moving hand."""

    name = "SIEVE"

    def __init__(self, cap):
        super().__init__(cap)
        self.q = []  # index 0 = oldest; hand walks from oldest to newest
        self.visited = {}
        self.hand = 0

    def _on_hit(self, key, size, cost, meta):
        self.visited[key] = True

    def _on_insert(self, key, size, cost, meta):
        self.q.append(key)
        self.visited[key] = False

    def _on_remove(self, key):
        self.visited.pop(key, None)

    def _victim(self, protect):
        # compact lazily-removed keys
        if len(self.q) > 2 * len(self.size) + 16:
            live = [k for k in self.q if k in self.size]
            self.hand = 0
            self.q = live
        n = len(self.q)
        if not n:
            return None
        for _ in range(2 * n + 2):
            if self.hand >= len(self.q):
                self.hand = 0
            k = self.q[self.hand]
            if k not in self.size or k == protect:
                self.hand += 1
                continue
            if self.visited.get(k):
                self.visited[k] = False
                self.hand += 1
                continue
            self.q.pop(self.hand)
            return k
        return None


class RRIP(Policy):
    """SRRIP / BRRIP / DRRIP (Jaleel et al., ISCA'10), 2-bit RRPV,
    fully associative with byte capacity.

    SRRIP inserts at RRPV=2 (long re-reference), BRRIP at 3 (distant) except
    1/32 of the time.  Hit -> RRPV=0.  Victim: any RRPV==3 (oldest first),
    else age everyone.  DRRIP duels the two on sampled keys.
    """

    MAX = 3

    def __init__(self, cap, mode="S", seed=1):
        super().__init__(cap)
        self.mode = mode
        self.name = {"S": "SRRIP", "B": "BRRIP", "D": "DRRIP"}[mode]
        self.rrpv = OrderedDict()
        self.rng = random.Random(seed)
        self.psel = 512
        self.leader = {}

    def _leader_of(self, key):
        h = hash(key) % 64
        return "S" if h == 0 else ("B" if h == 1 else None)

    def _insert_rrpv(self, key):
        m = self.mode
        if m == "D":
            ld = self._leader_of(key)
            m = ld or ("S" if self.psel < 512 else "B")
        if m == "S":
            return self.MAX - 1
        return self.MAX - 1 if self.rng.random() < 1 / 32 else self.MAX

    def _on_miss(self, key, size, cost, meta):
        if self.mode == "D":
            ld = self._leader_of(key)
            if ld == "S":
                self.psel = min(1023, self.psel + 1)
            elif ld == "B":
                self.psel = max(0, self.psel - 1)

    def _on_hit(self, key, size, cost, meta):
        self.rrpv[key] = 0

    def _on_insert(self, key, size, cost, meta):
        self.rrpv[key] = self._insert_rrpv(key)

    def _on_remove(self, key):
        self.rrpv.pop(key, None)

    def _victim(self, protect):
        cands = [k for k in self.rrpv if k != protect]
        if not cands:
            return None
        while True:
            for k in cands:
                if self.rrpv[k] >= self.MAX:
                    return k
            for k in cands:
                self.rrpv[k] += 1


# ---------------------------------------------------------------------------
# Cost / size aware
# ---------------------------------------------------------------------------
class GDSF(Policy):
    """GreedyDual-Size-Frequency (Cherkasova'98):
    H = L + freq * cost / size; evict min H; L := H(victim).
    ``cost`` = net seconds saved by a hit."""

    name = "GDSF"

    def __init__(self, cap, use_freq=True):
        super().__init__(cap)
        self.use_freq = use_freq
        if not use_freq:
            self.name = "GD-Size"
        self.L = 0.0
        self.H = {}
        self.freq = {}
        self.cost = {}
        self.heap = []

    def _prio(self, key):
        f = self.freq[key] if self.use_freq else 1
        return self.L + f * max(self.cost[key], 1e-9) / max(self.size[key], 1)

    def _push(self, key):
        self.H[key] = self._prio(key)
        heapq.heappush(self.heap, (self.H[key], self.now, key))

    def _on_hit(self, key, size, cost, meta):
        self.freq[key] += 1
        self._push(key)

    def _on_insert(self, key, size, cost, meta):
        self.freq[key] = 1
        self.cost[key] = cost
        self._push(key)

    def _on_remove(self, key):
        self.H.pop(key, None)
        self.freq.pop(key, None)
        self.cost.pop(key, None)

    def _victim(self, protect):
        skipped = []
        while self.heap:
            h, t, k = heapq.heappop(self.heap)
            if k not in self.size or self.H.get(k) != h:
                continue
            if k == protect:
                skipped.append((h, t, k))
                continue
            for s in skipped:
                heapq.heappush(self.heap, s)
            self.L = h
            return k
        for s in skipped:
            heapq.heappush(self.heap, s)
        return None


class TouchedGDSF(GDSF):
    """GDSF where "the notebook still needs this" counts as a use.

    Before a cell runs, cash's upstream simulator computes the key every
    statement above it would request. ``hint('touch', keys=...)`` re-bases
    those entries to the current clock (as a read would), without counting a
    hit. Superseded entries stop being touched and age out through the
    clock; nothing is ever marked dead, so an entry the simulator cannot
    enumerate (loop iterations, call units, decorator calls) is simply
    ranked as plain GDSF ranks it.
    """

    name = "GDSF+touch"

    def hint(self, kind, **kw):
        if kind != "touch":
            return
        for key in kw["keys"]:
            if key in self.size:
                self._push(key)  # H = L(now) + freq * cost / size


class SampledGDSF(GDSF):
    """GDSF that evicts the lowest-H of ``k`` uniformly sampled entries
    instead of the global minimum (the approach Redis takes for LRU/LFU).

    Models a disk tier that cannot afford to read every entry's header to
    rank them: each eviction reads ``k`` headers. ``L`` only ever rises, since
    a sampled victim can sit below the current clock.
    """

    def __init__(self, cap, k=32, seed=7):
        super().__init__(cap)
        self.k = k
        self.name = f"GDSF-sampled(k={k})"
        self.rng = random.Random(seed)
        self.keys = []  # dense list for O(1) sampling
        self.pos = {}

    def _on_insert(self, key, size, cost, meta):
        super()._on_insert(key, size, cost, meta)
        self.pos[key] = len(self.keys)
        self.keys.append(key)

    def _on_remove(self, key):
        super()._on_remove(key)
        i = self.pos.pop(key, None)
        if i is not None:
            last = self.keys.pop()
            if i < len(self.keys):
                self.keys[i] = last
                self.pos[last] = i

    def _victim(self, protect):
        cands = [k for k in (self.rng.choice(self.keys) for _ in range(min(self.k, len(self.keys)))) if k != protect]
        if not cands:
            others = [k for k in self.keys if k != protect]
            return others[0] if others else None
        v = min(cands, key=lambda k: self.H[k])
        self.L = max(self.L, self.H[v])
        return v


class CostLRU(Policy):
    """LRU whose recency is shifted by a log value-density bonus:

        rank = last_access + tau * ln(benefit_seconds / MB)

    Evict the lowest rank.  Implementable in cash's disk tier with no
    extra syscall at ranking time: write the shifted stamp as the entry's
    mtime (os.utime) on write and on the access flush, and keep ranking by
    scandir.  ``tau`` is in logical requests here (seconds in a real
    deployment): one e-fold of value density buys ``tau`` of recency.
    """

    def __init__(self, cap, tau=2000.0, freq=False):
        super().__init__(cap)
        self.tau = tau
        self.freq_on = freq
        self.name = f"CostLRU(tau={tau:g}{',f' if freq else ''})"
        self.rank = {}
        self.dens = {}
        self.freq = {}
        self.heap = []

    def _push(self, key):
        f = math.log(1 + self.freq[key]) if self.freq_on else 0.0
        r = self.now + self.tau * (math.log(self.dens[key]) + f)
        self.rank[key] = r
        heapq.heappush(self.heap, (r, key))

    def _on_hit(self, key, size, cost, meta):
        self.freq[key] += 1
        self._push(key)

    def _on_insert(self, key, size, cost, meta):
        self.dens[key] = max(cost, 1e-6) / max(size / 1e6, 1e-6)
        self.freq[key] = 0
        self._push(key)

    def _on_remove(self, key):
        self.rank.pop(key, None)
        self.dens.pop(key, None)
        self.freq.pop(key, None)

    def _victim(self, protect):
        skipped = []
        while self.heap:
            r, k = heapq.heappop(self.heap)
            if k not in self.size or self.rank.get(k) != r:
                continue
            if k == protect:
                skipped.append((r, k))
                continue
            for s in skipped:
                heapq.heappush(self.heap, s)
            return k
        for s in skipped:
            heapq.heappush(self.heap, s)
        return None


# ---------------------------------------------------------------------------
# Notebook-structure aware
# ---------------------------------------------------------------------------
class _DeadFirst(Policy):
    """Evict entries judged dead first (LRU among them), then fall back to
    an ``inner`` policy's ranking over everything else.  ``inner`` is any
    Policy class; it is mirrored with an unbounded cap."""

    def __init__(self, cap, inner=LRU, inner_kw=None, age_on_dead=True):
        super().__init__(cap)
        self.g = inner(float("inf"), **(inner_kw or {}))
        self.od = OrderedDict()
        # GDSF ages by raising L to each victim's priority. Whether evicting
        # a DEAD entry should age the live ones too is a modelling choice.
        self.age_on_dead = age_on_dead

    def _dead(self, key):
        raise NotImplementedError

    def _on_hit(self, key, size, cost, meta):
        self.od.move_to_end(key)
        self.g.now = self.now
        self.g._on_hit(key, size, cost, meta)

    def _on_miss(self, key, size, cost, meta):
        self.g.now = self.now

    def _on_insert(self, key, size, cost, meta):
        self.od[key] = None
        self.g.now = self.now
        self.g.size[key] = size
        self.g._on_insert(key, size, cost, meta)

    def _on_remove(self, key):
        self.od.pop(key, None)
        if key in self.g.size:
            self.g._on_remove(key)
            self.g.size.pop(key, None)

    def _victim(self, protect):
        for k in self.od:
            if k != protect and self._dead(k):
                if self.age_on_dead and hasattr(self.g, "L") and hasattr(self.g, "H"):
                    self.g.L = max(self.g.L, self.g.H.get(k, 0))
                return k
        return self.g._victim(protect)


class SupersedeAware(_DeadFirst):
    """Slot generations.  Every entry has a slot (the statement position
    that produced it: notebook, cell, stmt -- or a decorated function +
    call site).  When a slot gets a NEW key, its previous keys become
    superseded.  The newest ``keep`` superseded generations per slot are
    still treated as live, so an undo can hit."""

    def __init__(self, cap, keep=1, inner=LRU, inner_kw=None, label=None, age_on_dead=True):
        super().__init__(cap, inner, inner_kw, age_on_dead)
        self.keep = keep
        self.name = f"Supersede(k={keep})+{label or inner.__name__}"
        self.slot_of = {}
        self.slot_gens = defaultdict(list)  # slot -> keys, oldest..newest

    def _dead(self, key):
        gens = self.slot_gens.get(self.slot_of.get(key))
        if not gens or key not in gens:
            return False
        return gens.index(key) < len(gens) - 1 - self.keep

    def _touch_slot(self, key, meta):
        slot = meta.get("slot") if meta else None
        if slot is None:
            return
        self.slot_of[key] = slot
        gens = self.slot_gens[slot]
        if key in gens:
            gens.remove(key)
        gens.append(key)

    def _on_hit(self, key, size, cost, meta):
        super()._on_hit(key, size, cost, meta)
        self._touch_slot(key, meta)  # an undo revives the old generation

    def _on_insert(self, key, size, cost, meta):
        super()._on_insert(key, size, cost, meta)
        self._touch_slot(key, meta)

    def _on_remove(self, key):
        super()._on_remove(key)
        slot = self.slot_of.pop(key, None)
        if slot is not None and key in self.slot_gens.get(slot, ()):
            self.slot_gens[slot].remove(key)


class HybridGC(SupersedeAware):
    """Dead = outside every notebook's live set AND older than the newest
    ``keep`` generations of its slot.  The live set protects inactive
    notebooks' current results; the generation window protects undo."""

    def __init__(self, cap, keep=1, inner=LRU, inner_kw=None, label=None):
        super().__init__(cap, keep, inner, inner_kw, label)
        self.name = f"Hybrid(k={keep})+{label or inner.__name__}"
        self.live = set()

    def hint(self, kind, **kw):
        if kind == "live":
            self.live = kw["keys"]

    def _dead(self, key):
        return key not in self.live and super()._dead(key)


class OwnerGC(_DeadFirst):
    """The deployable form of generation-awareness for notebooks.

    Each top-level statement entry is tagged at write time with the NOTEBOOK
    that wrote it (loop iterations and call units are not tagged: nothing
    enumerates them). ``hint('liveset', nb=..., keys=...)`` delivers that
    notebook's full current live set -- the key every top-level statement of
    its current source would request. An entry is dead only if all of:

    * it is tagged, and its owner's live set is known;
    * it is not in that live set;
    * it has been out of it for more than ``grace`` of the owner's runs,
      so the version an undo would restore is still protected.

    Everything else is ranked by the inner policy (GDSF, touched: being in a
    live set re-bases an entry as a read would).
    """

    def __init__(self, cap, grace=20, inner=GDSF):
        super().__init__(cap, inner=inner, age_on_dead=False)
        self.grace = grace
        self.name = f"OwnerGC(grace={grace})+{inner.__name__}"
        self.owner = {}  # key -> notebook
        self.live = {}  # notebook -> set of keys
        self.runs = defaultdict(int)
        self.last_live = {}  # key -> owner's run count when last live

    def hint(self, kind, **kw):
        if kind != "liveset":
            return
        nb, keys = kw["nb"], kw["keys"]
        self.runs[nb] += 1
        self.live[nb] = keys
        for k in keys:
            self.last_live[k] = self.runs[nb]
            if k in self.g.size:
                self.g.now = self.now
                self.g._push(k)  # touch: re-base to the current clock

    def _on_insert(self, key, size, cost, meta):
        super()._on_insert(key, size, cost, meta)
        if meta and not meta.get("loop") and meta.get("nb") is not None:
            self.owner[key] = meta["nb"]
            self.last_live[key] = self.runs[meta["nb"]]

    def _on_remove(self, key):
        super()._on_remove(key)
        self.owner.pop(key, None)
        self.last_live.pop(key, None)

    def _dead(self, key):
        nb = self.owner.get(key)
        if nb is None or nb not in self.live or key in self.live[nb]:
            return False
        return self.runs[nb] - self.last_live.get(key, 0) > self.grace


class LiveSetGC(_DeadFirst):
    """Mark-and-sweep over the notebooks' CURRENT sources.  cash's
    NotebookSimulator computes, without executing, the key every statement
    of a notebook would request on a run-all; the union over known
    notebooks is the live set.  Anything outside it can only be hit by an
    undo (or by a notebook cash cannot see).  Fed via hint('live')."""

    def __init__(self, cap, inner=LRU, inner_kw=None, label=None):
        super().__init__(cap, inner, inner_kw)
        self.name = f"LiveGC+{label or inner.__name__}"
        self.live = set()

    def hint(self, kind, **kw):
        if kind == "live":
            self.live = kw["keys"]

    def _dead(self, key):
        return key not in self.live
