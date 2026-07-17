"""Perpetual-miss guard: stop serialising a statement that can never hit (CAS-172).

**The shape this bounds.** Five independent user-testing rounds each surfaced a
new instance of one recurring failure: some input hashes *unstably* across runs,
so the statement's cache key differs every run, so it never hits — yet cash still
pays the (large) serialisation on every run. The cache can never pay the user
back, and the statement is net-negative forever. Known instances: CAS-165 (bare
fit on a DataFrame, -25 s), CAS-166 (>8 MiB content-hash sampling destabilises
keys across restarts), CAS-171 (a ``make_classification``-derived frame poisons
downstream caching, -7.9 s). We have conceded we cannot enumerate the causes, so
this module bounds the *consequence* regardless of cause.

**What is and is not guarded.** The guard fires on the perpetual-MISS
*signature* only: identical source, a cache key that keeps changing, zero hits.
It deliberately does NOT guard on raw net-negativity. A statement can be
net-negative *in-session* and still be the product's core value — an expensive
cell that saves 25 s across a kernel restart pays its serialisation back exactly
once, on the run that matters. Net-negative-in-session is that statement's
normal, healthy state. The discriminator is key CHURN, not cost.

**What the guard does and does not stop.**

* STOPS serialising — writes are the expensive half, and the wasted half.
* KEEPS hashing and KEEPS the lookup — both are cheap, and the lookup is what
  lets a statement recover on its own if its key later stabilises onto an entry
  that already exists.
* Persists only the *verdict* (guarded / not), and only when it FLIPS. The hot
  path never touches disk: CAS-149 removed a per-cell fsync that cost 8-12 ms a
  cell, and this must not reintroduce one under a new name. The churn counter is
  in-memory-only for exactly that reason — persisting it would mean a write per
  cell. The cost is that a session which accumulates fewer than
  ``GUARD_AFTER_CONSECUTIVE_CHURN_MISSES`` misses before a restart starts over;
  the cost of the alternative is the fsync we already paid once to delete.
* Re-probes periodically (see ``REPROBE_EVERY_N_RUNS``). A guard with no escape
  hatch is a new bug, not a fix.

**Why the re-probe is load-bearing.** Once we stop writing, a key that later
stabilises has no entry to hit — the guard would permanently condemn a statement
that would have cached fine (a user pins a seed, or a frame drops back under the
sampling threshold). So every Nth run one write is allowed through; the run after
it hits, which un-guards the statement. Two triggers, both cheap:

* *periodic* — every ``REPROBE_EVERY_N_RUNS`` runs, unconditionally. This is the
  safety net, and it is the only thing that catches a key oscillating over a
  small set (where the key never equals the immediately preceding one, so the
  repeat trigger below never sees it).
* *key repeat* — a guarded statement whose key equals the previous run's key has
  visibly stabilised. Probing at once recovers it in two runs rather than up to
  ``REPROBE_EVERY_N_RUNS``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_STORE_FILENAME = "_miss_guard.json"
# Bumping this invalidates every persisted verdict (they are re-learned).
_STORE_VERSION = 1

# Number of CONSECUTIVE key-churn misses (each run producing a cache key
# different from the previous run's, with no hit in between) before we stop
# serialising.
#
# Chosen conservatively, and the asymmetry is why: a WRONG guard costs a missed
# speedup, which is the failure mode this project refuses to ship; a guard that
# is too SLOW costs bounded waste — N serialisations, and only N, because the
# verdict then persists. So we buy evidence with the cheap currency. Five
# consecutive distinct keys with zero hits means five runs in which nothing the
# statement could have cached was ever reachable again. An interactive workflow
# that edits an upstream cell five times running, never once re-running the same
# state, is unusual — and if it happens, a single repeat run hits and resets the
# counter to zero before the guard ever fires. The known instances (CAS-165/166/
# 171) churn on EVERY run, so they reach five within five run-alls of an
# otherwise stable notebook. That gap is the discriminator.
GUARD_AFTER_CONSECUTIVE_CHURN_MISSES = 5

# Runs between blind re-probes once guarded (one write allowed through).
# Amortises the wasted serialisation to 1/10th — 90% of the waste removed —
# while bounding how long a statement whose key has silently stabilised can stay
# wrongly condemned. The key-repeat trigger normally recovers it in two runs;
# this is the net underneath that.
REPROBE_EVERY_N_RUNS = 10

# Shown on the badge (the existing ``skipped_reason`` field, same surface the
# size-aware skip uses) so a user can see the statement stopped caching AND why.
GUARD_SKIP_REASON = (
    f"Perpetual cache miss: {GUARD_AFTER_CONSECUTIVE_CHURN_MISSES} consecutive runs of "
    "this statement produced a different cache key and never hit, so serialising it "
    "can never pay back. Value not saved (lineage still tracked); cash keeps looking "
    f"it up and re-probes every {REPROBE_EVERY_N_RUNS} runs in case the key "
    "stabilises (CAS-172)."
)


def resolve_cache_dir(backend: Any) -> str | None:
    """Find the on-disk cache directory behind *backend*, or None.

    Walks a backend chain (``TieredBackend`` / ``CascadingBackend`` expose
    ``.backends``) breadth-first and returns the first real ``cache_dir``. None
    means there is nowhere to persist — a pure in-memory backend, which has no
    restart to survive anyway, so the guard degrades to session-scoped.

    Deliberately duck-typed and defensive: ``cash_instance`` is a ``MagicMock``
    in a good number of tests, and a mock answers every ``getattr`` with another
    mock. The ``isinstance`` checks are what make that return None instead of a
    mock masquerading as a path.
    """
    seen: set[int] = set()
    queue = [backend]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        cache_dir = getattr(current, "cache_dir", None)
        if isinstance(cache_dir, str) and cache_dir:
            return cache_dir
        inner = getattr(current, "backends", None)
        if isinstance(inner, (list, tuple)):
            queue.extend(inner)
    return None


@dataclass
class _Record:
    """Per-source-hash miss-guard state. In-memory except ``guarded``."""

    last_key: str
    churn_misses: int = 0
    guarded: bool = False
    runs_since_probe: int = 0
    probe_now: bool = False


class MissGuard:
    """Learns which statements can never hit, and stops them serialising.

    Keyed by ``source_hash`` (``sha256`` of the statement source), which is what
    makes "IDENTICAL source" literal: any edit to the statement is a different
    key here and starts over from zero evidence.
    """

    def __init__(self, cache_dir: str | None) -> None:
        self._path = os.path.join(cache_dir, _STORE_FILENAME) if cache_dir else None
        self._records: dict[str, _Record] = {}
        self._loaded = False

    # -- persistence ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Read persisted verdicts once per session, lazily.

        Best-effort by construction: a missing, unreadable, corrupt, or
        future-versioned store leaves the guard empty, which means every
        statement serialises — the pre-guard behaviour. The guard is a
        performance optimisation, so its failure mode must be "no optimisation",
        never "no cache".
        """
        if self._loaded:
            return
        self._loaded = True
        if not self._path:
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            logger.debug("[MISS_GUARD] no readable verdict store at %s", self._path)
            return
        if not isinstance(doc, dict) or doc.get("version") != _STORE_VERSION:
            return
        guarded = doc.get("guarded")
        if not isinstance(guarded, list):
            return
        for source_hash in guarded:
            if isinstance(source_hash, str):
                # ``last_key=""`` matches no real key, so the first run of the
                # new session reads as churn rather than as a stabilised key.
                self._records[source_hash] = _Record(last_key="", guarded=True)

    def _persist(self) -> None:
        """Write the guarded set. Called ONLY when a verdict flips.

        Never per cell — that is the CAS-149 fsync-per-cell regression under a
        new name. A flip happens a handful of times in a notebook's whole life.
        Atomic via ``os.replace`` so a crashed write can't leave a torn file for
        the next session to choke on; no ``fsync``, because losing the last
        verdict to a hard kernel kill only costs re-learning it.
        """
        if not self._path:
            return
        doc = {
            "version": _STORE_VERSION,
            "guarded": sorted(sh for sh, rec in self._records.items() if rec.guarded),
        }
        tmp_path = f"{self._path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            os.replace(tmp_path, self._path)
        except OSError:
            logger.debug("[MISS_GUARD] verdict persistence failed", exc_info=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # -- the state machine ----------------------------------------------

    def observe(self, source_hash: str, cache_key: str, hit: bool) -> None:
        """Record one lookup outcome for *source_hash*.

        Call once per run of a statement that actually performed a lookup.
        """
        self._ensure_loaded()
        rec = self._records.get(source_hash)
        if rec is None:
            # First sighting: record the baseline key and nothing else. A cold
            # run MUST serialise — that is the entire product — so churn is only
            # ever counted against a key we have already seen.
            self._records[source_hash] = _Record(last_key=cache_key)
            return

        rec.probe_now = False

        if hit:
            # The key matched an entry: this statement can pay back, whatever it
            # did before. Zero the evidence and release any guard. This is also
            # how a re-probe completes its recovery.
            rec.last_key = cache_key
            rec.churn_misses = 0
            if rec.guarded:
                rec.guarded = False
                rec.runs_since_probe = 0
                self._persist()
            return

        if cache_key != rec.last_key:
            rec.churn_misses += 1
        elif rec.guarded:
            # Same key as last run, still missing: the key has stabilised but we
            # stopped writing, so there is nothing on disk to hit. Probe now.
            rec.probe_now = True
            rec.runs_since_probe = 0
        # A repeated key on an UNGUARDED statement is a miss with a stable key —
        # a stale entry (TTL / changed file dep), not key instability. That is a
        # legitimate recompute-and-recache workflow, so it is not counted and the
        # counter is left where it is.
        rec.last_key = cache_key

        if not rec.guarded and rec.churn_misses >= GUARD_AFTER_CONSECUTIVE_CHURN_MISSES:
            rec.guarded = True
            rec.runs_since_probe = 0
            self._persist()

        # Cadence: the run on which the guard FIRES is guarded run #1 (it is the
        # first run whose write we suppress), so a blind probe lands on guarded
        # runs #R, #2R, ... — one write every R runs, counted from the moment we
        # stopped writing.
        if rec.guarded and not rec.probe_now:
            rec.runs_since_probe += 1
            if rec.runs_since_probe >= REPROBE_EVERY_N_RUNS:
                rec.probe_now = True
                rec.runs_since_probe = 0

    def should_serialise(self, source_hash: str) -> bool:
        """False only for a guarded statement on a non-probe run."""
        self._ensure_loaded()
        rec = self._records.get(source_hash)
        if rec is None or not rec.guarded:
            return True
        return rec.probe_now

    def is_guarded(self, source_hash: str) -> bool:
        """True once the verdict has flipped, probe run or not."""
        self._ensure_loaded()
        rec = self._records.get(source_hash)
        return rec is not None and rec.guarded
