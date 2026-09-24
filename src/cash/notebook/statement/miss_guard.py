"""Perpetual-miss guard: stop serialising a statement that can never hit.

**The shape this bounds.** Five independent rounds of user testing each surfaced a
new instance of one recurring failure: some input hashes *unstably* across runs,
so the statement's cache key differs every run, so it never hits — yet cash still
pays the (large) serialisation on every run. The cache can never pay the user
back, and the statement is net-negative forever. Known instances: a bare
fit on a DataFrame (-25 s); sampled content-hashing of a large file, which
destabilises keys across restarts; a ``make_classification``-derived frame that
poisons downstream caching (-7.9 s). We have conceded we cannot enumerate the
causes, so this module bounds the *consequence* regardless of cause.

**What is and is not guarded.** The guard fires on the perpetual-MISS
*signature* only: identical source, a cache key that keeps changing, zero hits.
It deliberately does NOT guard on raw net-negativity. A statement can be
net-negative *in-session* and still be the product's core value — an expensive
cell that saves 25 s across a kernel restart pays its serialisation back exactly
once, on the run that matters. Net-negative-in-session is that statement's
normal, healthy state. The discriminator is key CHURN, not cost.

Churn alone over-reaches in one direction, though: five upstream edits in a
row churn a key too, and that is an ordinary morning of model tuning.
So the
processor applies the verdict only to a statement whose write is not cheap
next to its compute (``StatementStore.write_is_cheap``): a small, slow
value keeps being written, since its wasted writes cost next to nothing.

**What the guard does and does not stop.**

* STOPS serialising — writes are the expensive half, and the wasted half.
* KEEPS hashing and KEEPS the lookup — both are cheap, and the lookup is what
  lets a statement recover on its own if its key later stabilises onto an entry
  that already exists.
* Persists only the *verdict* (guarded / not), and only when it FLIPS. The hot
  path never touches disk: an earlier change removed a per-cell fsync that cost 8-12 ms a
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

from dataclasses import dataclass

from cash.backends.cache_dir import MISS_GUARD_FILENAME

from ..versioned_json_store import VersionedJsonStore

# Bumping this invalidates every persisted verdict (they are re-learned).
_STORE_VERSION = 2

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
# counter to zero before the guard ever fires. The known instances
# churn on EVERY run, so they reach five within five run-alls of an
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
    "stabilises."
)


class _GuardedStore(VersionedJsonStore[bool]):
    """The source hashes currently guarded: the only part of the guard kept
    across kernels, as ``{source_hash: true}``."""

    FILENAME = MISS_GUARD_FILENAME
    VERSION = _STORE_VERSION
    FIELD = "guarded"
    LOG_TAG = "MISS_GUARD"

    def _load_value(self, value: object) -> bool | None:
        return True if value is True else None

    def guarded(self) -> list[str]:
        self._ensure_loaded()
        return list(self._items)

    def replace(self, guarded: set[str]) -> None:
        """Make *guarded* the whole persisted set."""
        self._ensure_loaded()
        self._items = dict.fromkeys(guarded, True)
        self._write()


@dataclass
class _Record:
    """Per-source-hash miss-guard state. In-memory except ``guarded``."""

    last_key: str
    churn_misses: int = 0
    guarded: bool = False
    runs_since_probe: int = 0
    probe_now: bool = False
    #: The statement's inputs and their lineages at the last lookup, and how
    #: often each changed from one churning run to the next: what the badge
    #: names as the cause.
    last_components: dict | None = None
    changed: dict | None = None
    churned_without_a_change: int = 0


class MissGuard:
    """Learns which statements can never hit, and stops them serialising.

    Keyed by ``source_hash`` (``sha256`` of the statement source), which is what
    makes "IDENTICAL source" literal: any edit to the statement is a different
    key here and starts over from zero evidence.
    """

    def __init__(self, cache_dir: str | None) -> None:
        self._store = _GuardedStore(cache_dir)
        self._records: dict[str, _Record] = {}
        self._loaded = False

    # -- persistence ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Seed the guarded verdicts from disk once per session, lazily.

        Best-effort, like every :class:`VersionedJsonStore`: a missing,
        unreadable, corrupt or future-versioned store leaves the guard empty,
        so every statement serialises. The guard is a performance
        optimisation, so its failure mode must be "no optimisation", never
        "no cache".
        """
        if self._loaded:
            return
        self._loaded = True
        for source_hash in self._store.guarded():
            # ``last_key=""`` matches no real key, so the first run of the
            # new session reads as churn rather than as a stabilised key.
            self._records[source_hash] = _Record(last_key="", guarded=True)

    def _persist(self) -> None:
        """Write the guarded set. Called ONLY when a verdict flips.

        Never per cell: the hot path stays in memory. A flip happens a
        handful of times in a notebook's whole life. No ``fsync``, because
        losing the last verdict to a hard kernel kill only costs re-learning
        it.
        """
        self._store.replace({sh for sh, rec in self._records.items() if rec.guarded})

    # -- the state machine ----------------------------------------------

    def observe(self, source_hash: str, cache_key: str, hit: bool, components: dict | None = None) -> None:
        """Record one lookup outcome for *source_hash*.

        Call once per run of a statement that actually performed a lookup.
        *components*, the statement's inputs and their lineages, let `cause`
        say which of them kept changing.
        """
        self._ensure_loaded()
        rec = self._records.get(source_hash)
        if rec is None:
            # First sighting: record the baseline key and nothing else. A cold
            # run MUST serialise — that is the entire product — so churn is only
            # ever counted against a key we have already seen.
            self._records[source_hash] = _Record(last_key=cache_key, last_components=components)
            return
        if not hit and cache_key != rec.last_key and components is not None and rec.last_components is not None:
            moved = [
                name
                for name in set(components) | set(rec.last_components)
                if components.get(name) != rec.last_components.get(name)
            ]
            if moved:
                rec.changed = rec.changed or {}
                for name in moved:
                    rec.changed[name] = rec.changed.get(name, 0) + 1
            else:
                rec.churned_without_a_change += 1
        if components is not None:
            rec.last_components = components

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

    def cause(self, source_hash: str) -> str | None:
        """What kept changing the key, for the badge, or None if not known."""
        rec = self._records.get(source_hash)
        if rec is None:
            return None
        if rec.changed:
            top = sorted(rec.changed.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
            names = " and ".join(f"`{name}`" for name, _n in top)
            return f"{names} changed each run"
        if rec.churned_without_a_change:
            return "something outside its inputs changed each run (a file it reads, or the code of a function it calls)"
        return None
