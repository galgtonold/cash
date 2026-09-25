"""When a value is cached, and when it is written past the RAM tier.

One object holds the whole rule, so the tiered backend, the notebook's
statement store, the config and ``cash info`` cannot disagree about it.

Past RAM (`decide`, `decide_rebuild`):

* an entry someone asked to keep (``@cash.cache``, ``@cash:persist``) is kept;
* otherwise it is kept when restoring it beats recomputing it by
  ``min_savings_pct`` of the compute, as the fitted cost model predicts, and it
  took at least ``compute_floor_s`` to compute;
* and only while it is worth the disk it takes (`value_policy`).

A notebook statement's value, in any tier (`too_cheap_to_store`,
`refuses_value`):

* a statement that took less than ``store_floor_s`` gets no entry at all;
* a value whose predicted restore exceeds the larger of ``restore_budget_s``
  and ``(1 - min_savings_pct)`` of its compute is not kept, only its
  metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from cash import cost_model

from ._base import store_seconds
from .value_policy import worth_its_bytes

if TYPE_CHECKING:
    from cash.config import CashConfig

__all__ = [
    "COMPUTE_FLOOR_S",
    "MIN_SAVINGS_PCT",
    "RESTORE_BUDGET_S",
    "STORE_FLOOR_S",
    "Decision",
    "PersistencePolicy",
    "restore_kind",
]

#: Nothing that computes faster than this is persisted past RAM. Not because
#: disk I/O is slower than recomputing (a small entry restores in well under a
#: millisecond) but because such statements buy almost nothing back and cost a
#: file each: lowering it to 10 ms on a real notebook restored six more
#: statements, saved no time, and grew the cache 355-fold.
COMPUTE_FLOOR_S = 0.1

#: The fraction of the compute a restore has to save.
MIN_SAVINGS_PCT = 0.20

#: A notebook statement that computes faster than this gets no cache entry,
#: not even a metadata-only one, so the next lookup is a fast clean miss
#: rather than a read that only finds "recompute". The default of
#: ``min_execution_time_to_cache_seconds``.
STORE_FLOOR_S = 0.01

#: The restore time a notebook value may always take, whatever its compute:
#: the fixed overhead of a cheap restore (opening a file) must not refuse a
#: trivial statement. The default of ``min_cache_fixed_budget_seconds``.
RESTORE_BUDGET_S = 0.05


def restore_kind(backend: Any) -> str:
    """The `cost_model` kind a notebook value is restored from: ``"ram"`` when
    the backend's first tier holds values in memory, else ``"disk"``."""
    tiers = getattr(backend, "backends", None)
    first = tiers[0] if tiers else backend
    return "ram" if getattr(first, "cost_kind", None) == "ram" else "disk"


class Decision(NamedTuple):
    """Whether one entry goes past RAM, and why not when it does not."""

    persist: bool
    #: Why it stayed in RAM: ``"compute"``, ``"bytes"`` or ``"replaced_in_cell"``;
    #: None when it is persisted, or when nothing was decided.
    skipped: str | None = None
    #: The bytes the entry was judged by, its referenced results included.
    weight: int = 0
    #: A ``"bytes"`` refusal the user should hear about from this entry, rather
    #: than from the entry that refers to it.
    report: bool = False


@dataclass(frozen=True)
class PersistencePolicy:
    """The persistence rule, with its tunables."""

    compute_floor_s: float = COMPUTE_FLOOR_S
    min_savings_pct: float = MIN_SAVINGS_PCT
    store_floor_s: float = STORE_FLOOR_S
    restore_budget_s: float = RESTORE_BUDGET_S

    @classmethod
    def from_config(cls, config: CashConfig) -> PersistencePolicy:
        return cls(
            min_savings_pct=float(config.min_cache_savings_pct),
            store_floor_s=float(config.min_execution_time_to_cache_seconds),
            restore_budget_s=float(config.min_cache_fixed_budget_seconds),
        )

    def describe(self) -> str:
        """One line for ``cash info``."""
        return (
            f"cost model ({self.compute_floor_s:g}s compute floor, {self.min_savings_pct:.0%} savings required; "
            f"notebook statements: {self.store_floor_s:g}s store floor, {self.restore_budget_s:g}s restore budget)"
        )

    def too_cheap_to_store(self, compute_s: float) -> bool:
        """Is a notebook statement that took *compute_s* too cheap for any entry?

        *compute_s* is wall clock, so it charges the statement for any
        scheduling stall too, and a coarse clock can report exactly 0.0 for an
        instantaneous statement; 0 is below the floor like any other.
        """
        return compute_s < self.store_floor_s

    def restore_budget(self, compute_s: float) -> float:
        """The longest restore a notebook value computed in *compute_s* may take:
        ``max(restore_budget_s, (1 - min_savings_pct) * compute_s)``. The fixed
        budget keeps cheap statements; the ratio takes over once compute is large
        enough to dominate, so a restore never costs most of a long compute."""
        return max(self.restore_budget_s, (1.0 - self.min_savings_pct) * compute_s)

    def refuses_value(self, compute_s: float, restore_s: float) -> bool:
        """Should a notebook value that took *compute_s* and restores in
        *restore_s* be kept as metadata only, in every tier?"""
        return compute_s > 0 and restore_s > self.restore_budget(compute_s)

    def pays_to_restore(
        self, compute_s: float, size_bytes: int, *, type_name: str = "", backend_kind: str = "disk"
    ) -> bool:
        """Does restoring *size_bytes* beat recomputing for *compute_s*?

        ``compute_s - restore > min_savings_pct * compute_s``, with the restore
        time predicted by `cost_model` for the value's type (the slowest family
        when *type_name* is unknown) on *backend_kind*.
        """
        if compute_s < self.compute_floor_s:
            return False
        restore = cost_model.estimated_restore_time(type_name, size_bytes, backend_kind)
        return compute_s - restore > self.min_savings_pct * compute_s

    def decide(
        self,
        key: str,
        metadata: dict[str, Any],
        *,
        backend_kind: str,
        deferred: bool = False,
        override: Callable[[float, int], bool] | None = None,
    ) -> Decision:
        """Should the entry *metadata* describes be written past RAM now?

        *deferred* marks a value the same cell replaces later; the end-of-cell
        pass (`decide_rebuild`) judges the last one. *override* replaces the
        cost model for an entry that carries no cost-model family.
        """
        size = metadata.get("size", 0) or 0
        cap_size = size or metadata.get("cost_model_size_bytes", 0)
        if metadata.get("force_persist"):
            return Decision(True, weight=cap_size)
        if deferred:
            return Decision(False, "replaced_in_cell", cap_size)
        if metadata.get("decorator_entry"):
            # The caller decided already; only the tiers' size caps still apply.
            return Decision(True, weight=cap_size)
        compute_s = store_seconds(metadata)
        if metadata.get("cost_model_family") is not None:
            pays = self.pays_to_restore(
                compute_s,
                metadata.get("cost_model_size_bytes", size),
                type_name=metadata.get("cost_model_type_name", ""),
                backend_kind=backend_kind,
            )
        elif override is not None:
            pays = override(compute_s, size)
        else:
            pays = self.pays_to_restore(compute_s, size, backend_kind=backend_kind)
        if not pays:
            return Decision(False, "compute", cap_size)

        # An entry is weighed with the entries it refers to (``call_ref_bytes``):
        # its own may be a few KB while what it restores is hundreds of MB. A
        # ``referenced`` entry is therefore not judged on its own, since
        # refusing it would leave its referrer restoring a reference to
        # nothing -- unless its size is only an estimate
        # (``value_bytes_estimated``), when nothing else weighs it. Either way
        # its referrer is the one that reports a refusal.
        weight = cap_size + int(metadata.get("call_ref_bytes") or 0)
        referenced = bool(metadata.get("referenced"))
        if referenced and metadata.get("value_bytes_estimated"):
            weight = int(metadata.get("value_bytes") or cap_size)
            if not worth_its_bytes(weight, compute_s):
                return Decision(False, "bytes", weight)
            return Decision(True, weight=weight)
        if not referenced and not worth_its_bytes(weight, compute_s):
            return Decision(False, "bytes", weight, report=True)
        return Decision(True, weight=weight)

    def decide_rebuild(self, metadata: dict[str, Any], rebuild_s: float, *, backend_kind: str) -> Decision:
        """Should a RAM-only notebook value be persisted, now that rebuilding
        it is known to cost *rebuild_s* (its whole upstream chain, not just
        its own statement)? Only a value with a cost-model family is judged."""
        if metadata.get("metadata_only") or metadata.get("cost_model_family") is None:
            return Decision(False)
        size = metadata.get("cost_model_size_bytes", metadata.get("size", 0))
        if not self.pays_to_restore(
            rebuild_s, size, type_name=metadata.get("cost_model_type_name", ""), backend_kind=backend_kind
        ):
            return Decision(False)
        weight = (metadata.get("size") or size) + int(metadata.get("call_ref_bytes") or 0)
        if not metadata.get("force_persist") and not worth_its_bytes(weight, rebuild_s):
            return Decision(False, "bytes", weight, report=True)
        return Decision(True, weight=weight)
