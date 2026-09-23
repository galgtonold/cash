"""When a value is written past the RAM tier.

One object holds the whole rule, so the tiered backend, the config and
``cash info`` cannot disagree about it:

* an entry someone asked to keep (``@cash.cache``, ``@cash:persist``) is kept;
* otherwise it is kept when restoring it beats recomputing it by
  ``min_savings_pct`` of the compute, as the fitted cost model predicts, and it
  took at least ``compute_floor_s`` to compute;
* and only while it is worth the disk it takes (`value_policy`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from cash import cost_model

from .value_policy import worth_its_bytes

if TYPE_CHECKING:
    from cash.config import CashConfig

__all__ = ["COMPUTE_FLOOR_S", "MIN_SAVINGS_PCT", "Decision", "PersistencePolicy"]

#: Nothing that computes faster than this is persisted past RAM. Not because
#: disk I/O is slower than recomputing (a small entry restores in well under a
#: millisecond) but because such statements buy almost nothing back and cost a
#: file each: lowering it to 10 ms on a real notebook restored six more
#: statements, saved no time, and grew the cache 355-fold.
COMPUTE_FLOOR_S = 0.1

#: The fraction of the compute a restore has to save.
MIN_SAVINGS_PCT = 0.20


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
    """The persistence rule, with its two tunables."""

    compute_floor_s: float = COMPUTE_FLOOR_S
    min_savings_pct: float = MIN_SAVINGS_PCT

    @classmethod
    def from_config(cls, config: CashConfig) -> PersistencePolicy:
        return cls(min_savings_pct=float(config.min_cache_savings_pct))

    def describe(self) -> str:
        """One line for ``cash info``."""
        return f"cost model ({self.compute_floor_s:g}s compute floor, {self.min_savings_pct:.0%} savings required)"

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
        compute_s = metadata.get("execution_time", 0) or 0
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
