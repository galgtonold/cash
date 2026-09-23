"""Randomness: what a statement draws, what that does to the RNG, and how it
flows into lineage.

* :mod:`.detect` -- finding unseeded draws in source, and the warnings.
* :mod:`.state` -- capturing and restoring RNG state across a cache hit.
* :mod:`.lineage` -- RNG state as hidden lineage variables and seed epochs.

The names below are what the rest of cash uses; everything else is internal
to its submodule.
"""

from .detect import (
    CashRandomnessWarning,
    RandomnessDetector,
    check_and_warn_randomness,
    describe_random_call,
    get_drawing_rng_modules,
    get_entropy_reseed_modules,
    get_seeding_rng_modules,
    warn_stale_estimator_fit,
    warn_stale_randomness,
    warn_unseeded_estimator_fit,
)
from .lineage import (
    entropy_write_lineage,
    hidden_lineage_reads,
    hidden_lineage_writes,
    hidden_write_lineage,
    observed_rng_reads,
    publish_seed_epochs,
    rng_lineage_fingerprint,
    rng_virtual_var,
    seed_cells_not_yet_run,
    seed_epoch_component,
    seed_epochs,
)
from .state import (
    capture_object_rng_states,
    capture_rng_state,
    restore_object_rng_states,
    restore_rng_state,
    rng_carrier_kind,
    rng_modules_changed,
)

__all__ = [
    "CashRandomnessWarning",
    "RandomnessDetector",
    "capture_object_rng_states",
    "capture_rng_state",
    "check_and_warn_randomness",
    "describe_random_call",
    "entropy_write_lineage",
    "get_drawing_rng_modules",
    "get_entropy_reseed_modules",
    "get_seeding_rng_modules",
    "hidden_lineage_reads",
    "hidden_lineage_writes",
    "hidden_write_lineage",
    "observed_rng_reads",
    "publish_seed_epochs",
    "restore_object_rng_states",
    "restore_rng_state",
    "rng_carrier_kind",
    "rng_lineage_fingerprint",
    "rng_modules_changed",
    "rng_virtual_var",
    "seed_cells_not_yet_run",
    "seed_epoch_component",
    "seed_epochs",
    "warn_stale_estimator_fit",
    "warn_stale_randomness",
    "warn_unseeded_estimator_fit",
]
