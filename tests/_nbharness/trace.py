"""The upstream decision trace, as the ``upstream_trace`` fixture returns it."""


# =============================================================================
# UPSTREAM DECISION TRACE (debugging the simulation/re-execution engine)
# =============================================================================
#
# The simulation engine runs inside the kernel subprocess, and not every module
# logger's output is surfaced through IOPub into get_raw_output (notably the
# `simulator` logger never appears). `cash.notebook._trace.trace_event` writes
# structured JSONL to the file named by CASH_TRACE_FILE, which the cold kernel
# inherits from this process's environment — bulletproof, capture-quirk-free.
#
# Events currently emitted (see _trace.py call sites):
#   simulate_enter      {cell_idx, reassigned, mutated, method_receivers}
#   broken_after_pass2  {broken, tainted}     # lineage/mismatch staleness
#   broken_after_guard  {broken}              # + stale-value guard
#   schedule_reexec     {stmt}                # producer scheduled to re-run
#
# NOTE: requires a cold kernel (the default). The opt-in warm kernel
# (CASH_TEST_REUSE_KERNEL=1) boots once per worker before the env is set, so it
# will not pick up CASH_TRACE_FILE.


class TraceResult:
    """Queryable view over the captured trace records.

    ``run_all`` and ``rerun`` split the records at the ``__run_all_done__``
    boundary so you can inspect only what the re-run decided.
    """

    def __init__(self, records: list[dict]):
        self.records = records
        split = next((i for i, r in enumerate(records) if r.get("event") == "__run_all_done__"), len(records))
        self.run_all = records[:split]
        self.rerun = records[split + 1 :]

    def events(self, name: str, *, phase: str = "rerun") -> list[dict]:
        recs = {"rerun": self.rerun, "run_all": self.run_all, "all": self.records}[phase]
        return [r for r in recs if r.get("event") == name]

    def scheduled(self, *, phase: str = "rerun") -> list[str]:
        return [r["stmt"] for r in self.events("schedule_reexec", phase=phase)]

    def __repr__(self) -> str:
        return f"TraceResult(run_all={len(self.run_all)}, rerun={len(self.rerun)})"
