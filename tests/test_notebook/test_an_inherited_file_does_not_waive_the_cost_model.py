"""Only a statement that READS a file skips the cost model -- not everything after it.

The store waived the "too cheap to cache" floor and the restore-cost check
for any statement with file dependencies ("file I/O is inherently
expensive"). But the dependencies it looked at included every file the
statement's INPUTS were built from, so everything downstream of a load was
exempt: a value whose restore takes longer than recomputing it was cached
anyway, and served as a hit (round 28, r28s5: ~400 MiB frames restoring in
~10 s against ~4 s of compute, "1/1 cache hits" on the badge).
"""


def _waivers(cash_magics, tmp_path):
    from cash.notebook.statement.processor import StatementProcessor

    orig = StatementProcessor._should_skip_large_object_caching
    seen = {}

    def spy(self, captured_vars, execution_time, force_persist, has_file_dependencies=False):
        for name in captured_vars:
            seen[name] = has_file_dependencies
        return orig(self, captured_vars, execution_time, force_persist, has_file_dependencies=has_file_dependencies)

    StatementProcessor._should_skip_large_object_caching = spy
    try:
        path = tmp_path / "data.csv"
        path.write_text("a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(2000)), encoding="utf-8")
        cash_magics.cash("", f"import pandas as pd\ndf = pd.read_csv(r'{path}')")
        # Real work, so it clears the too-cheap floor and reaches the check.
        cash_magics.cash(
            "", "big = pd.concat([df] * 50, ignore_index=True).assign(w=sum(i * i for i in range(400_000)))"
        )
    finally:
        StatementProcessor._should_skip_large_object_caching = orig
    return seen


def test_a_statement_over_a_loaded_frame_gets_the_restore_check(cash_magics, tmp_path):
    seen = _waivers(cash_magics, tmp_path)
    assert seen.get("big") is False, seen


def test_the_statement_that_reads_the_file_is_still_waived(cash_magics, tmp_path):
    seen = _waivers(cash_magics, tmp_path)
    assert seen.get("df") is True, seen
