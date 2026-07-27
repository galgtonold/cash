# Clean fixture

<!--
Resolved against the synthetic tree in _fixtures/claims/src/, not the real
src/ tree -- see check_page(FIXTURES / "clean.md", src_root=CLEAN_SRC_ROOT) in
test_claims_lib.py. Pinning this fixture to the real cash/core.py used to mean
editing a 163-line method in a 4,583-line real file turned this unit test red
in the hard-gate docs-parity job.
-->

<!-- claim: cash/config.py:CashConfig.compress -->
Entries can be stored compressed.

<!-- claim: cash/notebook/cache_key.py:compute_cache_key @37576d6b -->
A statement is keyed on its code plus the lineage of every input it reads.

<!-- claim: cash/core.py:Cash.cache @b8318e6f, cash/config.py:CashConfig.max_cache_size == None -->
The cache is unbounded unless you set a size limit.

<!-- claim: cash/config.py:CashConfig broad="the claim is about the config object as a whole" -->
Every field can be set from an environment variable.
