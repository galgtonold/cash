# Clean fixture

<!-- claim: cash/config.py:CashConfig.compress -->
Entries can be stored compressed.

<!-- claim: cash/notebook/cache_key.py:compute_cache_key @deadbeef -->
A statement is keyed on its code plus the lineage of every input it reads.

<!-- claim: cash/core.py:Cash.cache @0badcafe, cash/config.py:CashConfig.max_cache_size == None -->
The cache is unbounded unless you set a size limit.

<!-- claim: cash/config.py:CashConfig broad="the claim is about the config object as a whole" -->
Every field can be set from an environment variable.
