"""The C watcher fast path (`cash.value_watch`, `cash._cwatch`): a value
whose content the watchers prove unchanged gets its earlier digest back,
and that digest is always the one hashing it again gives."""
