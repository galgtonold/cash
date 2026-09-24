"""What a cached computation depends on at run time.

Shared by ``@cash.cache`` and the notebook path: the files it reads
(``file_tracker``, fed by ``read_events`` and ``reader_patches``, and
``file_dep_snapshot``), the code it calls (``function_tracker``,
``module_symbols``) and the random state it draws from (``randomness``).
Nothing here imports ``cash.notebook``.
"""
