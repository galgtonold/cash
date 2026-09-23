"""The parts of ``@cash.cache``, behind the `cash.core.Cash` front.

`Cash` (in ``cash/core.py``) owns the registries and the wrappers; the modules
here hold what a cached call does, one concern each: how code, captures,
globals and arguments become its key (``code_identity``, ``code_args``,
``closure_fold``, ``globals_fold``, ``arg_hashing``, ``frozen``, ``rng``,
``file_deps``), what happens on a call (``runtime``, ``store``,
``iterators``), and what the user is told (``explain``, ``stored_keys``,
``reporting``, ``purity_checks``).
"""
