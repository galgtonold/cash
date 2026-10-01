"""The parts of ``@cash.cache``, behind the `cash.core.Cash` front.

`Cash` (in ``cash/core.py``) holds the settings and wires these together; the
modules here hold what a cached call does, one concern each: the function
``@cash.cache`` returns (``wrappers``), how code, captures, globals and
arguments become its key (``function_identity``, ``code_surface``,
``code_refs``, ``code_args``, ``closure_fold``, ``globals_fold``,
``global_reads``, ``global_values``, ``class_data``, ``module_attrs``,
``method_deps``, ``environment_fold``, ``key_values``, ``user_code``,
``arg_hashing``, ``frozen``, ``rng``, ``file_deps``), what happens on a call
(``runtime``, ``store``, ``iterators``), what the user is told (``explain``,
``stored_keys``, ``reporting``, ``purity_checks``, ``run_summary``), and
removing entries (``maintenance``).
"""
