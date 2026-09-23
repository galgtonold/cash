"""Static analysis of the code cash caches.

Shared by ``@cash.cache`` and the notebook path: which names a statement reads
and writes (``code_analyzer``), the ``# @cash:`` comment annotations
(``annotations``), and whether a statement is safe to cache
(``cacheability``, ``cacheability_decision``). Nothing here imports
``cash.notebook``.
"""
