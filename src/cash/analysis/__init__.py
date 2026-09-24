"""Static analysis of the code cash caches.

Shared by ``@cash.cache`` and the notebook path: which names a statement reads
and writes (``code_analyzer``), the ``# @cash:`` comment annotations
(``annotations``), whether a statement is safe to cache
(``cacheability``, ``cacheability_decision``), and what a decorated
function's code does (``purity_analyzer``, ``purity_flow``). Nothing here
imports ``cash.notebook``.
"""
