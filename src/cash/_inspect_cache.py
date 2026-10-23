"""What ``inspect.getattr_static`` keeps of the classes cash looks up.

On CPython 3.12.0-3.12.3, ``getattr_static`` keeps each class's MRO in a
plain ``lru_cache`` (``inspect._shadowed_dict_from_mro_tuple``), so a class
cash looked an attribute up on stayed alive after the notebook made it again
(``class C`` run again), up to 128 of them. Later Pythons key that cache
weakly. The entries are only a speed-up, so cash empties it when a
statement's analyses end (`forget_static_lookups`).
"""

from __future__ import annotations

import inspect

#: The strong cache, on the Pythons that have it; None elsewhere.
_STRONG_CACHE = getattr(inspect, "_shadowed_dict_from_mro_tuple", None)

__all__ = ["forget_static_lookups"]


def forget_static_lookups() -> None:
    """Let go of the classes ``inspect.getattr_static`` keeps alive."""
    cache = _STRONG_CACHE
    if cache is not None and cache.cache_info().currsize:
        cache.cache_clear()
