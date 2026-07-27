"""Synthetic stand-in; see cash/config.py in this fixture tree for why."""


def compute_cache_key(code: str, lineage: tuple[str, ...]) -> str:
    return f"{code}:{lineage}"
