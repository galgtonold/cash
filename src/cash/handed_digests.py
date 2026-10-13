"""Content digests of a call's arguments, handed from the notebook to the
decorator for the one call a statement is.

``df = clean(raw)`` with a ``@cash.cache`` ``clean`` hashed ``raw`` three
times: the statement's in-place-change fingerprint before it ran, the
decorator's key, and the fingerprint after. For a 4.2M-row polars frame that
was 3 x 0.4-1.1 s around a 1.2 s body. When the statement is nothing but that
call (`call_effects._call_shape`), only names are read between the
fingerprint and the key: the digest the one took is the one the other would
take. And when the call is a hit, its body never runs, so nothing between
the key and the binding of its result can change an argument: the
fingerprint after is the one before.

The notebook offers what it took (`offer`) while the statement runs; the
decorator takes it (`claim`) only for a call made directly by that
statement's own code -- not by a function the statement called, whose body
may have changed the argument first -- and only once, and says whether the
body ran (`Offer.ran_body`). Held per thread: another thread's call is never
the statement's.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

from .install_paths import is_cash_path

__all__ = ["Offer", "claim", "offer", "withdraw"]


class Offer:
    """The digests one statement took of its names before it ran."""

    __slots__ = ("claimed", "digests", "filename", "ran_body")

    def __init__(self, filename: str, digests: dict[int, tuple[Any, str]]) -> None:
        self.filename = filename
        #: ``id(value) -> (value, digest)``; each value held, so no id is reused.
        self.digests = digests
        self.claimed = False
        #: Set by the call that claimed it: whether its body ran.
        self.ran_body: bool | None = None

    def digest(self, value: Any) -> str | None:
        """The digest taken of this very *value*, or None."""
        found = self.digests.get(id(value))
        return found[1] if found is not None and found[0] is value else None


_local = threading.local()


def offer(filename: str, digests: dict[int, tuple[Any, str]]) -> Offer:
    """Offer *digests* to a call the statement compiled under *filename* makes."""
    made = Offer(filename, dict(digests))
    _local.offer = made
    return made


def withdraw() -> None:
    _local.offer = None


def claim() -> Offer | None:
    """The offer, for the decorator call being made now, when the statement's
    own code makes it: the first frame past cash's is the statement's
    module-level code. Once: a second call, or one nested in the first, gets
    None."""
    made = getattr(_local, "offer", None)
    if made is None or made.claimed:
        return None
    frame = sys._getframe(1)
    while frame is not None and is_cash_path(frame.f_code.co_filename):
        frame = frame.f_back
    if frame is None or frame.f_code.co_filename != made.filename or frame.f_code.co_name != "<module>":
        return None
    made.claimed = True
    return made
