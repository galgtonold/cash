"""What a builtin does with what it is handed: one table, read by every
analysis that lets ``len(x)`` or ``sorted(x)`` pass without following it.

Each analysis asks a slightly different question of the same call ("is
anything put into ``x``?", "does the result hold ``x``?", "may this run
before the loop does?"). The answers were kept as one hand-written list of
names per module, and the lists drifted: a name added to one was missing
from the next, and two lists under the same sentence held different names.
Here every builtin has ONE row saying what it really does, and each question
is a set derived from the rows. A new builtin is added as a row; a new
question is a new set at the foot of this module, never a list of names
somewhere else.

Every name is matched as text, so a caller that can see the user's bindings
checks that the name still means the builtin.
"""

from __future__ import annotations

__all__ = [
    "ADVANCES",
    "APPLIES",
    "ARGUMENT_BUILTINS",
    "CALL_WHAT_THEY_ARE_HANDED",
    "CHANGE_NOTHING_HANDED",
    "CLASS",
    "DRAWS",
    "FOLLOW_FROM_ARGUMENTS",
    "HAND_NOTHING_ON",
    "HOLDS",
    "HOLD_NOTHING_HANDED",
    "ITEM",
    "KEY",
    "LAZY",
    "LOOKUP",
    "NO_EFFECT",
    "NUMBER",
    "NUMBER_MAKERS",
    "PART",
    "PRINTS",
    "PROCESS",
    "PUT_NOTHING_IN",
    "STORE_NOTHING",
    "WITHOUT_OUTPUT",
    "traits",
]

#: Changes its argument: ``next(it)`` moves the iterator on.
ADVANCES = "advances"
#: Calls a function it is handed on every item (``map(f, xs)``): what the
#: call does is what ``f`` does.
APPLIES = "applies"
#: May be handed a function to call (``sorted(xs, key=f)``, ``iter(f, end)``).
KEY = "key"
#: Takes the items out of an iterator it is handed, now or as its result is
#: iterated: ``list(gen)`` leaves ``gen`` empty.
DRAWS = "draws"
#: Its result is a one-shot iterator, not a value that can be read twice.
LAZY = "lazy"
#: Its result is a new container or iterator that holds its arguments or
#: their items: ``list(xs)[0]`` is ``xs[0]``.
HOLDS = "holds"
#: Hands back one of the items it is handed (``max(rows)``), or what they add
#: up to, which for lists holds their items (``sum(lists, [])``).
ITEM = "item"
#: Hands back a part of the object itself, through which the object can be
#: changed: ``vars(o)['k'] = 1``, ``getattr(o, 'rows').append(1)``,
#: ``getattr(o, 'append')(1)``.
PART = "part"
#: Hands back the class of its argument, an object that was there before.
CLASS = "class"
#: Looks an attribute up by name, which may run a property or a
#: ``__getattr__`` of the user's.
LOOKUP = "lookup"
#: Writes to the standard output.
PRINTS = "prints"
#: Its answer belongs to this process: an address, or a hash that is salted
#: for text.
PROCESS = "process"
#: Its answer is a number.
NUMBER = "number"

#: One row per builtin: everything it does beyond reading its arguments and
#: handing back a new value. An empty row is a builtin that does only that.
_ROWS: dict[str, str] = {
    # Numbers, text and truth values.
    "abs": NUMBER,
    "ascii": "",
    "bin": "",
    "bool": NUMBER,
    "bytes": DRAWS,
    "callable": "",
    "chr": "",
    "complex": NUMBER,
    "divmod": NUMBER,
    "float": NUMBER,
    "format": "",
    "hex": "",
    "int": NUMBER,
    "isinstance": "",
    "issubclass": "",
    "len": NUMBER,
    "oct": "",
    "ord": NUMBER,
    "pow": "",
    "range": "",
    "repr": "",
    "round": NUMBER,
    "str": "",
    "all": DRAWS,
    "any": DRAWS,
    "bytearray": DRAWS,
    # One of the items, or their total.
    "max": f"{DRAWS} {ITEM} {KEY}",
    "min": f"{DRAWS} {ITEM} {KEY}",
    "sum": f"{DRAWS} {ITEM} {NUMBER}",
    "next": f"{ADVANCES} {DRAWS} {ITEM}",
    # A new container of the items.
    "dict": f"{DRAWS} {HOLDS}",
    "frozenset": f"{DRAWS} {HOLDS}",
    "list": f"{DRAWS} {HOLDS}",
    "set": f"{DRAWS} {HOLDS}",
    "slice": HOLDS,
    "sorted": f"{DRAWS} {HOLDS} {KEY}",
    "tuple": f"{DRAWS} {HOLDS}",
    # An iterator over the items.
    "enumerate": f"{DRAWS} {HOLDS} {LAZY}",
    "filter": f"{APPLIES} {DRAWS} {HOLDS} {LAZY}",
    "iter": f"{HOLDS} {KEY} {LAZY}",
    "map": f"{APPLIES} {DRAWS} {HOLDS} {LAZY}",
    "reversed": f"{HOLDS} {LAZY}",
    "zip": f"{DRAWS} {HOLDS} {LAZY}",
    # The object itself, looked into.
    "getattr": f"{LOOKUP} {PART}",
    "hasattr": LOOKUP,
    "hash": PROCESS,
    "id": PROCESS,
    "type": CLASS,
    "vars": PART,
    # Output.
    "print": PRINTS,
}

_TRAITS: dict[str, frozenset[str]] = {name: frozenset(row.split()) for name, row in _ROWS.items()}


def traits(name: str) -> frozenset[str]:
    """What the builtin *name* does beyond reading its arguments; empty for
    one that does only that, and for a name that has no row."""
    return _TRAITS.get(name, frozenset())


def _doing(*wanted: str) -> frozenset[str]:
    """The builtins that do one of *wanted*."""
    return frozenset(name for name, row in _TRAITS.items() if row.intersection(wanted))


def _never(*refused: str) -> frozenset[str]:
    """The builtins that do none of *refused*."""
    return frozenset(name for name, row in _TRAITS.items() if not row.intersection(refused))


#: Every builtin with a row.
ARGUMENT_BUILTINS = frozenset(_TRAITS)

#: Builtins that write nothing a reader could see: no output.
WITHOUT_OUTPUT = _never(PRINTS)

#: Builtins that change nothing they are handed and hand back no part of it
#: to change it through. ``map(f, xs)`` is one: what ``f`` does to an item
#: is the caller's to follow.
CHANGE_NOTHING_HANDED = _never(ADVANCES, PART)

#: The same, and no function among the arguments is called on the rest.
HAND_NOTHING_ON = _never(ADVANCES, PART, APPLIES)

#: Builtins a container can be handed without anything being put into it.
#: ``next(it)`` takes from it; ``getattr(c, 'append')`` is the way in.
PUT_NOTHING_IN = _never(PART)

#: Builtins with no effect of their own: nothing handed is changed, nothing
#: is printed, and no attribute of the user's is run. A call to one may be
#: skipped, or run a second time, and nothing differs.
NO_EFFECT = _never(ADVANCES, PART, LOOKUP, PRINTS)

#: Builtins whose answer follows from their arguments alone, in any process.
FOLLOW_FROM_ARGUMENTS = _never(ADVANCES, PART, LOOKUP, PRINTS, PROCESS)

#: Builtins that keep no reference to their arguments anywhere but in what
#: they hand back, and hand back no container of them.
STORE_NOTHING = _never(ADVANCES, PART, HOLDS)

#: Builtins whose result holds nothing of what they are handed: a new
#: number, text, truth value or range.
HOLD_NOTHING_HANDED = _never(HOLDS, ITEM, PART, CLASS, PRINTS)

#: Builtins that call a function they are handed.
CALL_WHAT_THEY_ARE_HANDED = _doing(APPLIES, KEY)

#: Builtins whose answer is a number.
NUMBER_MAKERS = _doing(NUMBER)
