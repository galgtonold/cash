"""Fixture module for test_helper_read_globals.py.

Deliberately a SEPARATE module: the defect it pins only appears when the global
is read by a helper whose ``__globals__`` are not the decorated function's.
"""


def _fast(prompt):
    return f"fast:{prompt}"


def _smart(prompt):
    return f"smart:{prompt}"


#: A provider registry -- the shape every LLM client has.
MODELS = {"fast": _fast, "smart": _smart}

THRESHOLD = 7


def complete(prompt, model="fast"):
    """A helper that reads MODELS. This is the whole point.

    The `sorted(MODELS)` matters and is not decoration. A global that is only
    subscripted is folded outright; one PASSED TO A CALL is folded only
    provisionally, and it is the provisional ones that get re-checked after the
    body runs. Without a pass-to-call here the watch entry is never created and
    the defect this module exists to pin does not occur -- the first version of
    these tests passed against the unfixed code for exactly that reason.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; have {sorted(MODELS)}")
    return MODELS[model](prompt)


def over_threshold(n):
    # Same reason: `max` makes THRESHOLD provisional.
    return n > max(THRESHOLD, 0)
