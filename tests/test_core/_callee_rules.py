"""A third module: the rule a cached function's cached CALLEE reads.

Separate file on purpose. The defect this backs is specifically about a global
that lives outside the cached caller's own module and is read on behalf of a
function further down the call chain -- put the constant in the caller's module
and it is folded by a channel that already worked, so the fixture would pass
against the unfixed code.
"""

THRESHOLD = 20

#: Read by nobody. The over-invalidation control edits this one.
UNUSED_SETTING = "quiet"


def keep(x):
    """A plain, undecorated helper reading the module constant."""
    return (x % 100) < THRESHOLD
