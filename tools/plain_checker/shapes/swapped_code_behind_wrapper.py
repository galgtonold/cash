"""A wrapper a call returns around a function whose ``__code__`` is swapped.

``wrapped = deco(base)`` closes over ``base`` (or holds it as ``__wrapped__``).
Run the call cell alone after the swap, then Run All: plain Python wraps the
redefined ``base``; a kept wrapper still held the swapped one and gave 30 for
6 under floors0. ``functools.partial`` is the control: it was never kept.
"""

from tools.plain_checker import notebook

_SLOW = "__import__('time').sleep(0.02)"
_SWAP = f"base.__code__ = (lambda x: ({_SLOW}, x * 10)[1]).__code__"
_PLAN = ["all", 2, "all", "restart", "all", "all"]

SHAPES = [
    notebook(
        "wraps-swap-slow",
        [
            "import functools, time\ndef base(x):\n    time.sleep(0.02)\n    return x * 2",
            "def deco(f):\n    @functools.wraps(f)\n    def wr(*a):\n        return f(*a)\n    return wr\nwrapped = deco(base)",
            "v = wrapped(3)",
            "w = [v, 'r']",
            _SWAP,
        ],
        plan=_PLAN,
    ),
    notebook(
        "closure-swap-slow",
        [
            "import time\ndef base(x):\n    time.sleep(0.02)\n    return x * 2",
            "def mk(f):\n    def wr(x):\n        return f(x)\n    return wr\nwrapped = mk(base)",
            "v = wrapped(3)",
            "w = [v, 'r']",
            _SWAP,
        ],
        plan=_PLAN,
    ),
    notebook(
        "partial-swap-slow",
        [
            "import functools, time\ndef base(x, y):\n    time.sleep(0.02)\n    return x * y",
            "p = functools.partial(base, 2)",
            "v = p(3)",
            "w = [v, 'r']",
            f"base.__code__ = (lambda x, y: ({_SLOW}, x + y)[1]).__code__",
        ],
        plan=_PLAN,
    ),
]
