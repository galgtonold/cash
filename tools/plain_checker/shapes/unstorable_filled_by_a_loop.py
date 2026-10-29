"""A loop that changes in place an object cash cannot store.

E10's c5: ``pbar = pyprind.ProgBar(...)`` is not stored (its own parts hold
it, or it is not worth an entry), and each ``for file in ...: ...;
pbar.update()`` changed it in place, so the loop ran every time. A loop like that may be restored; what reads the object
afterwards must still see it as a Run All leaves it.
"""

from tools.plain_checker import notebook

_SETUP = (
    "import sys\nclass Bar:\n    def __init__(self, stream):\n        self.n = 0\n        self.stream = stream\n        self.me = self\n"
    "    def update(self):\n        self.n += 1\n"
)
_CLOCKED = _SETUP.replace("self.n += 1", "self.n += 1\n        self.at = time.perf_counter()").replace(
    "import sys", "import sys, time"
)
_FILL_TWO_LEVELS = (
    "rows = 0\nfor s in (1, 2):\n    for t in (1, 2):\n        for i in range(150000):\n"
    "            rows = rows + i * s * t\n            bar.update()"
)
_FILL = "rows = 0\nfor s in (1, 2):\n    for i in range(300000):\n        rows = rows + i * s\n        bar.update()"

SHAPES = [
    notebook(
        "unstorable-fill-no-reader",
        [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL, "total = rows * 2"],
        names=["rows", "total"],
    ),
    notebook(
        "unstorable-fill-reader-below",
        [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL, "seen = bar.n"],
        names=["rows", "seen"],
    ),
    notebook(
        "unstorable-fill-reader-same-cell",
        [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL + "\nseen = bar.n"],
        names=["rows", "seen"],
    ),
    notebook(
        "unstorable-fill-two-loops-reader-alone",
        [_SETUP, "bar = Bar(sys.stdout)", _FILL, _FILL.replace("rows", "more"), "seen = bar.n"],
        names=["rows", "more", "seen"],
        plan=["all", 4, "restart", "all", 4],
    ),
    notebook(
        "unstorable-fill-reader-added-later",
        [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL, "total = rows * 2"],
        names=["rows", "total", "seen"],
        plan=["all", "all", "restart", "all", 2],
        edits={3: {2: "total = rows * 2\nseen = bar.n"}},
    ),
    notebook(
        "unstorable-fill-reader-in-the-same-loop",
        [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL + "\n    seen = bar.n"],
        names=["rows", "seen"],
    ),
    notebook(
        "unstorable-fill-reader-in-a-function",
        [_SETUP, "def show():\n    return bar.n", "bar = Bar(sys.stdout)\n" + _FILL, "seen = show()"],
        names=["rows", "seen"],
    ),
    notebook(
        "unstorable-fill-clocked-bar-two-levels-down",
        [_CLOCKED, "bar = Bar(sys.stdout)\n" + _FILL_TWO_LEVELS, "total = rows * 2"],
        names=["rows", "total"],
        source="E10's c5: the bar's clock named it, so the passes after the first kept missing",
    ),
]
