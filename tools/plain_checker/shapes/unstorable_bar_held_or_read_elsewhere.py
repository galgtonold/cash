"""Holders and readers of an unstorable bar a loop fills, from the batch 16 review.

Each must give what restart + Run All gives: anything that holds or reads
the bar another way than its name keeps the loop running.
"""

from tools.plain_checker import notebook

_SETUP_A = (
    "import sys, functools\nclass Bar:\n    def __init__(self, stream):\n        self.n = 0\n        self.stream = stream\n        self.me = self\n"
    "    def update(self):\n        self.n += 1\n"
)
_FILL_A = "rows = 0\nfor s in (1, 2):\n    for i in range(300000):\n        rows = rows + i * s\n        bar.update()"
_SINK_A = (
    "import sys\nclass Bar:\n    def __init__(self, sink):\n        self.n = 0\n        self.sink = sink\n        self.me = self\n"
    "    def update(self):\n        self.n += 1\n        if self.n % 200000 == 0:\n            self.sink.append(self.n)\n"
)
_PRINT_A = _SETUP_A.replace(
    "self.n += 1\n", "self.n += 1\n        if self.n % 200000 == 0:\n            print('tick', self.n)\n"
)
_DECO_A = "def deco(f):\n    def w(*a):\n        return f(*a)\n    return w\n"

_SHAPES_A = [
    notebook(
        "p-alias-reader",
        [_SETUP_A, "bar = Bar(sys.stdout)", "b2 = bar", _FILL_A, "seen = b2.n"],
        names=["rows", "seen"],
    ),
    notebook(
        "p-container-reader",
        [_SETUP_A, "bar = Bar(sys.stdout)\nbars = [bar]", _FILL_A, "seen = bars[0].n"],
        names=["rows", "seen"],
    ),
    notebook(
        "p-dict-holder-reader",
        [_SETUP_A, "bar = Bar(sys.stdout)\nui = {'bar': bar}", _FILL_A, "seen = ui['bar'].n"],
        names=["rows", "seen"],
    ),
    notebook(
        "p-partial-reader",
        [
            _SETUP_A,
            "show = functools.partial(lambda k: bar.n + k, 0)",
            "bar = Bar(sys.stdout)\n" + _FILL_A,
            "seen = show()",
        ],
        names=["rows", "seen"],
    ),
    notebook(
        "p-decorated-reader",
        [
            _SETUP_A + _DECO_A,
            "@deco\ndef show():\n    return bar.n",
            "bar = Bar(sys.stdout)\n" + _FILL_A,
            "seen = show()",
        ],
        names=["rows", "seen"],
    ),
    notebook(
        "p-lambda-in-dict-reader",
        [_SETUP_A, "fns = {'show': lambda: bar.n}", "bar = Bar(sys.stdout)\n" + _FILL_A, "seen = fns['show']()"],
        names=["rows", "seen"],
    ),
    notebook(
        "p-eval-reader", [_SETUP_A, "bar = Bar(sys.stdout)\n" + _FILL_A, "seen = eval('bar.n')"], names=["rows", "seen"]
    ),
    notebook(
        "p-globals-reader",
        [_SETUP_A, "bar = Bar(sys.stdout)\n" + _FILL_A, "seen = globals()['bar'].n"],
        names=["rows", "seen"],
    ),
    notebook(
        "p-sink-written",
        [_SINK_A, "sink = []\nbar = Bar(sink)", _FILL_A, "seen = len(sink)"],
        names=["rows", "seen", "sink"],
    ),
    notebook(
        "p-bar-prints", [_PRINT_A, "bar = Bar(sys.stdout)\n" + _FILL_A, "total = rows * 2"], names=["rows", "total"]
    ),
    notebook(
        "p-method-reader-class-held",
        [
            _SETUP_A + "class Rep:\n    def show(self):\n        return bar.n\nrep = Rep()\n",
            "bar = Bar(sys.stdout)\n" + _FILL_A,
            "seen = rep.show()",
        ],
        names=["rows", "seen"],
    ),
]
_BASE_B = "import sys\nclass Bar:\n    total = 0\n    def __init__(self, stream):\n        self.n = 0\n        self.stream = stream\n        self.me = self\n    def update(self):\n        self.n += 1\n"
_FILL_B = "rows = 0\nfor s in (1, 2):\n    for i in range(300000):\n        rows = rows + i * s\n        bar.update()"
_LOG_B = _BASE_B + "        if self.n % 200000 == 0:\n            log.append(self.n)\nlog = []\n"
_CLS_B = _BASE_B + "        Bar.total += 1\n"
_DICT_B = "import sys\nclass Bar:\n    def __init__(self, stream, tally):\n        self.n = 0\n        self.tally = tally\n        self.me = self\n    def update(self):\n        self.n += 1\n        self.tally['n'] = self.n\n"
_SHAPES_B = [
    notebook("p2-global-log", [_LOG_B, "bar = Bar(sys.stdout)\n" + _FILL_B, "seen = len(log)"], names=["rows", "seen"]),
    notebook(
        "p2-class-counter", [_CLS_B, "bar = Bar(sys.stdout)\n" + _FILL_B, "seen = Bar.total"], names=["rows", "seen"]
    ),
    notebook(
        "p2-tally-dict-same-cell",
        [_DICT_B, "tally = {}\nbar = Bar(sys.stdout, tally)\n" + _FILL_B, "seen = tally.get('n')"],
        names=["rows", "seen"],
    ),
    notebook(
        "p2-exec-reader",
        [
            _BASE_B,
            "bar = Bar(sys.stdout)\n" + _FILL_B,
            "ns = {}\nexec('seen = bar.n', globals(), ns)\nseen = ns['seen']",
        ],
        names=["rows", "seen"],
    ),
]
_BASE_C = "import sys\nclass Bar:\n    def __init__(self, stream):\n        self.n = 0\n        self.stream = stream\n        self.me = self\n    def update(self):\n        self.n += 1\n"
_FILL_C = "rows = 0\nfor s in (1, 2):\n    for i in range(300000):\n        rows = rows + i * s\n        bar.update()"
_SHAPES_C = [
    notebook(
        "p3-default-arg-reader",
        [_BASE_C, "bar = Bar(sys.stdout)\ndef report(b=bar):\n    return b.n", _FILL_C, "seen = report()"],
        names=["rows", "seen"],
    ),
    notebook(
        "p3-closure-reader",
        [
            _BASE_C,
            "bar = Bar(sys.stdout)\ndef mk(b):\n    return lambda: b.n\nreport = mk(bar)",
            _FILL_C,
            "seen = report()",
        ],
        names=["rows", "seen"],
    ),
    notebook(
        "p3-bound-method-reader",
        [
            _BASE_C + "    def peek(self):\n        return self.n\n",
            "bar = Bar(sys.stdout)\npeek = bar.peek",
            _FILL_C,
            "seen = peek()",
        ],
        names=["rows", "seen"],
    ),
]

SHAPES = [s for s in _SHAPES_A + _SHAPES_B + _SHAPES_C if s.name != "p-reader-cell-added-scratch"]
