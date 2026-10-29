"""Reviewer probes for unstorable-touched-unknown."""
from tools.plain_checker import notebook

_SETUP = (
    "import sys, functools\nclass Bar:\n    def __init__(self, stream):\n        self.n = 0\n        self.stream = stream\n        self.me = self\n"
    "    def update(self):\n        self.n += 1\n"
)
_FILL = "rows = 0\nfor s in (1, 2):\n    for i in range(300000):\n        rows = rows + i * s\n        bar.update()"
_SINK = (
    "import sys\nclass Bar:\n    def __init__(self, sink):\n        self.n = 0\n        self.sink = sink\n        self.me = self\n"
    "    def update(self):\n        self.n += 1\n        if self.n % 200000 == 0:\n            self.sink.append(self.n)\n"
)
_PRINT = _SETUP.replace("self.n += 1\n", "self.n += 1\n        if self.n % 200000 == 0:\n            print('tick', self.n)\n")
_DECO = "def deco(f):\n    def w(*a):\n        return f(*a)\n    return w\n"

SHAPES = [
    notebook("p-alias-reader", [_SETUP, "bar = Bar(sys.stdout)", "b2 = bar", _FILL, "seen = b2.n"], names=["rows", "seen"]),
    notebook("p-container-reader", [_SETUP, "bar = Bar(sys.stdout)\nbars = [bar]", _FILL, "seen = bars[0].n"], names=["rows", "seen"]),
    notebook("p-dict-holder-reader", [_SETUP, "bar = Bar(sys.stdout)\nui = {'bar': bar}", _FILL, "seen = ui['bar'].n"], names=["rows", "seen"]),
    notebook("p-partial-reader", [_SETUP, "show = functools.partial(lambda k: bar.n + k, 0)", "bar = Bar(sys.stdout)\n" + _FILL, "seen = show()"], names=["rows", "seen"]),
    notebook("p-decorated-reader", [_SETUP + _DECO, "@deco\ndef show():\n    return bar.n", "bar = Bar(sys.stdout)\n" + _FILL, "seen = show()"], names=["rows", "seen"]),
    notebook("p-lambda-in-dict-reader", [_SETUP, "fns = {'show': lambda: bar.n}", "bar = Bar(sys.stdout)\n" + _FILL, "seen = fns['show']()"], names=["rows", "seen"]),
    notebook("p-eval-reader", [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL, "seen = eval('bar.n')"], names=["rows", "seen"]),
    notebook("p-globals-reader", [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL, "seen = globals()['bar'].n"], names=["rows", "seen"]),
    notebook("p-sink-written", [_SINK, "sink = []\nbar = Bar(sink)", _FILL, "seen = len(sink)"], names=["rows", "seen", "sink"]),
    notebook("p-bar-prints", [_PRINT, "bar = Bar(sys.stdout)\n" + _FILL, "total = rows * 2"], names=["rows", "total"]),
    notebook("p-method-reader-class-held", [_SETUP + "class Rep:\n    def show(self):\n        return bar.n\nrep = Rep()\n", "bar = Bar(sys.stdout)\n" + _FILL, "seen = rep.show()"], names=["rows", "seen"]),
    notebook("p-reader-cell-added-scratch", [_SETUP, "bar = Bar(sys.stdout)\n" + _FILL, "total = rows * 2"], names=["rows", "total", "seen"],
             plan=["all", "restart", "all", 2], edits={2: {2: "seen = bar.n"}}),
]
