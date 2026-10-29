from tools.plain_checker import notebook
_BASE = "import sys\nclass Bar:\n    def __init__(self, stream):\n        self.n = 0\n        self.stream = stream\n        self.me = self\n    def update(self):\n        self.n += 1\n"
_FILL = "rows = 0\nfor s in (1, 2):\n    for i in range(300000):\n        rows = rows + i * s\n        bar.update()"
SHAPES = [
    notebook("p3-default-arg-reader", [_BASE, "bar = Bar(sys.stdout)\ndef report(b=bar):\n    return b.n", _FILL, "seen = report()"], names=["rows", "seen"]),
    notebook("p3-closure-reader", [_BASE, "bar = Bar(sys.stdout)\ndef mk(b):\n    return lambda: b.n\nreport = mk(bar)", _FILL, "seen = report()"], names=["rows", "seen"]),
    notebook("p3-bound-method-reader", [_BASE + "    def peek(self):\n        return self.n\n", "bar = Bar(sys.stdout)\npeek = bar.peek", _FILL, "seen = peek()"], names=["rows", "seen"]),
]
