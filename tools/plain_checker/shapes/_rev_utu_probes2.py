from tools.plain_checker import notebook
_BASE = "import sys\nclass Bar:\n    total = 0\n    def __init__(self, stream):\n        self.n = 0\n        self.stream = stream\n        self.me = self\n    def update(self):\n        self.n += 1\n"
_FILL = "rows = 0\nfor s in (1, 2):\n    for i in range(300000):\n        rows = rows + i * s\n        bar.update()"
_LOG = _BASE + "        if self.n % 200000 == 0:\n            log.append(self.n)\nlog = []\n"
_CLS = _BASE + "        Bar.total += 1\n"
_DICT = "import sys\nclass Bar:\n    def __init__(self, stream, tally):\n        self.n = 0\n        self.tally = tally\n        self.me = self\n    def update(self):\n        self.n += 1\n        self.tally['n'] = self.n\n"
SHAPES = [
    notebook("p2-global-log", [_LOG, "bar = Bar(sys.stdout)\n" + _FILL, "seen = len(log)"], names=["rows", "seen"]),
    notebook("p2-class-counter", [_CLS, "bar = Bar(sys.stdout)\n" + _FILL, "seen = Bar.total"], names=["rows", "seen"]),
    notebook("p2-tally-dict-same-cell", [_DICT, "tally = {}\nbar = Bar(sys.stdout, tally)\n" + _FILL, "seen = tally.get('n')"], names=["rows", "seen"]),
    notebook("p2-exec-reader", [_BASE, "bar = Bar(sys.stdout)\n" + _FILL, "ns = {}\nexec('seen = bar.n', globals(), ns)\nseen = ns['seen']"], names=["rows", "seen"]),
]
