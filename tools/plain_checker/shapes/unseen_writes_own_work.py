"""Shapes for code of the user's that cash runs itself (pickling a value to
store it), and for code whose text cannot be read (swapped in from a string)."""

from tools.plain_checker import notebook

_LOGGED = (
    "import pickle, time\n"
    "LOG = []\n"
    "class Logged:\n"
    "    def __init__(self, n):\n"
    "        self.rows = list(range(n))\n"
    "    def __reduce_ex__(self, protocol):\n"
    "        LOG.append(protocol)\n"
    "        return object.__reduce_ex__(self, protocol)\n"
    "def build(n):\n"
    "    time.sleep(0.02)\n"
    "    return Logged(n)"
)

SHAPES = [
    notebook(
        "reduce-ex-cash-pickles",
        cells=[
            _LOGGED,
            "G = build(1000)",
            "total = sum(G.rows)",
            "B = pickle.dumps(G)\nback = len(pickle.loads(B).rows)",
        ],
        names=["total", "back"],
        edits={3: {1: "G = build(2000)"}},
        source="unseen-special false positive: cash's own pickling ran __reduce_ex__",
    ),
    notebook(
        "code-swapped-from-a-string",
        cells=[
            "LOG = []\ndef inner(x):\n    return x * 2",
            "inner.__code__ = eval(compile('lambda x: x * 10', '<variant>', 'eval')).__code__",
            "v = inner(3)",
            "w = v + 1",
        ],
        names=["v", "w"],
        edits={3: {1: "inner.__code__ = eval(compile('lambda x: x * 20', '<variant>', 'eval')).__code__"}},
        source="unseen-special false positive: unreadable lambda",
    ),
    notebook(
        "code-swapped-from-a-string-writes",
        cells=[
            "LOG = []\ndef inner(x):\n    return x * 2",
            "inner.__code__ = eval(compile('lambda x: LOG.append(x) or x * 10', '<variant>', 'eval')).__code__",
            "v = inner(3)",
            "n = len(LOG)",
        ],
        names=["v", "n"],
        source="unreadable code that does change a global is still run",
    ),
]
