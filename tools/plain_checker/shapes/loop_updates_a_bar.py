"""One-unit loops that update an object holding a standard stream or a lock."""

from tools.plain_checker import notebook

BAR = (
    "import sys, time, threading\n"
    "class Bar:\n"
    "    def __init__(self, stream):\n"
    "        self.stream = stream\n"
    "        self.out = stream.write\n"
    "        self.cnt = 0\n"
    "    def update(self):\n"
    "        self.cnt += 1\n"
    "        if self.cnt % 1000 == 0:\n"
    "            self.out('.')\n"
)
LOOP = "for i in range(3000):\n    time.sleep(0.0003)\n    pbar.update()"


def _bar(name, make, read="r = (pbar.cnt, pbar.stream is sys.stdout, pbar.stream is sys.stderr)"):
    return notebook(name, cells=[BAR + make, LOOP, read], names=["r"])


SHAPES = [
    _bar("bar-stdout", "pbar = Bar(sys.stdout)"),
    _bar("bar-stderr", "pbar = Bar(sys.stderr)"),
    _bar("bar-stdout-alias", "pbar = Bar(sys.stdout)\nkeep = [pbar]", "r = (keep[0].cnt, keep[0] is pbar)"),
    _bar(
        "bar-free-lock",
        "pbar = Bar(sys.stdout)\npbar.lock = threading.Lock()",
        "r = (pbar.cnt, pbar.lock.acquire(blocking=False))",
    ),
    _bar(
        "bar-held-lock",
        "pbar = Bar(sys.stdout)\npbar.lock = threading.RLock()\npbar.lock.acquire()",
        "r = pbar.cnt",
    ),
    _bar("bar-file", "log = open('bar.log', 'w')\npbar = Bar(log)", "r = pbar.cnt"),
    notebook(
        "bar-tqdm",
        cells=["import time\nfrom tqdm import tqdm\npbar = tqdm(total=3000)", LOOP, "r = pbar.n"],
        names=["r"],
    ),
    notebook(
        "loop-over-tqdm",
        cells=[
            "import time\nfrom tqdm import tqdm\nacc = []",
            "for i in tqdm(range(3000)):\n    time.sleep(0.0003)\n    acc.append(i)",
            "r = (len(acc), acc[-1])",
        ],
        names=["r"],
    ),
]
