"""``operator.methodcaller`` of a method of the user's that changes what the
methodcaller hands it (review shared6 note 4): served stale on 5e84e39."""

from tools.plain_checker import notebook

H = (
    "import operator, time\n"
    "class A:\n"
    "    def add(self, lst, x):\n"
    "        time.sleep(0.3)\n"
    "        lst.append(x)\n"
    "        return len(lst)\n"
    "    def size(self, lst, x):\n"
    "        time.sleep(0.3)\n"
    "        return len(lst) + x\n"
    "a = A()"
)
L = "lst = [0]"
CASES = {
    "inline": (L, "r = operator.methodcaller('add', lst, 1)(a)"),
    "named": (L + "\nmc = operator.methodcaller('add', lst, 1)", "r = mc(a)"),
    "inline-kw": (L, "r = operator.methodcaller('add', lst, x=1)(a)"),
    "named-kw": (L + "\nmc = operator.methodcaller('add', lst=lst, x=1)", "r = mc(a)"),
    "inline-class": (L, "r = operator.methodcaller('add', a, lst, 1)(A)"),
    "ctl-reads": (L, "r = operator.methodcaller('size', lst, 1)(a)"),
}
SHAPES = [
    notebook(f"mc-{name}", cells=[H, build, call, "n = (len(lst), r)"], source="review shared6 note 4")
    for name, (build, call) in CASES.items()
]
