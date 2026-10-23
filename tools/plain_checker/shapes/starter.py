"""Starter shapes harvested from review repros."""

from tools.plain_checker import decorator, notebook

SHAPES = [
    notebook(
        "draw-twice-in-one-cell",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HG_TWICE = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nG = gen()\\n'\nif not os.path.exists('hg_twice.py') or open('hg_twice.py').read() != SRC_HG_TWICE:\n    with open('hg_twice.py', 'w') as fh:\n        fh.write(SRC_HG_TWICE)\nfrom hg_twice import G",
            "t = next(G)\nu = t + 1\nt = next(G)\nv = t * 10",
            "w = (u, v)",
        ],
        names=["t", "u", "v", "w"],
        source="reviews/rerun-new-lineage-2/review.md F2 Repro B (g_twice_cell)",
    ),
    notebook(
        "timestamps-around-work",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HW_TIME = 'CALLS = [0]\\ndef work(x):\\n    CALLS[0] += 1\\n    return CALLS[0]\\n'\nif not os.path.exists('hw_time.py') or open('hw_time.py').read() != SRC_HW_TIME:\n    with open('hw_time.py', 'w') as fh:\n        fh.write(SRC_HW_TIME)\nimport time\nfrom hw_time import work",
            "start = time.time()\nr = work(start)\nstart = time.time()",
            "z = r + 1",
        ],
        names=["r", "z"],
        source="reviews/rerun-new-lineage-2/review.md F2 Repro A (g_twice_time); call count returned as value",
    ),
    notebook(
        "module-edit-back-to-equal-result",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HS_SAME = 'def f(x, k=2):\\n    return x * k\\n'\nif not os.path.exists('hs_same.py') or open('hs_same.py').read() != SRC_HS_SAME:\n    with open('hs_same.py', 'w') as fh:\n        fh.write(SRC_HS_SAME)\nimport importlib, hs_same\nimportlib.reload(hs_same)",
            "def g(x):\n    import hs_same\n    return hs_same.f(x)",
            "t = g(3)",
            "u = t + 1",
        ],
        names=["t", "u"],
        edits={
            1: {
                0: "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HS_SAME = 'def f(x, k=10):\\n    return x * k\\n'\nif not os.path.exists('hs_same.py') or open('hs_same.py').read() != SRC_HS_SAME:\n    with open('hs_same.py', 'w') as fh:\n        fh.write(SRC_HS_SAME)\nimport importlib, hs_same\nimportlib.reload(hs_same)"
            },
            2: {
                0: "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HS_SAME = 'def f(x, k=5):\\n    return x * k * 2\\n'\nif not os.path.exists('hs_same.py') or open('hs_same.py').read() != SRC_HS_SAME:\n    with open('hs_same.py', 'w') as fh:\n        fh.write(SRC_HS_SAME)\nimport importlib, hs_same\nimportlib.reload(hs_same)"
            },
            3: {
                0: "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HS_SAME = 'def f(x, k=2):\\n    return x * k * 1\\n'\nif not os.path.exists('hs_same.py') or open('hs_same.py').read() != SRC_HS_SAME:\n    with open('hs_same.py', 'w') as fh:\n        fh.write(SRC_HS_SAME)\nimport importlib, hs_same\nimportlib.reload(hs_same)"
            },
        },
        source="reviews/rerun-new-lineage-2/review.md F1 Repro B (s_samevalue_mod); module reloaded in-kernel instead of per restart",
    ),
    notebook(
        "code-swap-back-to-equal-result",
        cells=[
            "def inner(x):\n    return x * 2",
            "t = inner(3)",
            "u = t + 1",
        ],
        names=["t", "u"],
        edits={
            1: {0: "def inner(x):\n    return x * 2\ninner.__code__ = (lambda x: x * 10).__code__"},
            2: {0: "def inner(x):\n    return x * 2\ninner.__code__ = (lambda x: x * 5 * 2).__code__"},
        },
        source="reviews/rerun-new-lineage-2/review.md F1 Repro A (s_samevalue_code); __code__ swap done in cell 0",
    ),
    notebook(
        "call-by-string-runs-once",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HD_COUNT = 'CALLS = [0]\\n'\nif not os.path.exists('hd_count.py') or open('hd_count.py').read() != SRC_HD_COUNT:\n    with open('hd_count.py', 'w') as fh:\n        fh.write(SRC_HD_COUNT)\nimport hd_count",
            "name = 'make'",
            "def make(x):\n    hd_count.CALLS[0] += 1\n    return x + 1",
            "t = globals()[name](3)",
            "u = t + 0",
            "w = u * 2",
            "n = hd_count.CALLS[0]",
        ],
        names=["t", "u", "w", "n"],
        source="reviews/rerun-new-lineage/review.md F1 Repro A (d_byname_count); call count read back as n",
    ),
    notebook(
        "module-generator-next",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HB_GEN = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nGEN = gen()\\n'\nif not os.path.exists('hb_gen.py') or open('hb_gen.py').read() != SRC_HB_GEN:\n    with open('hb_gen.py', 'w') as fh:\n        fh.write(SRC_HB_GEN)\nfrom hb_gen import GEN",
            "t = next(GEN)",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage/review.md claimed case b_gen (also reviews/from-import-value/review.md 'Finding E shows up here too')",
    ),
    notebook(
        "module-deque-popleft",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HQ_DEQUE = 'from collections import deque\\nQ = deque(range(100))\\n'\nif not os.path.exists('hq_deque.py') or open('hq_deque.py').read() != SRC_HQ_DEQUE:\n    with open('hq_deque.py', 'w') as fh:\n        fh.write(SRC_HQ_DEQUE)\nfrom hq_deque import Q",
            "t = Q.popleft()",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage/review.md F2 e_deque",
    ),
    notebook(
        "module-generator-send",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HQ_SEND = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nG = gen()\\n'\nif not os.path.exists('hq_send.py') or open('hq_send.py').read() != SRC_HQ_SEND:\n    with open('hq_send.py', 'w') as fh:\n        fh.write(SRC_HQ_SEND)\nfrom hq_send import G",
            "t = G.send(None)",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage/review.md F2 e_gensend",
    ),
    notebook(
        "module-object-draw-method",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HQ_METH = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\n\\nclass Drawer:\\n    def __init__(self):\\n        self.g = gen()\\n    def draw(self):\\n        return next(self.g)\\nOBJ = Drawer()\\n'\nif not os.path.exists('hq_meth.py') or open('hq_meth.py').read() != SRC_HQ_METH:\n    with open('hq_meth.py', 'w') as fh:\n        fh.write(SRC_HQ_METH)\nfrom hq_meth import OBJ",
            "t = OBJ.draw()",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage/review.md F2 e_genmeth",
    ),
    notebook(
        "module-dict-iter-then-pop",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HD_ITER = \"D = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}\\n\"\nif not os.path.exists('hd_iter.py') or open('hd_iter.py').read() != SRC_HD_ITER:\n    with open('hd_iter.py', 'w') as fh:\n        fh.write(SRC_HD_ITER)\nfrom hd_iter import D",
            "t = next(iter(D))",
            "D.pop(t)",
        ],
        names=["t"],
        source="reviews/rerun-new-lineage/review.md 'seen along the way' e_dictiter_popitem",
    ),
    notebook(
        "module-heappop",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HH_HEAP = 'H = list(range(100))\\n'\nif not os.path.exists('hh_heap.py') or open('hh_heap.py').read() != SRC_HH_HEAP:\n    with open('hh_heap.py', 'w') as fh:\n        fh.write(SRC_HH_HEAP)\nimport heapq\nfrom hh_heap import H",
            "t = heapq.heappop(H)",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage/review.md 'seen along the way' e_heappop",
    ),
    notebook(
        "module-helper-calls-next",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HE_HELPER = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nG = gen()\\ndef draw():\\n    return next(G)\\n'\nif not os.path.exists('he_helper.py') or open('he_helper.py').read() != SRC_HE_HELPER:\n    with open('he_helper.py', 'w') as fh:\n        fh.write(SRC_HE_HELPER)\nfrom he_helper import draw",
            "t = draw()",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage/review.md 'seen along the way' e_helper_next",
    ),
    notebook(
        "module-csv-chunks",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HC_CSV = \"import pandas as pd\\nR = pd.read_csv('hc_data.csv', chunksize=3)\\n\"\nif not os.path.exists('hc_csv.py') or open('hc_csv.py').read() != SRC_HC_CSV:\n    with open('hc_csv.py', 'w') as fh:\n        fh.write(SRC_HC_CSV)\nDATA = 'a\\n' + ''.join(f'{i}\\n' for i in range(30))\nif not os.path.exists('hc_data.csv') or open('hc_data.csv').read() != DATA:\n    with open('hc_data.csv', 'w') as fh:\n        fh.write(DATA)\nfrom hc_csv import R",
            "t = next(R)",
            "u = int(t['a'].sum())",
        ],
        names=["u"],
        source="reviews/rerun-new-lineage/review.md claimed case e_csv_chunks",
    ),
    notebook(
        "module-seeded-rng-instance",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HR_RNG = 'import random\\nR = random.Random(0)\\n'\nif not os.path.exists('hr_rng.py') or open('hr_rng.py').read() != SRC_HR_RNG:\n    with open('hr_rng.py', 'w') as fh:\n        fh.write(SRC_HR_RNG)\nfrom hr_rng import R",
            "t = R.random()",
            "u = round(t * 1000)",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage-2/review.md 'seen along the way' f_rand_inst",
    ),
    notebook(
        "draw-twice-in-try-blocks",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HJ_TRY = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nG = gen()\\n'\nif not os.path.exists('hj_try.py') or open('hj_try.py').read() != SRC_HJ_TRY:\n    with open('hj_try.py', 'w') as fh:\n        fh.write(SRC_HJ_TRY)\nfrom hj_try import G",
            "try:\n    t = next(G)\nexcept StopIteration:\n    pass\nu = t + 1\ntry:\n    t = next(G)\nexcept StopIteration:\n    pass\nv = t * 10",
            "w = (u, v)",
        ],
        names=["u", "v", "w"],
        source="reviews/rerun-new-lineage-3/review.md F1 Repro A (j_draw_try)",
    ),
    notebook(
        "draw-twice-in-with-blocks",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HJ_WITH = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nG = gen()\\n'\nif not os.path.exists('hj_with.py') or open('hj_with.py').read() != SRC_HJ_WITH:\n    with open('hj_with.py', 'w') as fh:\n        fh.write(SRC_HJ_WITH)\nimport contextlib\nfrom hj_with import G",
            "with contextlib.nullcontext():\n    t = next(G)\nu = t + 1\nwith contextlib.nullcontext():\n    t = next(G)\nv = t * 10",
            "w = (u, v)",
        ],
        names=["u", "v", "w"],
        source="reviews/rerun-new-lineage-3/review.md F1 (j_draw_with)",
    ),
    notebook(
        "draw-in-if-read-next-cell",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HK_IF = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nG = gen()\\n'\nif not os.path.exists('hk_if.py') or open('hk_if.py').read() != SRC_HK_IF:\n    with open('hk_if.py', 'w') as fh:\n        fh.write(SRC_HK_IF)\nfrom hk_if import G",
            "if True:\n    t = next(G)",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/e4-and-hidden-2/review.md finding A1 Repro 1 (k_if_one_later)",
    ),
    notebook(
        "draw-and-reader-in-loop-body",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HK_FOR = 'def gen():\\n    i = 0\\n    while True:\\n        yield i\\n        i += 1\\nG = gen()\\n'\nif not os.path.exists('hk_for.py') or open('hk_for.py').read() != SRC_HK_FOR:\n    with open('hk_for.py', 'w') as fh:\n        fh.write(SRC_HK_FOR)\nfrom hk_for import G",
            "for _ in range(1):\n    t = next(G)\n    u = t + 1",
            "w = u",
        ],
        names=["t", "u", "w"],
        source="reviews/e4-and-hidden-2/review.md finding A1 Repro 2 (i_for_twice)",
    ),
    notebook(
        "file-rewritten-later-in-cell",
        cells=[
            "import os\nP = 'f.txt'",
            "open(P, 'w').write('1')\nfor _ in range(1):\n    t = int(open(P).read())\nu = t\nopen(P, 'w').write('2')",
            "w = u",
        ],
        names=["t", "u", "w"],
        source="reviews/e4-and-hidden-2/review.md finding A4 (m_file_loop_one)",
    ),
    notebook(
        "iterator-one-step-away-in-closure",
        cells=[
            "def make():\n    st = {'it': iter(range(100))}\n    def h():\n        return next(st['it'])\n    return h",
            "h = make()",
            "x = h()",
            "y = h()",
        ],
        names=["x", "y"],
        source="reviews/e4-and-hidden-2/review.md finding B2 (iter_closure_dict)",
    ),
    notebook(
        "from-import-setting-edited-above",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HF_CFG = 'CFG = 1\\n'\nif not os.path.exists('hf_cfg.py') or open('hf_cfg.py').read() != SRC_HF_CFG:\n    with open('hf_cfg.py', 'w') as fh:\n        fh.write(SRC_HF_CFG)\nimport hf_cfg",
            "hf_cfg.CFG = 1",
            "from hf_cfg import CFG",
            "t = CFG + 1",
            "u = t + 1",
        ],
        names=["CFG", "t", "u"],
        edits={1: {1: "hf_cfg.CFG = 5"}},
        source="reviews/from-import-value/review.md G1 / R1 (b_runall_only, b_restart_jump cells)",
    ),
    notebook(
        "from-import-callable-instance",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HF_SCALE = 'class Scale:\\n    def __init__(self, k):\\n        self.k = k\\n    def __call__(self, x):\\n        return x * self.k\\nSCALE = Scale(1)\\n'\nif not os.path.exists('hf_scale.py') or open('hf_scale.py').read() != SRC_HF_SCALE:\n    with open('hf_scale.py', 'w') as fh:\n        fh.write(SRC_HF_SCALE)\nimport hf_scale",
            "hf_scale.SCALE.k = 1",
            "from hf_scale import SCALE",
            "t = SCALE(1) + 1",
            "u = t + 1",
        ],
        names=["t", "u"],
        edits={1: {1: "hf_scale.SCALE.k = 5"}},
        source="reviews/from-import-value/review.md 'G1 is not closed for callable values' (b_callobj)",
    ),
    notebook(
        "from-import-partial-rebound",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HF_PART = 'from functools import partial\\ndef mul(x, k):\\n    return x * k\\nPART = partial(mul, k=1)\\n'\nif not os.path.exists('hf_part.py') or open('hf_part.py').read() != SRC_HF_PART:\n    with open('hf_part.py', 'w') as fh:\n        fh.write(SRC_HF_PART)\nimport hf_part",
            "hf_part.PART = hf_part.partial(hf_part.mul, k=1)",
            "from hf_part import PART",
            "t = PART(1) + 1",
            "u = t + 1",
        ],
        names=["t", "u"],
        edits={1: {1: "hf_part.PART = hf_part.partial(hf_part.mul, k=5)"}},
        source="reviews/from-import-value/review.md 'G1 is not closed for callable values' (b_partial)",
    ),
    notebook(
        "from-import-partial-keywords-in-place",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HF_PKW = 'from functools import partial\\ndef mul(x, k):\\n    return x * k\\nPART = partial(mul, k=1)\\n'\nif not os.path.exists('hf_pkw.py') or open('hf_pkw.py').read() != SRC_HF_PKW:\n    with open('hf_pkw.py', 'w') as fh:\n        fh.write(SRC_HF_PKW)\nimport hf_pkw",
            "hf_pkw.PART.keywords['k'] = 1",
            "from hf_pkw import PART",
            "t = PART(1) + 1",
            "u = t + 1",
        ],
        names=["t", "u"],
        edits={1: {1: "hf_pkw.PART.keywords['k'] = 5"}},
        source="reviews/from-import-value/review.md 'G1 is not closed for callable values' (b_partial_inplace)",
    ),
    notebook(
        "from-import-dict-bumped-below",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HF_TABLE = \"TABLE = {'k': 1}\\n\"\nif not os.path.exists('hf_table.py') or open('hf_table.py').read() != SRC_HF_TABLE:\n    with open('hf_table.py', 'w') as fh:\n        fh.write(SRC_HF_TABLE)\nimport hf_table",
            "from hf_table import TABLE",
            "t = TABLE['k'] + 1",
            "hf_table.TABLE['k'] += 1",
        ],
        names=["t"],
        source="reviews/from-import-value/review.md results table 'helpers.TABLE[...] += 1 below' (b_below_inplace_mod)",
    ),
    notebook(
        "loop-fresh-memo-from-factory",
        cells=[
            "import functools\ndef slow(n):\n    return n * 2\ndef mk():\n    return functools.cache(slow)",
            "sizes = []\nfor i in range(2):\n    m = mk()\n    sizes.append(m.cache_info().currsize)\n    m(1)\n    m(2)",
        ],
        names=["sizes"],
        source="reviews/made-memo-not-served/review.md cell probe 'm = mk()' (test_probe_memo_cells)",
    ),
    notebook(
        "loop-fresh-memo-in-namespace",
        cells=[
            "import functools\nfrom types import SimpleNamespace\ndef slow(n):\n    return n * 2\ndef mk_ns():\n    return SimpleNamespace(f=functools.cache(slow))",
            "sizes = []\nfor i in range(2):\n    h = mk_ns()\n    sizes.append(h.f.cache_info().currsize)\n    h.f(1)\n    h.f(2)",
        ],
        names=["sizes"],
        source="reviews/made-memo-not-served/review.md cell probe 'h = mk_ns()'",
    ),
    notebook(
        "loop-fresh-memo-in-default-arg",
        cells=[
            "import functools\ndef g(n):\n    return n * 2\ndef mk():\n    def f(n, _m=functools.cache(g)):\n        return _m(n)\n    return f",
            "sizes = []\nfor i in range(2):\n    f = mk()\n    sizes.append(f.__defaults__[0].cache_info().currsize)\n    f(1)\n    f(2)",
        ],
        names=["sizes"],
        source="reviews/made-memo-not-served-2/review.md F2 (default-argument memo)",
    ),
    notebook(
        "reducer-appends-to-module-list",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HR_RED = 'SEEN = []\\nclass P:\\n    def __init__(self):\\n        self.v = 1\\n    def __getstate__(self):\\n        SEEN.append(1)\\n        return self.__dict__\\n'\nif not os.path.exists('hr_red.py') or open('hr_red.py').read() != SRC_HR_RED:\n    with open('hr_red.py', 'w') as fh:\n        fh.write(SRC_HR_RED)\nimport hr_red",
            "p = hr_red.P()",
            "n = len(hr_red.SEEN)",
        ],
        names=["n"],
        source="reviews/e-r5-reducer-r2/review.md R4 (mod_attr_read)",
    ),
    notebook(
        "reducer-clears-own-field",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_HR_SELF = \"class P:\\n    def __init__(self):\\n        self.cache = ['a']\\n    def __getstate__(self):\\n        self.cache.clear()\\n        return self.__dict__\\n\"\nif not os.path.exists('hr_self.py') or open('hr_self.py').read() != SRC_HR_SELF:\n    with open('hr_self.py', 'w') as fh:\n        fh.write(SRC_HR_SELF)\nimport hr_self",
            "p = hr_self.P()",
            "c = list(p.cache)",
        ],
        names=["c"],
        source="reviews/e-r5-reducer-r2/review.md R4 (writes_self)",
    ),
    decorator(
        "callable-arg-claiming-type-reads-global",
        setup="G = 10\nclass Liar:\n    @property\n    def __class__(self):\n        return type\n    def __call__(self, x):\n        return x + G\n@cache\ndef f(fn, x):\n    return fn(x)\n",
        steps=["obj = Liar()", "f(obj, 2)", "G = 20", "f(obj, 2)"],
        source="reviews/raising-and-claims/review.md item 3 (same-process variant)",
    ),
    decorator(
        "helper-code-swapped",
        setup="def helper(x):\n    return x * 2\n@cache\ndef f(x):\n    return helper(x)\n",
        steps=["f(3)", "helper.__code__ = (lambda x: x * 10).__code__", "f(3)"],
        source="reviews/shadow-dir-per-notebook-2/review.md check table 'Decorator path' (helper.__code__ edit, 6 then 30)",
    ),
    decorator(
        "returned-namespace-memo-starts-empty",
        setup="import functools\nfrom types import SimpleNamespace\ndef double(n):\n    return n * 2\n@cache\ndef mk_ns():\n    return SimpleNamespace(f=functools.cache(double))\n",
        steps=[
            "h = mk_ns()",
            "h.f.cache_info().currsize",
            "h.f(1)",
            "h.f(2)",
            "h2 = mk_ns()",
            "h2.f.cache_info().currsize",
        ],
        source="reviews/made-memo-not-served/review.md unit-form table 'memo in a SimpleNamespace'",
    ),
    decorator(
        "iadd-side-effect-on-arg",
        setup="class Acc:\n    def __init__(self):\n        self.n = 0\n    def __iadd__(self, k):\n        self.n += k\n        return self\n@cache\ndef f(v):\n    v += 4\n    return v.n\n",
        steps=["a = Acc()", "f(a)", "a.n", "f(a)", "a.n"],
        source="reviews/aug-assign-dunder/review.md item 5 ('def f(v): v += 4' with __iadd__)",
    ),
    decorator(
        "global-rebound-between-calls",
        setup="K = 1\n@cache\ndef f(x):\n    return x * K\n",
        steps=["f(2)", "K = 3", "f(2)"],
        source="reviews/aug-assign-dunder/review.md item 5 (global read by cached fn; adapted)",
    ),
    notebook(
        "thread-filled-list-read-again",
        cells=[
            "import threading\ndata = []\nev = threading.Event()\ndef worker():\n    ev.wait()\n    data.extend([1, 2, 3])\nt = threading.Thread(target=worker, daemon=True)\nt.start()",
            "snap = len(data)\nprint(f'snap={snap}')",
            "ev.set()\nt.join()\nprint(f'n={len(data)}')",
        ],
        names=["snap"],
        plan=["all", 1, "restart", "all", 1],
        source="tests/test_notebook_integration/language/test_async_and_threads.py::test_background_thread_event_gated_list_isolated_rerun",
    ),
]
