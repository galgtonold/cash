"""fiv triage shapes: corrected / variant decorator shapes (functions live in a real file)."""

from tools.plain_checker import decorator

_PRE = "import os, sys, importlib\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\n"


def _mod(name, src):
    return (
        _PRE
        + f"SRC = {src!r}\n"
        + f"if not os.path.exists('{name}.py') or open('{name}.py').read() != SRC:\n"
        + f"    with open('{name}.py', 'w') as fh:\n        fh.write(SRC)\n"
        + f"import {name}\n"
    )


_ENV_SRC = "import os, getpass\ndef who_inline():\n    return os.environ.get('ZZ_NB_TENANT')\ndef who_lib():\n    return getpass.getuser()\ndef miss(i):\n    return i\n"
_ENV_SETUP = (
    _mod("fiv_env", _ENV_SRC)
    + "import time\nwho_inline = cache(fiv_env.who_inline)\nwho_lib = cache(fiv_env.who_lib)\nmiss = cache(fiv_env.miss)\n"
)
_MISSES = "for _i in range(600):\n    miss(time.time_ns() * 1000 + _i)"

_SP_SETUP = (
    _ENV_SETUP
    + "SP = os.path.join(os.getcwd(), 'fake', 'site-packages')\nos.makedirs(SP, exist_ok=True)\n"
    + "PKG = \"import os\\ndef tenant():\\n    return os.environ.get('ZZ_NB_TENANT')\\n\"\n"
    + "if not os.path.exists(os.path.join(SP, 'zzpkg.py')) or open(os.path.join(SP, 'zzpkg.py')).read() != PKG:\n"
    + "    with open(os.path.join(SP, 'zzpkg.py'), 'w') as fh:\n        fh.write(PKG)\n"
    + "sys.path.insert(0, SP)\nimport zzpkg\n"
    + "def _who_pkg():\n    return zzpkg.tenant()\n"
)
_SP_SETUP = _mod("fiv_wp", "import zzpkg\ndef who_pkg():\n    return zzpkg.tenant()\n").replace(_PRE, "", 1)
_SP_SETUP = (
    _ENV_SETUP
    + "SP = os.path.join(os.getcwd(), 'fake', 'site-packages')\nos.makedirs(SP, exist_ok=True)\n"
    + "PKG = \"import os\\ndef tenant():\\n    return os.environ.get('ZZ_NB_TENANT')\\n\"\n"
    + "if not os.path.exists(os.path.join(SP, 'zzpkg.py')) or open(os.path.join(SP, 'zzpkg.py')).read() != PKG:\n"
    + "    with open(os.path.join(SP, 'zzpkg.py'), 'w') as fh:\n        fh.write(PKG)\n"
    + "sys.path.insert(0, SP)\n"
    + _SP_SETUP
    + "who_pkg = cache(fiv_wp.who_pkg)\n"
)
_MISSES_GETS = "for _i in range(600):\n    miss(time.time_ns() * 1000 + _i)\n    for _j in range(20):\n        os.environ.get('HOME')"

SHAPES = [
    decorator(
        "fiv-env-inline-filefn",
        setup=_ENV_SETUP,
        steps=[
            "os.environ['ZZ_NB_TENANT'] = 'acme'",
            "who_inline()",
            "os.environ['ZZ_NB_TENANT'] = 'globex'",
            "who_inline()",
        ],
        source="class-patches-stay F1 control: inline env read, fn in a real file, no toggles",
    ),
    decorator(
        "fiv-env-lib-reader",
        setup=_ENV_SETUP,
        steps=["os.environ['LOGNAME'] = 'acme'", "who_lib()", "os.environ['LOGNAME'] = 'globex'", "who_lib()"],
        source="class-patches-stay F1 control: library (stdlib getpass) env read, no toggles",
    ),
    decorator(
        "fiv-env-lib-reader-after-600-misses",
        setup=_ENV_SETUP,
        steps=[_MISSES, "os.environ['LOGNAME'] = 'acme'", "who_lib()", "os.environ['LOGNAME'] = 'globex'", "who_lib()"],
        source="class-patches-stay F1 deco mode: 600 cached misses, then a library reader's os.environ.get",
    ),
    decorator(
        "fiv-env-inline-after-600-misses",
        setup=_ENV_SETUP,
        steps=[
            _MISSES,
            "os.environ['ZZ_NB_TENANT'] = 'acme'",
            "who_inline()",
            "os.environ['ZZ_NB_TENANT'] = 'globex'",
            "who_inline()",
        ],
        source="class-patches-stay F1 deco mode with the inline reader",
    ),
    decorator(
        "fiv-global-object-attr-changed-filefn",
        setup=_mod(
            "fiv_obj",
            "class Holder:\n    def __init__(self):\n        self.kept = 1\nOBJ = Holder()\ndef f(x):\n    return OBJ.kept + x\n",
        )
        + "f = cache(fiv_obj.f)\n",
        steps=["f(1)", "fiv_obj.OBJ.kept = 5", "f(1)"],
        source="global-object-attr-changed with the fn in a real file",
    ),
    decorator(
        "fiv-lazy-dict-property-runs",
        setup=_mod(
            "fiv_lazy",
            "RUNS = [0]\nclass Lazy:\n    def __init__(self):\n        self.kept = 1\n    @property\n    def __dict__(self):\n        RUNS[0] += 1\n        return {'kept': 1}\nOBJ = Lazy()\ndef f(x):\n    return OBJ.kept + x\n",
        )
        + "f = cache(fiv_lazy.f)\n",
        steps=["fiv_lazy.RUNS[0] = 0", "f(1)", "fiv_lazy.RUNS[0]", "f(1)", "fiv_lazy.RUNS[0]"],
        source="reviews/label-and-own-dict/review.md B2 repro (Lazy computed __dict__ counter, OBJ.kept + x)",
    ),
    decorator(
        "fiv-env-sitepkg-reader",
        setup=_SP_SETUP,
        steps=[
            "os.environ['ZZ_NB_TENANT'] = 'acme'",
            "who_pkg()",
            "os.environ['ZZ_NB_TENANT'] = 'globex'",
            "who_pkg()",
        ],
        source="class-patches-stay F1 control: reader in a site-packages path, no toggles",
    ),
    decorator(
        "fiv-env-sitepkg-reader-after-600-misses",
        setup=_SP_SETUP,
        steps=[
            _MISSES,
            "os.environ['ZZ_NB_TENANT'] = 'acme'",
            "who_pkg()",
            "os.environ['ZZ_NB_TENANT'] = 'globex'",
            "who_pkg()",
        ],
        source="reviews/class-patches-stay/review.md F1 deco mode: 600 @cache misses, then an installed package's os.environ.get",
    ),
    decorator(
        "fiv-env-sitepkg-reader-after-600-misses-gets",
        setup=_SP_SETUP,
        steps=[
            _MISSES_GETS,
            "os.environ['ZZ_NB_TENANT'] = 'acme'",
            "who_pkg()",
            "os.environ['ZZ_NB_TENANT'] = 'globex'",
            "who_pkg()",
        ],
        source="as above, with os.environ.get calls between misses (pure.py's pattern) so Mapping.get specialises",
    ),
]

#: The faithful ones; the rest were steps toward them.
SHAPES = [s for s in SHAPES if s.name in {"fiv-env-sitepkg-reader-after-600-misses", "fiv-lazy-dict-property-runs"}]
