"""Stable value digests for the round-34 replay harness.

Imported in two places: inside the replay kernel (namespace digests after each
step) and by the plain-mode call hook of decorator scripts (argument digests).
Standard library only at import time; pandas / numpy / polars / sklearn are
used only when a value of that type shows up.

A digest is a short hex string. It is equal for equal values within one Python
environment (same package versions), which is all the oracle needs: plain and
cash runs use the same environment.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import types

_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_PRIM = (int, float, bool, complex, str, bytes, type(None))
_MAX_DEPTH = 6
_SKIP_NAMES = {"In", "Out", "get_ipython", "exit", "quit", "open"}


def _h(*parts) -> "hashlib._Hash":
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        if isinstance(p, str):
            p = p.encode("utf-8", "surrogatepass")
        h.update(p)
        h.update(b"\x00")
    return h


def _hex(*parts) -> str:
    return _h(*parts).hexdigest()


def _tname(v) -> str:
    t = type(v)
    return f"{t.__module__}.{t.__qualname__}"


def _safe_repr(v) -> str:
    return _ADDR.sub("0x?", repr(v))


def _np_digest(a) -> str:
    import numpy as np

    if a.dtype.hasobject:
        return _hex("nd", str(a.dtype), repr(a.shape), digest(a.tolist()))
    a2 = np.ascontiguousarray(a)
    h = _h("nd", str(a.dtype), repr(a.shape))
    h.update(memoryview(a2).cast("B") if a2.size else b"")
    return h.hexdigest()


def _pd_digest(v, kind: str) -> str:
    import pandas as pd

    meta = [kind]
    if kind == "DataFrame":
        meta += [repr(list(map(str, v.columns))), repr([str(x) for x in v.dtypes]), repr(v.shape)]
    elif kind == "Series":
        meta += [str(v.name), str(v.dtype), repr(v.shape)]
    else:  # Index
        meta += [str(v.dtype), repr(v.shape)]
    try:
        if kind == "Index":
            hv = pd.util.hash_pandas_object(v)
        else:
            hv = pd.util.hash_pandas_object(v, index=True)
        h = _h(*meta)
        h.update(memoryview(hv.to_numpy()).cast("B") if len(hv) else b"")
        if kind != "Index":
            h.update(_pd_digest(v.index, "Index").encode())
        return h.hexdigest()
    except Exception:
        # Unhashable cells (lists, dicts): fall back to the values as Python objects.
        try:
            obj = v.to_dict() if kind != "DataFrame" else v.to_dict(orient="split")
            return _hex(*meta, digest(obj))
        except Exception:
            return _hex(*meta, "unhashable")


def _pl_digest(v, kind: str) -> str:
    try:
        if kind == "DataFrame":
            hv = v.hash_rows(seed=0)
            meta = [repr(v.schema), repr(v.shape)]
        elif kind == "LazyFrame":
            return _hex("LazyFrame", repr(v.collect_schema()))
        else:
            hv = v.hash(seed=0)
            meta = [str(v.name), str(v.dtype), repr(v.shape)]
        h = _h("pl", kind, *meta)
        h.update(memoryview(hv.to_numpy()).cast("B") if len(hv) else b"")
        return h.hexdigest()
    except Exception as e:  # noqa: BLE001
        return _hex("pl", kind, "ERR", type(e).__name__)


_SK_FITTED = ("coef_", "intercept_", "feature_importances_", "classes_", "n_features_in_")


def _sk_digest(v, depth: int) -> str:
    parts = ["sk", _tname(v)]
    try:
        params = v.get_params(deep=False)
        parts.append(digest({k: params[k] for k in sorted(params)}, depth + 1))
    except Exception as e:  # noqa: BLE001
        parts.append("params-ERR " + type(e).__name__)
    fitted = {}
    for k, val in sorted(vars(v).items()):
        if k.endswith("_") and not k.startswith("_"):
            fitted[k] = val
    for k in _SK_FITTED:
        if k not in fitted:
            try:
                fitted[k] = getattr(v, k)
            except Exception:  # noqa: BLE001 - not fitted, or not this kind of model
                pass
    for k in sorted(fitted):
        parts.append(k)
        parts.append(digest(fitted[k], depth + 1))
    return _hex(*parts)


def digest(v, depth: int = 0, _seen=None) -> str:
    """A stable digest of *v*; see the module docstring for the rules."""
    if depth > _MAX_DEPTH:
        return _hex("deep", _tname(v))
    t = type(v)
    mod = t.__module__ or ""
    name = t.__name__
    try:
        if v is None or t in (bool, int, str, bytes, complex):
            return _hex("p", name, repr(v))
        if t is float:
            return _hex("f", repr(v))
        if mod.startswith("pandas"):
            if name in ("DataFrame", "Series"):
                return _pd_digest(v, name)
            if "Index" in name:
                return _pd_digest(v, "Index")
            if name in ("Timestamp", "Timedelta", "Period", "Interval"):
                return _hex("p", name, repr(v))
        if mod.startswith("numpy"):
            if name in ("ndarray", "memmap"):
                return _np_digest(v)
            if hasattr(v, "dtype") and hasattr(v, "item"):  # numpy scalar
                return _hex("npscalar", str(v.dtype), repr(v.item()))
            if name.endswith("Generator") or name == "RandomState":
                return _hex("rng", name, digest(v.bit_generator.state if hasattr(v, "bit_generator") else v.get_state(), depth + 1))
        if mod.startswith("polars"):
            if name in ("DataFrame", "Series", "LazyFrame"):
                return _pl_digest(v, name)
        if mod.startswith("scipy.sparse") and hasattr(v, "tocsr"):
            c = v.tocsr()
            return _hex("sparse", repr(c.shape), _np_digest(c.data), _np_digest(c.indices), _np_digest(c.indptr))
        if hasattr(v, "get_params") and hasattr(v, "fit"):  # sklearn-style estimators (also lightgbm/xgboost/catboost wrappers)
            return _sk_digest(v, depth)
        if name == "Tree" and mod.startswith("sklearn"):
            return _hex("sktree", digest(v.__getstate__(), depth + 1))
        if isinstance(v, (list, tuple)):
            if len(v) > 5000 and all(type(x) in _PRIM for x in v[:50]):
                return _hex("seqbig", name, _safe_repr(v))
            return _hex("seq", name, *(digest(x, depth + 1) for x in v))
        if isinstance(v, (set, frozenset)):
            return _hex("set", name, *sorted(digest(x, depth + 1) for x in v))
        if isinstance(v, dict):
            return _hex("dict", name, *(digest(k, depth + 1) + ":" + digest(x, depth + 1) for k, x in v.items()))
        if isinstance(v, os.PathLike):
            return _hex("path", os.fspath(v))
        if isinstance(v, (range, slice)):
            return _hex("p", name, repr(v))
        if mod in ("datetime", "decimal", "fractions", "uuid") or isinstance(v, (types.SimpleNamespace,)):
            return _hex("p", name, _safe_repr(v))
        import dataclasses
        import enum

        if isinstance(v, enum.Enum):
            return _hex("enum", _tname(v), repr(v.value))
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return _hex("dc", _tname(v), *(f.name + "=" + digest(getattr(v, f.name, None), depth + 1) for f in dataclasses.fields(v)))
    except Exception as e:  # noqa: BLE001
        return _hex("ERR", _tname(v), type(e).__name__)
    return _hex("type", _tname(v))


# ---------------------------------------------------------------- code identity (for the Z2 bound only)

_file_memo: dict = {}


def file_digest(path: str) -> str | None:
    """Content digest of a file (memoised on size + mtime), or None."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, st.st_mtime_ns)
    d = _file_memo.get(key)
    if d is None:
        h = hashlib.blake2b(digest_size=12)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        d = h.hexdigest()
        _file_memo[key] = d
    return d


def path_digest(p: str) -> str | None:
    """Digest of a file, or of a directory's files (first 2000), or None."""
    if os.path.isfile(p):
        return file_digest(p)
    if os.path.isdir(p):
        parts = []
        for root, dirs, files in os.walk(p):
            dirs.sort()
            for f in sorted(files):
                fp = os.path.join(root, f)
                parts.append(os.path.relpath(fp, p) + "=" + str(file_digest(fp)))
                if len(parts) > 2000:
                    break
        return _hex("dir", *parts)
    return None


def code_digest(co, depth: int = 0) -> str:
    parts = [co.co_code, repr(co.co_names), repr(co.co_varnames)]
    for c in co.co_consts:
        if isinstance(c, types.CodeType) and depth < 8:
            parts.append(code_digest(c, depth + 1))
        else:
            parts.append(_safe_repr(c))
    return _hex(*parts)


def _all_names(co, out=None):
    out = set() if out is None else out
    out.update(co.co_names)
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            _all_names(c, out)
    return out


def _under_cwd(path: str | None) -> bool:
    if not path:
        return False
    try:
        return os.path.abspath(path).startswith(os.getcwd() + os.sep)
    except Exception:  # noqa: BLE001
        return False


def func_identity(f, depth: int = 0) -> str:
    """Code of *f*, its defaults, and what the globals it reads hold: plain
    values by digest, functions by their own identity (3 levels), modules by
    module_identity (source file digest for project modules)."""
    f = getattr(f, "__wrapped__", f)
    co = getattr(f, "__code__", None)
    if co is None:
        return _hex("callable", _tname(f), getattr(f, "__qualname__", ""))
    parts = [getattr(f, "__module__", "") or "", getattr(f, "__qualname__", ""), code_digest(co)]
    g = getattr(f, "__globals__", {}) or {}
    for n in sorted(_all_names(co)):
        if n in g:
            val = g[n]
            if isinstance(val, types.ModuleType):
                parts.append(n + "=" + module_identity(val))
            elif isinstance(val, types.FunctionType) or hasattr(val, "__wrapped__"):
                if depth < 3:
                    parts.append(n + "=" + func_identity(val, depth + 1))
            elif isinstance(val, type):
                parts.append(n + "=" + _tname(val))
            elif not callable(val):
                parts.append(n + "=" + digest(val, 2))
    if getattr(f, "__defaults__", None):
        parts.append(digest(f.__defaults__, 2))
    return _hex(*parts)


def module_identity(m) -> str:
    f = getattr(m, "__file__", None)
    if _under_cwd(f):
        return _hex("mod", m.__name__, str(file_digest(f)))
    return _hex("mod", m.__name__, str(getattr(m, "__version__", "")))


def namespace_state(ns: dict) -> str:
    """JSON of {"v": {name: digest}, "c": {name: code identity}, "s": {name: short str}}
    over the user names of a notebook namespace."""
    import json
    import time

    t0 = time.perf_counter()
    vals, codes, strs = {}, {}, {}
    for k, v in list(ns.items()):
        if k.startswith("_") or k in _SKIP_NAMES:
            continue
        if k == "cash" and isinstance(v, types.ModuleType):
            continue  # the harness's own `import cash` in cash mode
        try:
            if isinstance(v, types.ModuleType):
                vals[k] = "module"
                codes[k] = module_identity(v)
                continue
            if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)) or (
                callable(v) and hasattr(v, "__wrapped__")
            ):
                vals[k] = _hex("type", _tname(v))
                codes[k] = func_identity(v)
                continue
            if isinstance(v, type):
                vals[k] = _hex("class", v.__qualname__)
                codes[k] = _hex("cls", v.__qualname__, *(func_identity(x) for x in vars(v).values() if isinstance(x, types.FunctionType)))
                continue
            vals[k] = digest(v)
            if isinstance(v, str) and len(v) < 500:
                strs[k] = v
            elif isinstance(v, os.PathLike):
                strs[k] = os.fspath(v)
        except Exception as e:  # noqa: BLE001
            vals[k] = "ERR " + type(e).__name__
    return json.dumps({"v": vals, "c": codes, "s": strs, "secs": round(time.perf_counter() - t0, 4)})


def args_digest(args, kwargs) -> str:
    """Digest of a call's arguments; a str/PathLike that names an existing
    path also carries that path's content digest."""
    parts = []
    for x in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        parts.append(digest(x))
        if isinstance(x, (str, os.PathLike)):
            s = os.fspath(x)
            if isinstance(s, str) and 0 < len(s) < 4096 and "\n" not in s:
                try:
                    pd_ = path_digest(s)
                except Exception:  # noqa: BLE001
                    pd_ = None
                if pd_:
                    parts.append("file=" + pd_)
    parts.append(repr(sorted(kwargs)))
    return _hex("args", *parts)
