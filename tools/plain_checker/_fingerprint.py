"""A comparable text for a value, the same in any process: no object
addresses, arrays and frames by content. Plain stdlib at import, so the
kernel and the decorator sessions can load it from its source."""

from __future__ import annotations

import hashlib
import re

_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")
_LONG = 400


def _short(text: str) -> str:
    text = _ADDRESS.sub("0x?", text)
    if len(text) > _LONG:
        return text[:200] + "...sha1:" + hashlib.sha1(text.encode()).hexdigest()[:16]
    return text


def fingerprint(value, depth=0):
    kind = type(value)
    module = getattr(kind, "__module__", "") or ""
    try:
        if depth < 4 and kind in (list, tuple):
            return _short(kind.__name__ + "[" + ", ".join(fingerprint(v, depth + 1) for v in value) + "]")
        if depth < 4 and kind is dict:
            return _short("{" + ", ".join(f"{k!r}: {fingerprint(v, depth + 1)}" for k, v in value.items()) + "}")
        if depth < 4 and kind in (set, frozenset):
            return _short(kind.__name__ + "{" + ", ".join(sorted(fingerprint(v, depth + 1) for v in value)) + "}")
        if module.startswith("numpy") and kind.__name__ == "ndarray":
            if value.dtype.hasobject:
                return _short(f"ndarray{value.shape}object:{fingerprint(value.tolist(), depth + 1)}")
            import numpy

            data = numpy.ascontiguousarray(value).tobytes()
            return f"ndarray{value.shape}{value.dtype}:{hashlib.sha1(data).hexdigest()[:16]}"
        if module.startswith("pandas") and kind.__name__ in ("DataFrame", "Series"):
            text = value.to_csv() + "\n" + repr(list(map(str, getattr(value, "dtypes", [value.dtype]))))
            return f"{kind.__name__}{value.shape}:{hashlib.sha1(text.encode()).hexdigest()[:16]}"
    except Exception:
        pass
    try:
        return _short(repr(value))
    except Exception as exc:
        return f"<repr raised {type(exc).__name__}>"


def fingerprints(namespace, names):
    missing = "<unbound>"
    return {name: (fingerprint(namespace[name]) if name in namespace else missing) for name in names}
