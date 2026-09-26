"""Remote objects (S3, GCS, Azure, HTTP) as tracked cache dependencies.

A local file can be checked for change by reading it. A remote object cannot -
downloading it to see whether it moved defeats the point of caching. Object
stores answer that question directly instead: every object carries a
*validator* the store maintains itself - an ETag, a version id, a GCS
generation - that changes when the bytes change. :class:`RemoteFileDataSource`
reads that validator with a single metadata request and folds it into the cache
key.

Two properties fall out of using the store's own validator, and both matter:

* **It is identical on every machine.** A local path is a fact about one
  filesystem, so a cache keyed on one cannot travel; an ETag is a fact about the
  object, so a teammate's key matches yours. See ``docs/tutorials/feature-guides/sharing-caches.md``.
* **A hit costs no download.** The metadata request is tens of milliseconds
  against a ``GET`` that may be hundreds of megabytes, so tracking a remote read
  usually *reduces* network traffic rather than adding to it.

Usage::

    import cash
    from cash import RemoteFileDataSource

    @cash.cache(depends_on=[RemoteFileDataSource("s3://bucket/events.parquet")])
    def load_events():
        return pd.read_parquet("s3://bucket/events.parquet")

``http://`` and ``https://`` need nothing beyond the standard library. Other
schemes are resolved through fsspec and its filesystem for that scheme
(``s3fs`` for ``s3://``, ``gcsfs`` for ``gs://``, and so on); a missing one
raises :class:`~cash.exceptions.DependencyNotFoundError`.
"""

from __future__ import annotations

import contextvars
import hashlib
import itertools
import logging
import time

# ``urllib.request`` drags in http.client, email and ssl - a real cost on every
# ``import cash``, paid by everyone whether or not they touch a URL. It is
# imported inside the HTTP path instead. ``urllib.parse`` is cheap and needed to
# recognise a scheme, so it stays up here.
import urllib.parse
from typing import Any

from ._active import active_config
from ._clock import perf_counter as _perf_counter
from ._memo import REMOTE_URLS, LruMemo
from .data_source import DataSource
from .diagnostics import warn_diagnostic
from .exceptions import CashCacheIneffectiveWarning, DependencyNotFoundError

__all__ = ["RemoteFileDataSource"]

logger = logging.getLogger(__name__)

#: Seconds to wait for the metadata request before giving up and recomputing.
DEFAULT_TIMEOUT = 10.0

# Info/header keys that carry a real validator, best first. ETag is preferred
# over a version id because it changes on *content*, whereas a version id also
# changes on a no-op re-upload - both are correct, ETag just recomputes less.
_STRONG_INFO_KEYS = (
    "ETag",
    "etag",
    "VersionId",
    "version_id",
    "generation",
    "Generation",
    "content_settings_etag",
)

# Query/fragment keys that pin a URL to one specific object *version*. Unlike a
# user-supplied "this looks immutable" pattern, this is not a guess: the storage
# contract says a pinned version's bytes cannot change, so reading the pin is
# reading a guarantee. S3 uses ``?versionId=``, GCS ``#generation=``.
_PINNED_KEYS = frozenset({"versionid", "generation"})

# HEAD is the right request, but presigned URLs are signed for a single method
# and some CDNs/origins simply refuse it. These codes mean "not HEAD" rather
# than "not there", so we retry for a single byte and read the headers off that.
_HEAD_REJECTED = frozenset({403, 405, 501})


class RemoteLedger:
    """What this process remembers about remote objects, by URL.

    Warnings already given (once per URL and kind, each set bounded) and the
    tokens held inside a revalidation window. By URL rather than by source:
    auto-tracked reads build a fresh source per check, so per-source state
    would never survive to be used -- and those are exactly the reads whose
    owner has nowhere to put ``immutable=True``. Tokens are only kept when a
    window is configured, so the default (revalidate always) keeps none.
    """

    MAX = 1024

    def __init__(self) -> None:
        self.warned_failures: set[str] = set()
        self.warned_weak_tokens: set[str] = set()
        self.warned_validation_cost: set[str] = set()
        self.tokens: LruMemo[str, tuple[float, str]] = LruMemo(REMOTE_URLS)

    def first_time(self, marks: set[str], mark: str) -> bool:
        """Is *mark* new to *marks*? Records it, while the set has room."""
        if mark in marks:
            return False
        if len(marks) < self.MAX:
            marks.add(mark)
        return True

    def keep_token(self, url: str, token: str) -> None:
        self.tokens[url] = (time.monotonic(), token)

    def reset(self) -> None:
        """Forget everything, as a new session starts."""
        self.warned_failures.clear()
        self.warned_weak_tokens.clear()
        self.warned_validation_cost.clear()
        self.tokens.clear()
        _READ_OPTIONS.clear()


REMOTE_LEDGER = RemoteLedger()


# ---------------------------------------------------------------------------
# The options a read was made with
#
# `pd.read_parquet("s3://b/k", storage_options={"client_kwargs": {"endpoint_url":
# ...}})` -- MinIO, an on-prem store, a named profile. The freshness check must
# ask the same store the read did: asked without the options it went to AWS
# with the ambient credentials, failed, and the call never cached, with a
# warning that blamed the user's access. (Had a bucket of the same name
# existed there, its ETag would have vouched for the wrong object.)
#
# Two halves. What names the store (endpoint, region, profile, anonymous
# access) is written into the entry, so a later process asks the same store.
# Credentials are not: they are kept in this process only, by URL, and used
# while what the entry names still matches them.
# ---------------------------------------------------------------------------

#: The options that say WHICH store and how to address it, as a tree of the
#: keys fsspec filesystems take. Only these are written to disk; a key not
#: listed here (``key``, ``secret``, ``token``, ``account_key``, ``headers``,
#: ...) never leaves the process.
_ADDRESSING_OPTIONS: dict[str, Any] = {
    "anon": None,
    "profile": None,
    "endpoint_url": None,
    "region_name": None,
    "requester_pays": None,
    "version_aware": None,
    "use_ssl": None,
    "project": None,
    "account_name": None,
    "client_kwargs": {"endpoint_url": None, "region_name": None},
    "config_kwargs": {"signature_version": None, "s3": {"addressing_style": None}},
}

_SCALARS = (str, int, float, bool, type(None))

#: The full options of the last tracked read of each URL, credentials included.
_READ_OPTIONS: LruMemo[str, dict[str, Any]] = LruMemo(REMOTE_URLS)


def addressing_options(options: Any, shape: dict[str, Any] | None = None) -> dict[str, Any]:
    """The part of *options* that names the store, safe to write to disk."""
    shape = _ADDRESSING_OPTIONS if shape is None else shape
    kept: dict[str, Any] = {}
    if not isinstance(options, dict):
        return kept
    for key, sub in shape.items():
        if key not in options:
            continue
        value = options[key]
        if sub is None:
            if isinstance(value, _SCALARS):
                kept[key] = value
        else:
            nested = addressing_options(value, sub)
            if nested:
                kept[key] = nested
    return kept


def remember_read_options(url: str, options: dict[str, Any]) -> None:
    """Record the options a tracked read of *url* was made with.

    An entry already held keeps its credentials when *options* names the same
    store: a cache hit replays only what the entry wrote down.
    """
    held = _READ_OPTIONS.get(url)
    if held is not None and addressing_options(held) == addressing_options(options):
        if len(held) >= len(options):
            return
    _READ_OPTIONS[url] = dict(options)


def read_options(url: str, recorded: dict[str, Any] | None = None) -> dict[str, Any]:
    """The options to check *url* with: those this process read it with,
    while they name the store the entry *recorded*; else what it recorded."""
    held = _READ_OPTIONS.get(url)
    if held is not None and (recorded is None or addressing_options(held) == recorded):
        return held
    return dict(recorded or {})


# Distinguishes one failed resolution from the next so the key genuinely moves.
# See ``_unresolved_token`` for why that is the failure behaviour.
_failure_serial = itertools.count()


# ---------------------------------------------------------------------------
# Validation cost
#
# With ``depends_on=`` chains one call can fan out to many sources, and until
# it is measured nothing shows where the time went. The COUNT is the actionable
# half: it is what tells you to swap N HEADs for one prefix LIST.
#
# The accumulator is a ContextVar so concurrent calls - asyncio tasks, threads -
# each measure their own work instead of stealing each other's numbers.
# ---------------------------------------------------------------------------

_validation_cost: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar(
    "cash_remote_validation_cost", default=None
)


class measured_validation:
    """Context manager recording token resolutions inside the block.

    Exposes ``count`` and ``seconds`` after exit. Nests safely: an inner block
    measures its own work and adds it to the enclosing one.

    Pass *sink* (a timing-breakdown dict) to have the result written on exit
    under ``<key>`` and ``<key>_count``. Writing from ``__exit__`` rather than
    after the block is what makes the numbers survive an early ``return`` out
    of the middle of the measured region.
    """

    def __init__(self, sink: dict[str, float] | None = None, key: str = "remote_validate") -> None:
        self.count = 0
        self.seconds = 0.0
        self._sink = sink
        self._key = key
        self._token: contextvars.Token | None = None
        self._outer: list[float] | None = None
        self._cell: list[float] = [0.0, 0.0]

    def __enter__(self) -> measured_validation:
        self._outer = _validation_cost.get()
        self._token = _validation_cost.set(self._cell)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._token is not None:
            _validation_cost.reset(self._token)
        self.count = int(self._cell[0])
        self.seconds = self._cell[1]
        if self._outer is not None:
            self._outer[0] += self._cell[0]
            self._outer[1] += self._cell[1]
        if self._sink is not None and self.count:
            self._sink[self._key] = self._sink.get(self._key, 0.0) + self.seconds
            self._sink[f"{self._key}_count"] = self._sink.get(f"{self._key}_count", 0) + self.count


def _record_validation(seconds: float) -> None:
    """Add one resolution to the enclosing measurement, if any."""
    cell = _validation_cost.get()
    if cell is not None:
        cell[0] += 1
        cell[1] += seconds


# Two independent ways validation can be a bad deal, because they catch
# different failures. RELATIVE catches "you paid more than you saved". ABSOLUTE
# catches the case that is technically net-positive and still miserable: 8 s of
# HEAD requests to save 60 s is a win on paper and unusable in a notebook.
# Either alone triggers the warning.
#
# The relative rule carries a floor, the same shape the cost model uses for its
# own gate (``max(fixed_budget, ratio x time)``): without it, a 100 ms function
# would warn about 60 ms of validation, which is true and useless.
VALIDATION_WARN_RATIO = 0.5
VALIDATION_WARN_FLOOR_SECONDS = 0.25
VALIDATION_WARN_ABSOLUTE_SECONDS = 2.0


def validation_is_expensive(seconds: float, saved_seconds: float | None) -> bool:
    """Whether *seconds* of validation is a bad trade against *saved_seconds*."""
    if seconds > VALIDATION_WARN_ABSOLUTE_SECONDS:
        return True
    if not saved_seconds or saved_seconds <= 0:
        return False
    return seconds > max(VALIDATION_WARN_FLOOR_SECONDS, VALIDATION_WARN_RATIO * saved_seconds)


def warn_validation_cost_once(label: str, count: int, seconds: float, saved_seconds: float | None) -> None:
    """Warn a single time per *label* that freshness checking is costing real time."""
    if not REMOTE_LEDGER.first_time(REMOTE_LEDGER.warned_validation_cost, label):
        return
    saved = f", against {saved_seconds:.2f}s of compute it avoids" if saved_seconds and saved_seconds > 0 else ""
    warn_diagnostic(
        CashCacheIneffectiveWarning,
        "REMOTE-FRESHNESS-COST",
        f"cash spent {seconds:.2f}s checking {count} remote "
        f"{'source' if count == 1 else 'sources'} for freshness on {label}{saved}, "
        f"so proving the result fresh costs more than recomputing it would.",
        "mark sources that cannot change with "
        "RemoteFileDataSource(..., immutable=True) or pin a version in the URL; "
        "for reads cash tracked automatically, widen the window with "
        "cash.configure(remote_revalidate_max_age_seconds=...), accepting that "
        "a change goes unnoticed for that long.",
    )


class _NoTokenError(ValueError):
    """The object was reachable but exposed nothing that tracks its content."""


def pinned_version(url: str) -> str | None:
    """Return the pinned object version in *url*, or ``None`` if unpinned.

    ``s3://bucket/key?versionId=abc`` and ``gs://bucket/key#generation=17`` each
    name one immutable version of an object, so the pin *is* the state token and
    no request is needed to read it.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    for blob in (parts.query, parts.fragment):
        if not blob:
            continue
        for key, value in urllib.parse.parse_qsl(blob):
            if key.lower() in _PINNED_KEYS and value:
                return f"{key.lower()}:{value}"
    return None


def _warn_weak_token(url: str, detail: str) -> None:
    """Warn once that the only available validator is a size.

    Size alone misses any edit that keeps the byte count the same - a corrected
    value in a fixed-width column, a rewritten row - so the entry can go stale
    without the token moving.
    """
    if not REMOTE_LEDGER.first_time(REMOTE_LEDGER.warned_weak_tokens, url):
        return
    warn_diagnostic(
        CashCacheIneffectiveWarning,
        "REMOTE-SIZE-ONLY",
        f"cash is tracking {url!r} by size alone ({detail}): the store exposes "
        f"no ETag, version id or modification time, so an edit that keeps the "
        f"byte count will NOT invalidate the cached result.",
        "pin a version in the URL (?versionId=, #generation=), or write a "
        "DataSource whose token is something you control -- a run id, a digest "
        "from a manifest.",
    )


def _http_headers(url: str, timeout: float) -> Any:
    """Response headers for *url*, via HEAD with a one-byte GET fallback.

    Through an opener built here, which reads the proxy settings in effect
    now. ``urllib.request.urlopen`` reuses one opener that read them at the
    first ``urlopen`` in the process, so a proxy set later was ignored.
    """
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener()
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.headers
    except urllib.error.HTTPError as exc:
        if exc.code not in _HEAD_REJECTED:
            raise
        logger.debug("[REMOTE] HEAD %s rejected with %s; retrying ranged GET", url, exc.code)
    ranged = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0"})
    with opener.open(ranged, timeout=timeout) as response:
        return response.headers


def _http_token(url: str, timeout: float) -> str:
    """State token for an ``http(s)://`` object, from its cache validators."""
    headers = _http_headers(url, timeout)
    etag = headers.get("ETag")
    if etag:
        # A weak validator (``W/"abc"``) is kept verbatim: it still changes with
        # semantic content, which is exactly what the cache key needs.
        return f"etag:{etag.strip()}"
    last_modified = headers.get("Last-Modified")
    length = headers.get("Content-Length")
    if last_modified:
        return f"mtime:{last_modified}|size:{length}"
    if length is not None:
        _warn_weak_token(url, "no ETag or Last-Modified header")
        return f"size:{length}"
    raise _NoTokenError("the response carried no ETag, Last-Modified or Content-Length header")


#: The package that gives fsspec its filesystem for a scheme. Each one
#: installs fsspec too.
_FSSPEC_PACKAGES = {
    "s3": "s3fs",
    "s3a": "s3fs",
    "gs": "gcsfs",
    "gcs": "gcsfs",
    "az": "adlfs",
    "abfs": "adlfs",
    "abfss": "adlfs",
}


def _missing_filesystem(url: str, exc: ImportError) -> DependencyNotFoundError:
    """The error for a scheme whose fsspec filesystem is not installed."""
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    package = _FSSPEC_PACKAGES.get(scheme)
    if package is not None:
        fix = f"fsspec's {scheme} filesystem, the {package} package: pip install {package}"
    else:
        fix = f"fsspec and its filesystem for {scheme}:// (fsspec says: {exc})"
    return DependencyNotFoundError(
        f"Tracking {scheme}:// objects needs {fix}. Or track the object with a custom DataSource."
    )


def _fsspec_token(url: str, storage_options: dict[str, Any]) -> str:
    """State token for any fsspec-addressable object, from its stat info."""
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - exercised via a stubbed import
        raise _missing_filesystem(url, exc) from exc

    try:
        fs, path = fsspec.core.url_to_fs(url, **storage_options)
    except ImportError as exc:  # fsspec is there, the scheme's filesystem is not
        raise _missing_filesystem(url, exc) from exc
    # s3fs answers `info` from a listing it cached earlier -- the reader's own,
    # made moments ago -- and would report the ETag the object had then.
    invalidate = getattr(fs, "invalidate_cache", None)
    if invalidate is not None:
        invalidate(path)
    if any(char in path for char in "*?["):
        return _listing_token(url, fs.glob(path, detail=True))
    info = fs.info(path)
    if info.get("type") == "directory":
        # A prefix (`pd.read_parquet("s3://b/p/")`) reads every object under
        # it. The prefix itself carries nothing: s3fs reports it as size 0
        # with no ETag, so a new partition never moved the token and the old
        # total was served for good.
        return _listing_token(url, fs.find(path, detail=True))
    token, weak = _info_token(info)
    if token is None:
        raise _NoTokenError("the filesystem reported no ETag, version, mtime or size")
    if weak:
        _warn_weak_token(url, "the filesystem reports no ETag, version or mtime")
    return token


def _info_token(info: dict[str, Any]) -> tuple[str | None, bool]:
    """``(token, size_only)`` for one object's fsspec info, strongest first."""
    for key in _STRONG_INFO_KEYS:
        value = info.get(key)
        if value:
            return f"{key.lower()}:{value}", False
    modified = info.get("LastModified") or info.get("last_modified") or info.get("mtime")
    size = info.get("size")
    if modified is not None:
        return f"mtime:{modified}|size:{size}", False
    if size is not None:
        return f"size:{size}", True
    return None, False


def _listing_token(url: str, entries: dict[str, dict[str, Any]]) -> str:
    """One token for every object under a prefix or matching a glob.

    Each object's own validator, with its name, so a new object, a removed
    one and an edited one all move it. One LIST request per check, however
    many objects there are (a page per thousand on S3).
    """
    digest = hashlib.sha256()
    count = 0
    weak = False
    for name in sorted(entries):
        info = entries[name]
        if info.get("type") == "directory":
            continue
        token, size_only = _info_token(info)
        if token is None:
            raise _NoTokenError(f"the filesystem reported no ETag, version, mtime or size for {name!r}")
        weak = weak or size_only
        digest.update(f"{name}\0{token}\n".encode())
        count += 1
    if weak:
        _warn_weak_token(url, "some objects under it report no ETag, version or mtime")
    return f"listing:{count}:{digest.hexdigest()}"


class RemoteFileDataSource(DataSource):
    """Tracks a remote object by the validator its store maintains.

    Args:
        url: The object's URL - ``s3://``, ``gs://``, ``az://``, ``http(s)://``,
            or anything else fsspec can address.
        immutable: Declare that this object's bytes can never change, so its
            token is resolved once per instance and every later check is free.
            Left as ``None`` (the default) it is inferred **only** when the URL
            pins a specific version (``?versionId=``, ``#generation=``), which
            the storage contract guarantees; it is never guessed from the path.
            Setting it wrongly means the entry never invalidates, silently, so
            declare it only for content-addressed or write-once data.
        max_age: Seconds a resolved token may be reused before the store is
            asked again. ``0`` (the default) defers to the cash-level
            ``remote_revalidate_max_age_seconds``, itself ``0`` = revalidate
            every time. Trading correctness for latency, so raise it
            deliberately: for the window's duration a changed object goes
            unnoticed. Windowed tokens are shared per URL, so they apply to
            automatically-tracked reads too.
        storage_options: Passed through to fsspec (credentials, endpoint,
            profile). Ignored for ``http(s)://``, which uses the standard library.
        timeout: Seconds to wait for the metadata request.

    Failure is **closed**: if the store cannot be reached or refuses the request,
    the source reports a value that has never been seen before, so the call
    recomputes rather than serving a result whose freshness could not be checked.
    A transient outage costs you the speedup; it never costs you correctness.
    """

    def __init__(
        self,
        url: str,
        *,
        immutable: bool | None = None,
        max_age: float = 0.0,
        storage_options: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.url = str(url)
        self.storage_options = dict(storage_options or {})
        self.timeout = timeout
        self.max_age = max_age
        self._pinned = pinned_version(self.url)
        self.immutable = bool(self._pinned) if immutable is None else bool(immutable)
        self._cached_token: str | None = None

    def get_id(self) -> str:
        """Identity of the dependency - the URL, so it matches across machines."""
        return f"remote:{self.url}"

    def state_token(self) -> str:
        """The validator folded into the cache key.

        Reads the store's ETag / version id / generation, honouring
        ``immutable`` and the revalidation window. Never raises: an unresolvable
        object yields a token that forces a recompute.
        """
        if self.immutable and self._cached_token is not None:
            # A promise about the object, made by whoever built THIS source -
            # so it is deliberately not shared with other sources for the same
            # URL, which may not have made it.
            return self._cached_token
        max_age = self._effective_max_age()
        if max_age > 0:
            entry = REMOTE_LEDGER.tokens.get(self.url)
            if entry is not None and (time.monotonic() - entry[0]) < max_age:
                return entry[1]
        started = _perf_counter()
        token = self._resolve()
        # A pin costs no request; counting it would inflate the very number a
        # user consults to decide whether validation is worth its price.
        if self._pinned is None:
            _record_validation(_perf_counter() - started)
        self._cached_token = token
        if max_age > 0:
            REMOTE_LEDGER.keep_token(self.url, token)
        return token

    def _effective_max_age(self) -> float:
        """This source's window, falling back to the cash-level default.

        The default comes from the settings that govern the running code
        (``active_config``): those of the ``Cash`` instance whose call is
        checking this source, so a source used by a ``Cash(...)`` of your own
        follows that instance, not the default one.
        """
        if self.max_age:
            return self.max_age
        try:
            return float(active_config().remote_revalidate_max_age_seconds)
        except Exception:  # noqa: BLE001 - a config problem must not break a read
            return 0.0

    def _resolve(self) -> str:
        """Read the token from the store, degrading to a recompute on failure."""
        if self._pinned is not None:
            # The URL already names the version - the answer is in our hand.
            return self._pinned
        try:
            scheme = urllib.parse.urlsplit(self.url).scheme.lower()
            if scheme in ("http", "https"):
                return _http_token(self.url, self.timeout)
            return _fsspec_token(self.url, self.storage_options)
        except DependencyNotFoundError:
            # A missing library is the caller's to fix, not a transient blip:
            # silently recomputing forever would hide it.
            raise
        except Exception as exc:  # noqa: BLE001 - remote clients raise a zoo of types
            self._warn_failure(exc)
            return self._unresolved_token()

    def _unresolved_token(self) -> str:
        """A token that has never been used before, so the call recomputes.

        A *constant* "unresolved" token would be worse than useless: the second
        failure would find the entry the first failure stored and serve it -
        precisely the stale hit this class exists to prevent. The cost is that
        each failed check leaves behind an entry no future call can reach; a
        store that is down long enough for that to matter has a bigger problem.
        """
        return f"unresolved:{self.url}:{next(_failure_serial)}"

    def _warn_failure(self, exc: BaseException) -> None:
        """Warn once per URL and failure kind that freshness went unverified."""
        ledger_key = f"{self.url}|{type(exc).__name__}"
        if not REMOTE_LEDGER.first_time(REMOTE_LEDGER.warned_failures, ledger_key):
            logger.debug("[REMOTE] %s still unresolvable: %s", self.url, exc)
            return
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "REMOTE-STATE-UNREADABLE",
            f"cash could not read the state of {self.url!r} to check whether the "
            f"cached result is still fresh ({type(exc).__name__}: {exc}), so the "
            f"call recomputes rather than risk serving a stale result.",
            "restore the access the exception names -- credentials, network, "
            "permissions -- and caching resumes on its own; nothing needs "
            "resetting.",
            # No stacklevel: warn_diagnostic blames the nearest frame outside
            # cash, so the warning names the user's line whether it came
            # through the decorator or from a direct state_token() call.
        )

    def __repr__(self) -> str:
        detail = f"{self.url!r}"
        if self.immutable:
            detail += ", immutable=True"
        if self.max_age:
            detail += f", max_age={self.max_age!r}"
        return f"RemoteFileDataSource({detail})"
