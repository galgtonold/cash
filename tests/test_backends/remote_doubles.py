"""In-process stand-ins for S3 and Redis that COUNT round trips.

The remote backends are the two whose cost is not dominated by anything this
machine does. A local benchmark of them measures the loopback, and a benchmark
against a real endpoint measures whoever's network happened to run it -- so
neither answers the question that matters, which is **how many times does one
cache operation go to the other end?**

That number is machine-independent, so it can be asserted. These doubles are
functional enough for the real backend code to run against unmodified, and
they record every call. ``latency`` optionally sleeps per round trip, which
turns a count into an estimate for a given RTT without needing the RTT to be
real.

Deliberately not ``MagicMock``: a mock returns a mock for everything, so a
backend that fetched the wrong object, or fetched one twice, would still pass.
These store and return real bytes, so a test that asserts a count is also
asserting the operation worked.
"""
from __future__ import annotations

import fnmatch
import time
from collections import Counter


class _Recorder:
    """Shared call log for a double."""

    def __init__(self, latency: float = 0.0):
        self.calls: Counter[str] = Counter()
        self.bytes_out = 0          # bytes the double handed back to the caller
        self.latency = latency

    def record(self, op: str, payload: bytes = b"") -> None:
        self.calls[op] += 1
        self.bytes_out += len(payload)
        if self.latency:
            time.sleep(self.latency)

    @property
    def round_trips(self) -> int:
        return sum(self.calls.values())

    def reset(self) -> None:
        self.calls.clear()
        self.bytes_out = 0


class FakeS3Client(_Recorder):
    """The subset of the boto3 S3 client that ``S3Backend`` actually calls.

    Every method here is one HTTP request against a real endpoint, which is
    why each one records.
    """

    def __init__(self, latency: float = 0.0):
        super().__init__(latency)
        self.store: dict[tuple[str, str], bytes] = {}

    # -- the operations the backend uses -----------------------------------
    def put_object(self, Bucket, Key, Body, **_kw):  # noqa: N803 - boto3 casing
        self.record("put_object")
        self.store[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, Bucket, Key, Range=None, **_kw):  # noqa: N803
        try:
            blob = self.store[(Bucket, Key)]
        except KeyError:
            raise _client_error("NoSuchKey") from None
        if Range is not None:
            blob = blob[_parse_range(Range, len(blob))]
        self.record("get_object", blob)
        return {"Body": _Body(blob), "ContentLength": len(blob)}

    def delete_object(self, Bucket, Key, **_kw):  # noqa: N803
        self.record("delete_object")
        self.store.pop((Bucket, Key), None)
        return {}

    def delete_objects(self, Bucket, Delete, **_kw):  # noqa: N803
        """One request for up to 1000 keys. The batch form of delete_object."""
        self.record("delete_objects")
        for obj in Delete.get("Objects", []):
            self.store.pop((Bucket, obj["Key"]), None)
        return {}

    def list_objects_v2(self, Bucket, Prefix="", **_kw):  # noqa: N803
        self.record("list_objects_v2")
        return {"Contents": self._contents(Bucket, Prefix),
                "KeyCount": len(self._contents(Bucket, Prefix))}

    def get_paginator(self, operation_name):
        if operation_name != "list_objects_v2":
            raise NotImplementedError(operation_name)
        return _FakePaginator(self)

    def _contents(self, bucket, prefix):
        return [{"Key": k, "Size": len(v)}
                for (b, k), v in self.store.items()
                if b == bucket and k.startswith(prefix)]


class _FakePaginator:
    """One page, one request -- which is what a real paginator costs."""

    def __init__(self, client: FakeS3Client):
        self._client = client

    def paginate(self, Bucket, Prefix="", **_kw):  # noqa: N803
        self._client.record("list_objects_v2")
        contents = self._client._contents(Bucket, Prefix)
        yield {"Contents": contents} if contents else {"KeyCount": 0}


class _Body:
    def __init__(self, blob: bytes):
        self._blob = blob

    def read(self, n=-1):
        return self._blob


def _parse_range(header: str, size: int) -> slice:
    """``bytes=0-127`` -> slice(0, 128). Open-ended end clamps to the object."""
    spec = header.split("=", 1)[1]
    start_s, _, end_s = spec.partition("-")
    start = int(start_s)
    end = int(end_s) if end_s else size - 1
    return slice(start, min(end + 1, size))


def _client_error(code: str):
    import botocore.exceptions
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": code}}, "GetObject")


class FakeRedisClient(_Recorder):
    """Redis, where a PIPELINE is one round trip however many commands it holds.

    That distinction is the whole point of counting here: the backend already
    pipelines its two-key reads and writes, so they cost one round trip, and a
    change that split them would be invisible to a test that counted commands
    instead.
    """

    def __init__(self, latency: float = 0.0):
        super().__init__(latency)
        self.store: dict[str, bytes] = {}

    def pipeline(self):
        return _FakePipeline(self)

    def get(self, key):
        blob = self.store.get(key)
        self.record("get", blob or b"")
        return blob

    def mget(self, keys):
        blobs = [self.store.get(k) for k in keys]
        self.record("mget", b"".join(b for b in blobs if b))
        return blobs

    def set(self, key, value, **_kw):
        self.record("set")
        self.store[key] = bytes(value)
        return True

    def delete(self, *keys):
        self.record("delete")
        for k in keys:
            self.store.pop(k, None)
        return len(keys)

    def expire(self, key, ttl):
        self.record("expire")
        return True

    def scan(self, cursor=0, match="*", count=100):
        self.record("scan")
        return 0, [k for k in list(self.store) if fnmatch.fnmatch(k, match)]

    def scan_iter(self, match="*", count=100):
        self.record("scan_iter")
        return iter([k for k in list(self.store) if fnmatch.fnmatch(k, match)])

    def ping(self):
        self.record("ping")
        return True


class _FakePipeline:
    """Queues commands and spends exactly one round trip on ``execute``."""

    def __init__(self, client: FakeRedisClient):
        self._client = client
        self._queued: list[tuple[str, tuple]] = []

    def get(self, key):
        self._queued.append(("get", (key,)))
        return self

    def set(self, key, value, **_kw):
        self._queued.append(("set", (key, value)))
        return self

    def expire(self, key, ttl):
        self._queued.append(("expire", (key, ttl)))
        return self

    def delete(self, *keys):
        self._queued.append(("delete", keys))
        return self

    def execute(self):
        results = []
        out = bytearray()
        for op, args in self._queued:
            if op == "get":
                blob = self._client.store.get(args[0])
                results.append(blob)
                if blob:
                    out += blob
            elif op == "set":
                self._client.store[args[0]] = bytes(args[1])
                results.append(True)
            elif op == "expire":
                results.append(True)
            elif op == "delete":
                for k in args[0]:
                    self._client.store.pop(k, None)
                results.append(len(args[0]))
        self._queued.clear()
        self._client.record("pipeline.execute", bytes(out))
        return results
