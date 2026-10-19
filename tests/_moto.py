"""An in-process S3 emulator, warmed before any test's clock runs.

moto loads the S3 backend (and IAM, events, cloudwatch with it) on the
server's first request. On a cold, loaded machine that import alone took most
of the 30 s test timeout, inside the first test's fixture. Importing it here,
at collection, and making one request when the server starts keeps that out of
what a test is timed on.
"""

from __future__ import annotations

import contextlib

import pytest

try:
    import moto.s3.responses  # the backend the first request loads
    import moto.s3.urls
    import moto.server
except ImportError:
    moto = None


@contextlib.contextmanager
def moto_s3_endpoint():
    """Start a moto server, answer one request, and yield its endpoint URL."""
    pytest.importorskip("s3fs")
    if moto is None:
        pytest.skip("moto is not installed")
    import boto3

    server = moto.server.ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        endpoint = f"http://127.0.0.1:{server.get_host_and_port()[1]}"
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        ).list_buckets()
        yield endpoint
    finally:
        server.stop()
