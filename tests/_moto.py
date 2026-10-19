"""An in-process S3 emulator, warmed before any test's clock runs.

moto loads the S3 backend (and IAM, events, cloudwatch with it) on the
server's first request. On a cold, loaded machine that import alone took most
of the 30 s test timeout, inside the first test's fixture. Importing it here,
at collection, and making one request when the server starts keeps that out of
what a test is timed on.

The server is built on the calling thread, not by moto's ``ThreadedMotoServer``:
that one builds it on its own thread and waits, with no timeout, for an event
set only once the build succeeds, so a failure there hung the test for its 30 s
and the only stack left showed the wait. ``http.server`` also asks
``socket.getfqdn`` for the bound address's name, a reverse lookup that can
stall on macOS; the server needs no name, so it binds without one.
"""

from __future__ import annotations

import contextlib
import socketserver
import threading

import pytest

try:
    import moto.s3.responses  # the backend the first request loads
    import moto.s3.urls
    import moto.server
except ImportError:
    moto = None


def _start_server():
    """A threaded moto WSGI server on a free local port, serving on a daemon thread."""
    from moto.moto_server.werkzeug_app import DomainDispatcherApplication, create_backend_app
    from werkzeug.serving import ThreadedWSGIServer

    class Server(ThreadedWSGIServer):
        def server_bind(self):
            socketserver.TCPServer.server_bind(self)  # no getfqdn: see the module docstring
            self.server_name, self.server_port = self.server_address[:2]

    server = Server("127.0.0.1", 0, DomainDispatcherApplication(create_backend_app))
    threading.Thread(target=server.serve_forever, name="moto-s3", daemon=True).start()
    return server


@contextlib.contextmanager
def moto_s3_endpoint():
    """Start a moto server, answer one request, and yield its endpoint URL."""
    pytest.importorskip("s3fs")
    if moto is None:
        pytest.skip("moto is not installed")
    import boto3

    server = _start_server()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        ).list_buckets()
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
