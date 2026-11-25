"""A remote object replaced while the cell reading it still runs is read again
next time.

Another job published a new version of ``https://.../data.csv`` while a slow
loader was running. The statement was stored with the NEW object's token next
to the value computed from the old one, so every later Run All served the old
data as fresh. As for a local file, it now runs uncached with a
``STORE-INPUT-CHANGED`` warning, and the next Run All reads the settled object.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler

import pytest

from tests._local_http import LocalThreadingHTTPServer

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


class _Origin:
    """One object; replaced as its first download is answered, so that
    download is the last of the old object."""

    def __init__(self) -> None:
        self.etag, self.body = '"v1"', b"x\n1\n"
        self.gets = 0
        self.lock = threading.Lock()

    def answer(self, download: bool) -> tuple[str, bytes]:
        with self.lock:
            found = self.etag, self.body
            if download:
                self.gets += 1
                if self.gets == 1:
                    self.etag, self.body = '"v2"', b"x\n2\n"
            return found


class _Handler(BaseHTTPRequestHandler):
    def _headers(self, download: bool) -> bytes:
        etag, body = self.server.state.answer(download)
        self.send_response(200)
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return body

    def do_HEAD(self):
        self._headers(False)

    def do_GET(self):
        self.wfile.write(self._headers(True))

    def log_message(self, *args):
        pass


@pytest.fixture
def origin():
    server = LocalThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = _Origin()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/data.csv"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_next_run_all_reads_the_settled_object(nb_runner, origin):
    pytest.importorskip("pandas")
    nb_runner.create_notebook(
        [
            "import time\nimport pandas as pd",
            "def load(u):\n    d = pd.read_csv(u)\n    time.sleep(0.5)\n    return d",
            f"df = load({origin!r})",
            "t = int(df.x.sum())",
            "print(t)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip() == "1", "the cell must read the object before it is replaced"
    assert "STORE-INPUT-CHANGED" in nb_runner.get_raw_output(3)

    nb_runner.run_all()
    assert nb_runner.get_output(5).strip() == "2"
