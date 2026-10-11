"""HTTP servers on 127.0.0.1 for tests, bound without a DNS lookup.

``http.server.HTTPServer.server_bind`` names the server by
``socket.getfqdn(host)``, a reverse lookup of 127.0.0.1. On the macOS CI
runners that lookup sometimes hung for more than the 30 s test timeout
before the server existed. Nothing in the tests reads ``server_name``, so
it is the address itself.
"""

from __future__ import annotations

import http.server
import socketserver


class _NamedByAddress:
    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)  # type: ignore[arg-type]
        host, port = self.server_address[:2]  # type: ignore[attr-defined]
        self.server_name = str(host)
        self.server_port = port


class LocalHTTPServer(_NamedByAddress, http.server.HTTPServer):
    """``http.server.HTTPServer`` without the reverse lookup of its host."""


class LocalThreadingHTTPServer(_NamedByAddress, http.server.ThreadingHTTPServer):
    """``http.server.ThreadingHTTPServer`` without the reverse lookup of its host."""
