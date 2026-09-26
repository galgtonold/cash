"""A test that timed out and passed on its retry is reported with the frames
it was stuck in.

The report of retried-away failures kept only the ``E`` lines, and for a
timeout that is ``Failed: Timeout (>30.0s) from pytest-timeout`` -- how long
pytest waited, not where. Two tests timed out on every macOS job that way,
run after run, and the log never said what they were waiting for.
"""

from tests.conftest import _retry_evidence

# What `str(report.longrepr)` holds under CI's --tb=short.
_TIMED_OUT = """\
tests/test_notebook/test_remote_statement_deps.py:79: in test_a_remote_read_contributes_a_component
    component = compute_file_hash_component(set(), {origin.url})
src/cash/remote_source.py:381: in _http_headers
    with opener.open(request, timeout=timeout) as response:
/usr/lib/python3.13/socket.py:720: in readinto
    return self._sock.recv_into(b)
E   Failed: Timeout (>30.0s) from pytest-timeout."""

_FAILED_ASSERTION = """\
tests/test_core/test_ttl.py:206: in test_an_entry_expired_under_the_tier_default_says_so
    assert explanation.reason == "ttl_expired", explanation
E   AssertionError: no_entry
E   assert 'no_entry' == 'ttl_expired'"""


def test_a_timeout_is_reported_with_the_frames_it_was_stuck_in():
    lines = _retry_evidence(_TIMED_OUT)

    assert lines[-1].startswith("E   Failed: Timeout")
    assert "socket.py:720: in readinto  return self._sock.recv_into(b)" in lines[-2]
    assert any("remote_source.py:381: in _http_headers  with opener.open(" in ln for ln in lines)


def test_control_a_failed_assertion_is_reported_by_its_message_alone():
    assert _retry_evidence(_FAILED_ASSERTION) == [
        "E   AssertionError: no_entry",
        "E   assert 'no_entry' == 'ttl_expired'",
    ]
