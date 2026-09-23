"""A kernel that cannot bind its ports is replaced, not waited on.

Parallel workers boot and restart kernels at the same time. A restart
relaunches the kernel on the ports it had, and a kernel booting elsewhere
can take one of them in between: the new kernel dies with ``ZMQError:
Address in use``. The harness then waited on it -- 30 s in the restart, and
120 s in every later test's reset of the warm kernel -- so one lost race
failed the test and all its reruns (seen on a 24-worker run). Here the race
is forced: the port is taken while the kernel is down.
"""

from __future__ import annotations

import os
import signal
import socket
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]


def _kill_and_wait(km) -> None:
    process = km.provisioner.process  # the Popen; sync on either manager
    os.kill(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process.poll() is not None, "the kernel did not die"


def _hold(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # The dead kernel's connections leave the port in TIME_WAIT.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen()
    return sock


def test_a_restart_whose_port_was_taken_gets_a_working_kernel(nb_runner):
    nb_runner.create_notebook(["x = 41", "print('X', x + 1)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    km = nb_runner.client.km
    _kill_and_wait(km)
    held = _hold(km.shell_port)
    try:
        nb_runner.restart()
        nb_runner.run_all()
    finally:
        held.close()
    assert "X 42" in nb_runner.get_output(2), nb_runner.get_raw_output(2)


def test_a_kernel_that_died_since_the_last_test_is_replaced(nb_runner):
    """The next test on the worker starts on the kernel the last one left.
    Dead, it answered nothing, and the reset before the test waited out its
    120 s timeout -- in the test and in both reruns."""
    nb_runner.create_notebook(["print('Y', 6 * 7)"])
    nb_runner.start_kernel()
    _kill_and_wait(nb_runner.client.km)
    nb_runner.start_kernel()  # what the next test's start does
    nb_runner.run_all()
    assert "Y 42" in nb_runner.get_output(1), nb_runner.get_raw_output(1)
