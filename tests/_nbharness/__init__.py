"""The notebook integration test harness.

Drives real Jupyter kernels over real ``.ipynb`` files for the tests in
``tests/test_notebook_integration``; that folder's ``conftest.py`` holds only
the pytest fixtures built on it.

* ``runner``: :class:`NotebookTestRunner`, what the ``nb_runner`` fixture hands
  a test, and its output helpers.
* ``badge``: reading the cash badge in a cell's output.
* ``kernels``: booting, reusing and killing the kernels behind the runner.
* ``trace``: the upstream decision trace behind the ``upstream_trace`` fixture.
* ``replay_harness``, ``replay_corpus``, ``session_harness``,
  ``session_oracle``: scripted sessions checked against a plain run.
"""
