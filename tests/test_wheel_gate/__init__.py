"""Makes this directory a package so its ``conftest`` is not a top-level module.

``tests/test_notebook_integration`` imports shared helpers by bare name --
``from conftest import shows_cached`` -- which only works while its directory
is the one contributing a top-level ``conftest`` module. This directory was
also unpackaged, so collecting both in one run gave two modules claiming the
name ``conftest``, the first import won, and twelve integration modules failed
to collect with::

    ImportError: cannot import name 'shows_cached' from 'conftest'
    (.../tests/test_wheel_gate/conftest.py)

Only visible when both directories are collected together, which is what
``testpaths = ["tests"]`` does -- so the configured default invocation was the
one that broke. Packaging this directory gives its conftest the qualified name
``tests.test_wheel_gate.conftest`` and leaves the bare name to the directory
that relies on it.
"""
