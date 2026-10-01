"""Tests that pin how the notebook code does its work, not what a user sees.

Each test here calls or patches a private part of the notebook package (a
memo, a private method, the boundary between two internal objects). A
refactor of the internals is expected to break them; rewrite them with the
new internals. A test of behaviour a user can observe belongs in the
feature folders instead.
"""
