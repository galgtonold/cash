# Security Policy

## Supported versions

Cash is in its `0.1.x` beta series. Security fixes are made against the latest
released `0.1.x` version. Because the cache format may change between minor
versions during `0.x`, always upgrade to the newest release before reporting an
issue.

| Version | Supported |
|---------|-----------|
| `0.1.x` (latest) | ✅ |
| older `0.1.x`    | ⚠️ upgrade first |
| `0.0.x` / TestPyPI | ❌ |

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public
issue for anything exploitable.

Use GitHub's private vulnerability reporting: go to the repository's
[**Security** tab → **Report a vulnerability**](https://github.com/galgtonold/cash/security/advisories/new).
This opens a private advisory visible only to the maintainers.

When reporting, please include:

- the cash version and Python version,
- a minimal reproduction,
- the impact you observed, and
- any suggested remediation if you have one.

We aim to acknowledge a report within a few days and to keep you updated as we
investigate. Once a fix is available we will coordinate a disclosure timeline
with you and credit you in the release notes unless you prefer to remain
anonymous.

## The cache is executable — the most important thing to know

Cash persists results by **pickling** Python objects. Unpickling runs arbitrary
code, so **loading a cache is equivalent to running a Python script from whoever
produced it.** This is inherent to Python's `pickle` and is not a bug in cash.

- Only load a `.cash/` directory, or point cash at a Redis/S3 store, that was
  produced by a source you trust as much as you'd trust running their code.
- Your own local `.cash/` directory is as safe as the code that wrote it.

The full trust model is documented in
[Backends → Security](https://cash-lib.readthedocs.io/en/latest/api/backends/#security).
Reports that amount to "unpickling a cache from an untrusted party runs code"
describe this documented, intended behavior rather than a vulnerability — but if
you find a way for cash to execute untrusted code **without** loading an
untrusted cache, that is a vulnerability and we want to hear about it.
