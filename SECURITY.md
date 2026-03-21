# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Cash, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Send a description of the vulnerability to the maintainers via a
   [GitHub Security Advisory](https://github.com/cash-caching/cash/security/advisories/new).
3. Include:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix release**: Depends on severity, typically within 2 weeks for critical issues

## Scope

Security issues we care about:
- **Cache poisoning**: Malicious data injection into cache
- **Arbitrary code execution**: Via pickle deserialization of untrusted cache data
- **Information disclosure**: Sensitive data leaking through cache files
- **Denial of service**: Resource exhaustion through cache operations

## Known Considerations

### Pickle Deserialization Trust Boundary

Cash uses Python's `pickle` for serialization by default.  **Pickle can execute
arbitrary code during deserialization**, so cache files must be treated as
trusted data.

- **FileBackend** stores `.meta` (pickled metadata dict) and `.data` (pickled
  values) files in the cache directory (default `~/.cache/cash/`).
  Both are deserialized with `pickle.load()` on cache reads.
- **RedisBackend** stores pickled blobs in Redis keys.
- **InMemoryBackend** keeps references in process memory (no serialization).

**Mitigations:**
- Ensure appropriate file permissions on cache directories (`chmod 700`).
- Do not share cache directories across trust boundaries.
- Consider `CloudPickleSerializer` for environments that need broader type
  support, but note it carries the same trust requirements.

### Network Backends

- Redis and S3 backends transmit data over the network. Use TLS/encryption where
  appropriate.
- File-based caches store data on the local filesystem. Ensure appropriate file
  permissions on cache directories.
