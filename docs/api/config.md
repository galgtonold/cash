# Configuration

The dataclass and helpers that drive configuration. For the *how*
(precedence rules, TOML / env / kwargs layering, simple-mode vs
tier-stack mode, runtime mutation), see
[Configuration](../getting-started/configuration.md).

## Imports

```python
from cash import CashConfig, get_config, create_default_config
```

## Fields at a glance

Every field below is also settable via:

* `CASH_<UPPERCASE>` env var (e.g. `CASH_CACHE_DIR`, `CASH_DEBUG=1`)
* `[tool.cash]` table in `pyproject.toml`
* `~/.config/cash/config.toml` (or `%APPDATA%/cash/config.toml` on Windows)
* `Cash(field_name=value, ...)` kwargs
* `cash.configure(field_name=value)` at runtime

Last-write-wins by precedence: kwargs > env vars > project TOML > user
TOML > defaults.

::: cash.CashConfig
    options:
      show_if_no_docstring: true
      members: true

---

::: cash.config.TierConfig
    options:
      show_if_no_docstring: true
      members: true

---

::: cash.get_config

::: cash.create_default_config
