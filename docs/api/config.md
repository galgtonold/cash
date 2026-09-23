# Configuration

For both paths: the configuration objects. Every setting, its environment
variable, its default and the path it affects are listed once, in the
[Configuration guide](../getting-started/configuration.md).

```python
from cash import CashConfig, get_config, create_default_config
```

::: cash.CashConfig
    options:
      members: false

---

::: cash.config.TierConfig
    options:
      show_if_no_docstring: true
      members: true
      filters: ["!^_"]

---

::: cash.get_config
    options:
      show_signature: false

::: cash.create_default_config
