"""Configuration: the settings, where they are read from, and the template.

- `schema`: `CashConfig`, `TierConfig` and what each value may be.
- `sources`: reading a config file and the ``CASH_*`` variables.
- `resolve`: merging every layer into one `CashConfig` (`get_config`).
- `template`: the documented config file (`create_default_config`).
- `notices`: the once-per-process report of a setting cash cannot use.
"""
