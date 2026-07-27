# Fenced fixture

An anchor shown as an example inside a fence must be ignored:

```markdown
<!-- claim: cash/core.py:Cash.cache @7a77d1c5 -->
Cash keys a call on the function source plus its arguments.
```

But a real one outside a fence must still be found:

<!-- claim: cash/config.py:CashConfig.compress -->
Entries can be stored compressed.
