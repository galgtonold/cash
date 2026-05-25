# Skip Annotation Fixture

A skipped fence:

<!-- test:skip reason="illustrative output only" -->
```python
print("this should not run")
```

A normal fence:

```python
x = 1
```

A skipped fence with multi-word reason:

<!-- test:skip reason="needs OpenAI API key" -->
```python
import openai
client = openai.OpenAI()
```
