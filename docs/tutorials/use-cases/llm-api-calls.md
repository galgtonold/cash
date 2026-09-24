# LLM API calls

!!! info "Applies to: decorator"
    Code that calls a hosted or local model and re-runs the same prompts while
    you work on everything around them.

Iterating on prompts means sending the same input again and again. With the call
cached, a repeat costs nothing and returns at once. This works with any client:
OpenAI, Anthropic, a local server.

## The pattern

```python
import anthropic
import cash

client = anthropic.Anthropic()
cash.register_hasher(anthropic.Anthropic, lambda c: "anthropic")   # see below

@cash.cache
def chat(prompt: str, model: str = "claude-sonnet-4-6"):
    return client.messages.create(  # @cash:assume-safe
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text

reply = chat("Explain monads in 3 sentences.")  # first call: hits the API
reply = chat("Explain monads in 3 sentences.")  # cache hit: no request
```

The prompt and the model are arguments, so changing either is a new key and a
new request. Edit the parsing downstream of `chat` and re-run: the request is a
hit and only your parsing runs again.

`async def` works the same way; see [Async functions](../feature-guides/async-caching.md).
Two concurrent awaits of the same new prompt both send a request unless you
construct `Cash(use_locking=True)`.

## What you will see

Without the two marked lines, cash warns twice, and both warnings are worth
understanding once:

- [`KEY-UNHASHABLE-GLOBAL`](../../warnings.md#key-unhashable-global): `chat`
  reads the global `client`, which holds connections and can't be hashed.
  Cash is telling you that swapping the client won't change the key. That is
  fine here, because the arguments decide the answer. Registering a hasher for
  the client's type says so. If you point clients at different endpoints,
  return the endpoint URL instead of a constant.
- [`IMPURE-OBSERVED-EFFECTS`](../../warnings.md#impure-observed-effects), on
  the first call: it opened a network connection. A hit skips the request,
  which is the point, but cash can't know you want that. `# @cash:assume-safe`
  on the line that makes the request accepts it. A `ttl=` does not silence this
  warning. See [Side effects](../../decorator.md#side-effects).

Do the same for any SDK client a cached function reads, such as the OpenAI
client below.

## What to cache

Cache calls where the same input should give the same output:
`temperature=0` prompts, classification, embeddings, retrieval lookups.

Make anything that changes the answer an **argument**, so it is in the key:
`temperature`, `seed`, the system prompt, the model. A sampled call without a
seed varies by design, so either give it a `ttl=` or leave it undecorated.

<!-- claim: cash/decorator/runtime.py:entry_expired @f9bb16b6 -->
```python
@cash.cache(ttl=3600)   # the index behind it is refreshed hourly
def web_search_with_llm(query):
    return rag_pipeline(query)
```

## Embeddings

Embeddings depend only on the text and the model, and retrieval work calls them
constantly:

```python
import openai
import cash

openai_client = openai.OpenAI()

@cash.cache
def embed(text: str, model: str = "text-embedding-3-small"):
    response = openai_client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
```

Embed a corpus once; tuning retrieval-k, similarity or reranking never calls the
API again.

## Pages fetched for context

```python
import httpx
import cash

@cash.cache(ttl=86400)   # refetch once a day
def fetch(url: str) -> str:
    return httpx.get(url, timeout=30).text
```

## Counting what you saved

```python
chat.cache_info()
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5, ...}
```

Hits times your cost per request is what you didn't spend. A sudden drop in
`hit_rate` usually means a prompt template changed.

## Caveats

- **Create the client once, at module level.** Don't pass it as an argument or
  cache its constructor.
- **Streaming.** A cached function that yields the stream stores the chunks and
  replays them on a hit, all at once rather than paced by the model. See
  [Iterators](../feature-guides/iterator-caching.md).
- **Large context.** Every argument is hashed on every call. For a large
  retrieved context, pass a stable id (`doc_id`, `version`) and load the text
  inside the cached function.
- **Sensitive prompts** are stored with the response until evicted. Use
  `Cash(backend=InMemoryBackend())` to keep them off disk; see
  [Choosing a backend](../feature-guides/choosing-a-backend.md).

## Related

- [Async functions](../feature-guides/async-caching.md)
- [The `@cash.cache` guide](../../decorator.md#ttl)
- [Custom hashers](../feature-guides/custom-hashers.md)
