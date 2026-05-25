# Caching LLM API Calls

Iterating on prompts means calling the same model with the same input dozens of times. Cash caches the response, so the second iteration costs $0 and 0ms instead of $0.05 and 800ms. This applies to any LLM API client (OpenAI, Anthropic, local) and any inference endpoint.

## Why this matters

- **Cost.** Every call has a dollar cost. A single afternoon of prompt iteration on a frontier model can easily run into double-digit dollars if every re-run hits the API.
- **Latency.** Every API call is hundreds of milliseconds minimum, and often seconds for long outputs. Iterating against a network round-trip kills flow state.
- **Determinism.** Same prompt at temperature 0 *should* return the same response. Cache it once, treat it as a pure function for the rest of the session.

## Quick start (sync)

```python
import anthropic
import cash

client = anthropic.Anthropic()

@cash.cache
def chat(prompt: str, model: str = "claude-sonnet-4-6"):
    return client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text

reply = chat("Explain monads in 3 sentences.")  # First call: hits the API
reply = chat("Explain monads in 3 sentences.")  # Second call: instant from cache
```

That's it. The decorator keys on the function arguments, so changing the prompt or the model produces a new key and a new API call. Reusing the same arguments returns the cached response.

## Quick start (async)

Most production LLM code is async. Cash supports `async def` directly:

```python
import anthropic
import cash

client = anthropic.AsyncAnthropic()

@cash.cache
async def chat_async(prompt: str, model: str = "claude-sonnet-4-6"):
    msg = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
```

See [Async Caching](../feature-guides/async-caching.md) for concurrency semantics — Cash coalesces concurrent calls to the same key into a single in-flight request.

## TTL for non-deterministic prompts

If your function pulls from a moving source (web search, retrieval over a refreshed index, a model with non-zero temperature), give the cache a finite shelf life:

<!-- test:skip reason="uses undefined rag_pipeline function; illustrative snippet" -->
```python
@cash.cache(ttl=3600)  # 1 hour
def web_search_with_llm(query):
    return rag_pipeline(query)
```

After an hour, the next call re-runs the pipeline and writes a fresh entry.

## The prompt-iteration hot loop

This is where Cash earns its keep during development:

- **Edit the prompt → re-run.** New key, new API call, response cached.
- **Edit downstream parsing → re-run.** Same prompt → cache hit on the API call. Only the parsing re-runs.
- **Try the same prompt with three different parsers.** One API call total, three parses.

The expensive thing (the API call) happens once per unique prompt. Everything downstream is free to iterate on.

## What to cache vs not

Cache:

- Deterministic prompts at `temperature=0`.
- Retrieval lookups and embedding calls — these are pure functions of their input.
- Classification calls (sentiment, intent, toxicity) — usually deterministic and called the same way thousands of times.
- Anything where the same input is genuinely expected to produce the same output.

Don't cache (or cache carefully):

- Streaming responses where you need token-by-token UX. The cache materializes the stream — fine for batch, awkward for live UI.
- `temperature > 0` calls where you actually want sampling variation. Either use a TTL, or annotate with `@cash:no-cache` per call.
- User-facing chat where freshness matters more than cost.

## Handling non-determinism explicitly

When the model itself is non-deterministic, make that visible in the cache key:

- **Include `temperature` as an argument.** Different temperatures get different keys automatically.
- **Include `seed` as an argument** when the provider supports it. With a fixed seed, sampled outputs become deterministic and cacheable.
- **For non-seeded sampling, use `ttl=`.** Don't pretend the call is pure — give the cache an expiry.

<!-- test:skip reason="redefines chat function overwriting the called version; claim counts would mismatch" -->
```python
@cash.cache
def chat(prompt: str, temperature: float = 0.0, seed: int | None = None):
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text
```

## Embeddings

Embeddings are perfect cache candidates. They're pure functions of `(text, model)` and you call them constantly during retrieval development:

```python
import openai
import cash

openai_client = openai.OpenAI()

@cash.cache
def embed(text: str, model: str = "text-embedding-3-small"):
    response = openai_client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
```

Embedding a thousand documents once, then iterating on similarity logic, retrieval-k, or reranking — none of those iterations re-hit the embedding API.

## Web scraping for context

The same pattern works for any HTTP fetch you feed into an LLM. Cache `fetch(url)`, control freshness with TTL:

```python
import httpx
import cash

@cash.cache(ttl=86400)  # refresh once a day
def fetch(url: str) -> str:
    return httpx.get(url, timeout=30).text
```

Now your RAG pipeline can re-run all afternoon while only re-fetching pages whose entries have expired.

## Cost tracking

Cash tracks hits and misses per function:

```python
chat.cache_info()
# CacheInfo(hits=42, misses=8, ...)
```

Multiply hits by your per-call cost for a quick spend-avoided estimate. For a Sonnet call at ~$0.05/request, 42 hits ≈ $2.10 saved on that function in this session.

## Caveats

- **Don't cache the client object itself.** Initialize the client once at module scope. Caching the constructor adds nothing and complicates serialization.
- **Streaming responses.** Cash materializes streams via [Iterator Caching](../feature-guides/iterator-caching.md). On a cache miss the stream is consumed and stored; on a hit you get the full materialized response back. If you need true streaming UX on hits, do post-processing (e.g. yield chunks of the cached text) downstream of the cached function.
- **Large inputs (RAG with 10K-token context).** The cache key hashes every argument. If you pass a 10K-token context blob on every call, you're hashing it on every call. Either pass a stable id (`doc_id` + `version`) and resolve the context inside the cached function, or supply a custom hasher — see [Custom Hashers](../feature-guides/custom-hashers.md).
- **PII in cache keys.** If your prompts contain sensitive data, that data sits in the cache directory until evicted. For sensitive workflows use `Cash(backend=InMemoryBackend())` so nothing touches disk — see [Choosing a Backend](../feature-guides/choosing-a-backend.md).

## Related

- [Async Caching](../feature-guides/async-caching.md) — concurrency semantics for `async def` cached functions.
- [Controlling Cache Behavior](../feature-guides/controlling-cache-behavior.md) — TTL, `@cash:no-cache`, and per-call opt-outs.
- [Custom Hashers](../feature-guides/custom-hashers.md) — when prompts include large or complex objects.
- [Choosing a Backend](../feature-guides/choosing-a-backend.md) — in-memory vs disk for PII-sensitive workflows.
- [Iterator Caching](../feature-guides/iterator-caching.md) — how Cash handles streaming responses under the hood.
