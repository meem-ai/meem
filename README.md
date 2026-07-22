# meem

meem is a human-like memory system for AI agents, implemented as an OpenCode plugin.

It lets your agent remember naturally while you work. There is no memory list to organize, no manual promotion between categories, and no routine cleanup. The agent records small things as they come up, recalls them when relevant, and gradually keeps the memories that continue to matter.

Like human memory, not everything is remembered equally:

- Short-term memory holds useful but potentially temporary context.
- Long-term memory keeps knowledge that has helped repeatedly.
- Lifetime memory preserves the most durable facts, preferences, and lessons.

A memory becomes more established by being useful. Deliberately finding a memory strengthens it more than having it surface automatically, so frequently relevant knowledge naturally lasts without requiring you to manage it.

## Install

Add `meem` to `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["meem-ai"]
}
```

Restart OpenCode. That is the complete setup.

No other configuration is required. meem's defaults are designed to work out of the box and are usually the best place to start.

On its first memory operation, meem downloads and caches the 47M-parameter Granite Embedding Small R2 model through Transformers.js. It runs locally with an 8k context window and does not require an account or API key. If Hugging Face is temporarily unavailable, the memory tool explains the problem and can be retried later.

The default local model supports English only. For memories in other languages, configure a multilingual embedding model or service under Optional Configuration.

## How It Works

During normal work, the agent can save concise facts, preferences, decisions, corrections, and lessons. It can later remember in two ways:

- Relevant, established memories quietly return when they are likely to help.
- The agent can actively search when it needs to remember something specific.

Exceptionally relevant short-term memories may also return automatically. A memory appears at most once per conversation unless it has already been compacted out, keeping recall useful without repeatedly filling the context.

The model can interact with memories by using the tools `meem_remember` and `meem_search`.

## Managing Memories

`npx -y meem-ai inspect` provides an interactive view of stored memories, including their tiers and retention state. It can move or delete individual memories.

For scripts and agents, use `meem list` for a compact table or add `--json` for structured output. It defaults to newest first and accepts `--limit`, `--sort createdAt:asc`, `--filter text`, and `--tier short|long|lifetime`. Use `meem view <id>` for one complete memory, and `meem promote <id>`, `meem demote <id>`, or `meem delete <id>` for individual changes. Deletion asks for confirmation unless passed `--yes`.

`npx -y meem-ai clear` deletes every memory and automatic insertion record after confirmation. Add `--yes` to skip confirmation.

## Defaults

All configuration is optional. With only `"plugin": ["meem-ai"]`, meem uses:

- Local English embeddings with `onnx-community/granite-embedding-small-english-r2-ONNX`
- No API key or remote service
- Persistent storage at `~/.config/meem/memory.lancedb`
- Up to four memories per automatic recall
- Short-term memories expiring after 30 days without being useful
- Long-term memories expiring after 365 days without being useful
- Lifetime memories never expiring
- A 1536-token embedding chunk size, reduced only if a configured model has a smaller context window

Use these defaults unless you have a specific reason to change them.

## Optional Configuration

Settings can be added to the plugin entry in `opencode.json`. Every setting shown below is optional:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    [
      "meem-ai",
      {
        "storagePath": "/path/to/memory.lancedb",
        "autoRecallLimit": 4,
        "autoPreviousUserMessageLimit": 5,
        "searchRecallLimit": 8,
        "shortTermRetentionDays": 30,
        "longTermRetentionDays": 365,
        "embedding": {
          "baseUrl": "http://localhost:11434/v1",
          "apiKeyEnv": "LOCAL_MODEL_API_KEY",
          "model": "embedding-model-name",
          "contextSize": 8192
        }
      }
    ]
  ]
}
```

- `storagePath` changes where memories are stored.
- `autoRecallLimit` changes how many memories can be inserted automatically.
- `autoPreviousUserMessageLimit` changes how many user messages before the latest are included in automatic recall queries.
- `searchRecallLimit` changes the default and maximum number of memories returned by active search.
- `shortTermRetentionDays` changes how long an unused short-term memory remains available.
- `longTermRetentionDays` changes how long an unused long-term memory remains available.
- `embedding.baseUrl` switches from the bundled local model to an OpenAI-compatible embeddings endpoint.
- `embedding.model` selects the model exposed by that endpoint. It is required only when `baseUrl` is set.
- `embedding.apiKeyEnv` names the environment variable containing the endpoint's API key. It can be omitted for endpoints that do not require authentication.
- `embedding.contextSize` describes the selected model's context window. meem uses it only to reduce chunk size when necessary.

Setting `baseUrl` never silently sends data to OpenAI. meem only calls the exact OpenAI-compatible endpoint you configure.

The same optional settings can instead be stored in `~/.config/meem/config.json`.

### Optional Memory Tuning

Recall gates and tier promotion are configurable too. The defaults are balanced for normal use and usually should not be changed:

```json
{
  "searchSimilarityThreshold": 0.42,
  "autoLifetimeSimilarityThreshold": 0.6,
  "autoLongSimilarityThreshold": 0.7,
  "autoShortSimilarityThreshold": 0.8,
  "deduplicationSimilarityThreshold": 0.965,
  "shortTermPromotionScore": 3,
  "longTermPromotionScore": 8,
  "automaticUseWeight": 1,
  "searchUseWeight": 2
}
```

Similarity thresholds range from `0` to `1`. Lower recall thresholds surface more memories. Promotion scores determine when memories move up a tier, while the use weights make active search reinforce a memory more strongly than automatic recall.

## Optional Environment Variables

Environment variables are also optional and override file or OpenCode settings:

- `MEEM_STORAGE_PATH`
- `MEEM_AUTO_RECALL_LIMIT`
- `MEEM_AUTO_PREVIOUS_USER_MESSAGE_LIMIT`
- `MEEM_SEARCH_RECALL_LIMIT`
- `MEEM_SHORT_TERM_RETENTION_DAYS`
- `MEEM_LONG_TERM_RETENTION_DAYS`
- `MEEM_SEARCH_SIMILARITY_THRESHOLD`
- `MEEM_AUTO_LIFETIME_SIMILARITY_THRESHOLD`
- `MEEM_AUTO_LONG_SIMILARITY_THRESHOLD`
- `MEEM_AUTO_SHORT_SIMILARITY_THRESHOLD`
- `MEEM_DEDUPLICATION_SIMILARITY_THRESHOLD`
- `MEEM_SHORT_TERM_PROMOTION_SCORE`
- `MEEM_LONG_TERM_PROMOTION_SCORE`
- `MEEM_AUTOMATIC_USE_WEIGHT`
- `MEEM_SEARCH_USE_WEIGHT`
- `MEEM_EMBEDDING_URL`
- `MEEM_EMBEDDING_API_KEY`
- `MEEM_EMBEDDING_API_KEY_ENV`
- `MEEM_EMBEDDING_MODEL`
- `MEEM_EMBEDDING_CONTEXT_SIZE`

## References

[OpenCode plugins](https://opencode.ai/docs/plugins) · [OpenCode config](https://opencode.ai/docs/config/#plugins) · [Transformers.js](https://huggingface.co/docs/transformers.js) · [Granite Embedding](https://huggingface.co/onnx-community/granite-embedding-small-english-r2-ONNX)

## License

Copyright (c) 2026 Linus Schlumberger

MIT License, see [LICENSE.md](LICENSE.md).
