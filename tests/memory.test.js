import assert from "node:assert/strict"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"

import {
  DEFAULT_LONG_TERM_RETENTION_DAYS,
  DEFAULT_SHORT_TERM_RETENTION_DAYS,
  LanceMemoryStore,
  LONG_TERM_PROMOTION_SCORE,
  MemoryEngine,
  TOOL_WRITE_DESCRIPTION,
  automaticMemoryMessage,
  compactionSummaryText,
  isAutomaticMemoryAnchor,
  memoryIdFromContent,
  resolveConfig,
  SEARCH_USE_WEIGHT,
} from "../dist/index.js"

class FakeEmbedder {
  async embed(text) {
    if (text.includes("lifetime")) {
      return [0, 0, 1]
    }
    if (text.includes("different")) {
      return [0, 1, 0]
    }
    return [1, 0, 0]
  }
}

const withEngine = async (run) => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const path = join(directory, "memory.lancedb")
  const engine = new MemoryEngine(new LanceMemoryStore(path), new FakeEmbedder())
  try {
    await run(engine)
  } finally {
    await engine.close()
    await rm(directory, { recursive: true, force: true })
  }
}

test("stores and deduplicates semantic memories", async () => {
  await withEngine(async (engine) => {
    const first = await engine.remember({ content: "User prefers concise replies" })
    const duplicate = await engine.remember({ content: "User likes short answers" })

    assert.equal(first.created, true)
    assert.equal(first.status, "created")
    assert.equal(duplicate.created, false)
    assert.equal(duplicate.status, "duplicate")
    assert.equal(duplicate.memory.id, first.memory.id)
    assert.equal((await engine.recall("concise replies", "search", 5)).length, 1)
  })
})

test("default retention is one day for short-term and seven days for long-term", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const previousHome = process.env.HOME

  try {
    process.env.HOME = directory
    const config = await resolveConfig({ storagePath: join(directory, "memory.lancedb") })

    assert.equal(DEFAULT_SHORT_TERM_RETENTION_DAYS, 1)
    assert.equal(DEFAULT_LONG_TERM_RETENTION_DAYS, 7)
    assert.equal(config.shortTermRetentionDays, 1)
    assert.equal(config.longTermRetentionDays, 7)
  } finally {
    if (previousHome === undefined) {
      delete process.env.HOME
    } else {
      process.env.HOME = previousHome
    }
    await rm(directory, { recursive: true, force: true })
  }
})

test("confirmed semantic duplicate refreshes existing memory", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  let now = Date.UTC(2026, 0, 1)
  const engine = new MemoryEngine(
    new LanceMemoryStore(join(directory, "memory.lancedb")),
    new FakeEmbedder(),
    {},
    () => now,
  )

  try {
    const first = await engine.remember({ content: "User prefers concise replies" })
    now += 1_000
    const duplicate = await engine.remember({ content: "User likes short answers" })
    assert.equal(duplicate.status, "duplicate")
    assert.equal(duplicate.memory.updatedAt, first.memory.updatedAt)

    const confirmed = await engine.remember({ content: "User likes short answers", confirm: true })
    assert.equal(confirmed.status, "confirmed_duplicate")
    assert.equal(confirmed.memory.id, first.memory.id)
    assert.equal(confirmed.memory.updatedAt, new Date(now).toISOString())
    assert.equal((await engine.recall("concise replies", "search", 5)).length, 1)
  } finally {
    await engine.close()
    await rm(directory, { recursive: true, force: true })
  }
})

test("search reinforces and promotes memory more than automatic recall", async () => {
  await withEngine(async (engine) => {
    const remembered = await engine.remember({ content: "The project uses npm" })
    const automatic = await engine.recall("project package manager", "automatic", 1)
    assert.equal(automatic[0]?.memory.tier, "short")

    const searched = await engine.recall("project package manager", "search", 1)
    assert.equal(searched[0]?.memory.tier, "long")
    assert.equal(SEARCH_USE_WEIGHT > 1, true)
    assert.equal(LONG_TERM_PROMOTION_SCORE > SEARCH_USE_WEIGHT, true)
    assert.equal(searched[0]?.memory.searchUses, 1)
  })
})

test("recall gates and excluded IDs prevent repeated exposure", async () => {
  await withEngine(async (engine) => {
    const relevant = await engine.remember({ content: "Relevant fact" })
    await engine.remember({ content: "different topic", tier: "long" })

    const results = await engine.recall("relevant query", "search", 5, new Set([relevant.memory.id]))
    assert.deepEqual(results, [])
  })
})

test("active search can verify memories already shown automatically", async () => {
  await withEngine(async (engine) => {
    const remembered = await engine.remember({ content: "The user likes ice cream." })
    const shown = new Set([remembered.memory.id])

    assert.deepEqual(await engine.recall("ice cream preference", "automatic", 5, shown), [])
    assert.equal((await engine.recall("ice cream preference", "search", 5))[0]?.memory.id, remembered.memory.id)
  })
})

test("expires inactive short and long memories but keeps lifetime memories", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const path = join(directory, "memory.lancedb")
  let now = Date.UTC(2026, 0, 1)
  const engine = new MemoryEngine(
    new LanceMemoryStore(path),
    new FakeEmbedder(),
    { shortTermMilliseconds: 1_000, longTermMilliseconds: 2_000 },
    () => now,
  )

  try {
    await engine.remember({ content: "Temporary detail" })
    await engine.remember({ content: "different durable detail", tier: "long" })
    await engine.remember({ content: "lifetime preference", tier: "lifetime" })

    now += 3_000
    const results = await engine.recall("lifetime query", "search", 5)
    assert.equal(results.length, 1)
    assert.equal(results[0]?.memory.tier, "lifetime")
  } finally {
    await engine.close()
    await rm(directory, { recursive: true, force: true })
  }
})

test("using a memory refreshes its retention period", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  let now = Date.UTC(2026, 0, 1)
  const engine = new MemoryEngine(
    new LanceMemoryStore(join(directory, "memory.lancedb")),
    new FakeEmbedder(),
    { shortTermMilliseconds: 1_000, longTermMilliseconds: 2_000 },
    () => now,
  )

  try {
    await engine.remember({ content: "Useful temporary detail" })
    now += 600
    assert.equal((await engine.recall("useful query", "automatic", 1)).length, 1)
    now += 600
    assert.equal((await engine.recall("useful query", "automatic", 1)).length, 1)
  } finally {
    await engine.close()
    await rm(directory, { recursive: true, force: true })
  }
})

test("automatic recall works for fresh short-term preferences", async () => {
  await withEngine(async (engine) => {
    await engine.remember({ content: "The user likes ice cream." })

    const results = await engine.recall("What dessert would the user enjoy?", "automatic", 1)

    assert.equal(results[0]?.memory.content, "The user likes ice cream.")
  })
})

test("automatic recall applies tier-specific thresholds", async () => {
  await withEngine(async (engine) => {
    await engine.remember({ content: "Short preference" })
    await engine.remember({ content: "different long preference", tier: "long" })
    await engine.remember({ content: "lifetime preference", tier: "lifetime" })

    assert.equal((await engine.recall("Short preference", "automatic", 1)).length, 1)
    assert.equal((await engine.recall("different query", "automatic", 1)).length, 1)
    assert.equal((await engine.recall("lifetime query", "automatic", 1)).length, 1)
  })
})

test("parallel memory writes and searches are safe", async () => {
  await withEngine(async (engine) => {
    const writes = await Promise.all([
      engine.remember({ content: "Parallel memory alpha" }),
      engine.remember({ content: "different parallel memory beta" }),
      engine.remember({ content: "lifetime parallel memory gamma", tier: "lifetime" }),
    ])

    const searches = await Promise.all([
      engine.recall("Parallel memory alpha", "search", 1),
      engine.recall("different parallel memory beta", "search", 1),
      engine.recall("lifetime parallel memory gamma", "search", 1),
    ])

    assert.equal(new Set(writes.map(({ memory }) => memory.id)).size, 3)
    assert.equal(
      searches.every((results) => results.length === 1),
      true,
    )
  })
})

test("automatic insertion records survive engine reload", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const path = join(directory, "memory.lancedb")
  const first = new MemoryEngine(new LanceMemoryStore(path), new FakeEmbedder())

  try {
    const remembered = await first.remember({ content: "The user likes ice cream." })
    await first.rememberAutomaticInsertion("session-1", "message-1", [remembered.memory.id])
    await first.close()

    const second = new MemoryEngine(new LanceMemoryStore(path), new FakeEmbedder())
    const contexts = await second.automaticContexts("session-1", new Set(["message-1"]))

    assert.equal(contexts[0]?.memories[0]?.content, "The user likes ice cream.")
    await second.close()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test("snapshot automatic insertion survives reload after referenced short-term memory expires", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const path = join(directory, "memory.lancedb")
  let now = Date.UTC(2026, 0, 1)
  const policy = { shortTermMilliseconds: 1_000 }
  const first = new MemoryEngine(new LanceMemoryStore(path), new FakeEmbedder(), policy, () => now)

  try {
    const remembered = await first.remember({ content: "The user likes ice cream." })
    const snapshot = `[meem:${remembered.memory.id}] The user liked ice cream at insertion time.`
    await first.rememberAutomaticInsertion("session-1", "message-1", [remembered.memory.id], snapshot)
    await first.close()

    now += 2_000
    const second = new MemoryEngine(new LanceMemoryStore(path), new FakeEmbedder(), policy, () => now)
    const contexts = await second.automaticContexts("session-1", new Set(["message-1"]))

    assert.equal(contexts.length, 1)
    assert.deepEqual(contexts[0]?.insertion.memoryIds, [remembered.memory.id])
    assert.equal(contexts[0]?.insertion.snapshot, snapshot)
    assert.deepEqual(contexts[0]?.memories, [])
    await second.close()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test("snapshot automatic insertion loads when the memory table is absent", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const path = join(directory, "memory.lancedb")
  const first = new MemoryEngine(new LanceMemoryStore(path), new FakeEmbedder())

  try {
    await first.rememberAutomaticInsertion("session-1", "message-1", ["memory-1"], "[meem:memory-1] Archived fact")
    await first.close()

    const second = new MemoryEngine(new LanceMemoryStore(path), new FakeEmbedder())
    const contexts = await second.automaticContexts("session-1", new Set(["message-1"]))

    assert.equal(contexts.length, 1)
    assert.deepEqual(contexts[0]?.insertion.memoryIds, ["memory-1"])
    assert.equal(contexts[0]?.insertion.snapshot, "[meem:memory-1] Archived fact")
    assert.deepEqual(contexts[0]?.memories, [])
    await second.close()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test("loads memories by ordered IDs and skips missing IDs", async () => {
  await withEngine(async (engine) => {
    const first = await engine.remember({ content: "First ordered memory" })
    const second = await engine.remember({ content: "different second ordered memory" })

    const memories = await engine.memoriesByIds([second.memory.id, "missing", first.memory.id, second.memory.id])

    assert.deepEqual(
      memories.map((memory) => memory.id),
      [second.memory.id, first.memory.id, second.memory.id],
    )
  })
})

test("loads no ordered memories when the table is missing", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const engine = new MemoryEngine(new LanceMemoryStore(join(directory, "memory.lancedb")), new FakeEmbedder())

  try {
    assert.deepEqual(await engine.memoriesByIds(["missing"]), [])
  } finally {
    await engine.close()
    await rm(directory, { recursive: true, force: true })
  }
})

test("extracts text from assistant compaction summary messages", () => {
  const text = compactionSummaryText({
    info: { role: "assistant", summary: true },
    parts: [
      { type: "text", text: "Keep this" },
      { type: "text", text: "Ignore this", synthetic: true },
      { type: "text", text: "and this" },
    ],
  })

  assert.equal(text, "Keep this\nand this")
  assert.equal(compactionSummaryText({ info: { role: "assistant" }, parts: [{ type: "text", text: "No" }] }), undefined)
})

test("automatic memory anchors support assistant summaries", () => {
  const anchor = {
    info: { id: "summary-1", sessionID: "session-1", role: "assistant", summary: true, time: { created: 1 } },
    parts: [],
  }
  const memoryIds = ["memory-1", "memory-2"]
  const message = automaticMemoryMessage(anchor, memoryIdFromContent(memoryIds.join("\n")), "[meem:memory-1] Fact")

  assert.equal(isAutomaticMemoryAnchor(anchor), true)
  assert.equal(message.info.role, "assistant")
  assert.equal(message.info.summary, false)
  assert.equal(message.info.id.startsWith("msg_meem_memory_summary-1_"), true)
  assert.equal(message.parts[0]?.messageID, message.info.id)
})

test("remember tool guidance asks for aggressive tiny memories without secrets or raw logs", () => {
  assert.match(TOOL_WRITE_DESCRIPTION, /Save memories aggressively/)
  assert.match(TOOL_WRITE_DESCRIPTION, /almost-throwaway detail/)
  assert.match(TOOL_WRITE_DESCRIPTION, /Do not store secrets/)
  assert.match(TOOL_WRITE_DESCRIPTION, /raw logs/)
})

test("store clear removes memories and automatic insertion records", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const path = join(directory, "memory.lancedb")
  const store = new LanceMemoryStore(path)
  const engine = new MemoryEngine(store, new FakeEmbedder())

  try {
    const remembered = await engine.remember({ content: "The user likes ice cream." })
    await engine.rememberAutomaticInsertion("session-1", "message-1", [remembered.memory.id])

    await store.clear()

    assert.equal(await engine.hasMemories(), false)
    assert.deepEqual(await engine.automaticContexts("session-1", new Set(["message-1"])), [])
  } finally {
    await engine.close()
    await rm(directory, { recursive: true, force: true })
  }
})

test("custom recall and promotion policy changes memory behavior", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-test-"))
  const engine = new MemoryEngine(new LanceMemoryStore(join(directory, "memory.lancedb")), new FakeEmbedder(), {
    searchSimilarityThreshold: 0,
    shortTermPromotionScore: 5,
    longTermPromotionScore: 20,
    searchUseWeight: 5,
  })

  try {
    const remembered = await engine.remember({ content: "Temporary detail" })
    const results = await engine.recall("different query", "search", 1)

    assert.equal(results[0]?.memory.id, remembered.memory.id)
    assert.equal(results[0]?.memory.tier, "long")
  } finally {
    await engine.close()
    await rm(directory, { recursive: true, force: true })
  }
})
