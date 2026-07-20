import assert from "node:assert/strict"
import { execFile } from "node:child_process"
import { mkdtemp, rm, symlink } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { PassThrough, Writable } from "node:stream"
import { promisify } from "node:util"
import test from "node:test"

import { runCli } from "../dist/cli.js"
import { LanceMemoryStore, MemoryEngine } from "../dist/index.js"

const execFileAsync = promisify(execFile)

class FakeEmbedder {
  async embed() {
    return [1, 0, 0]
  }
}

class CaptureStream extends Writable {
  chunks = []

  _write(chunk, _encoding, callback) {
    this.chunks.push(Buffer.from(chunk).toString("utf8"))
    callback()
  }

  text() {
    return this.chunks.join("")
  }
}

class TerminalInput extends PassThrough {
  isTTY = true
  rawMode = false
  #whenRaw
  #resolveRaw

  constructor() {
    super()
    this.#whenRaw = new Promise((resolve) => {
      this.#resolveRaw = resolve
    })
  }

  setRawMode(enabled) {
    this.rawMode = enabled
    if (enabled) {
      this.#resolveRaw()
    }
  }

  async whenRaw() {
    await this.#whenRaw
  }
}

class TerminalStream extends CaptureStream {
  isTTY = true
  columns = 100
  rows = 12
}

const withStoragePath = async (storagePath, run) => {
  const previous = process.env.MEEM_STORAGE_PATH
  process.env.MEEM_STORAGE_PATH = storagePath
  try {
    return await run()
  } finally {
    if (previous === undefined) {
      delete process.env.MEEM_STORAGE_PATH
    } else {
      process.env.MEEM_STORAGE_PATH = previous
    }
  }
}

test("clear --yes clears configured storage path", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-cli-test-"))
  const storagePath = join(directory, "memory.lancedb")
  const engine = new MemoryEngine(new LanceMemoryStore(storagePath), new FakeEmbedder())

  try {
    await engine.remember({ content: "The user likes ice cream." })
    await engine.close()

    await withStoragePath(storagePath, async () => {
      const stdout = new CaptureStream()
      const stderr = new CaptureStream()
      const code = await runCli(["clear", "--yes"], {
        stdin: new PassThrough(),
        stdout,
        stderr,
      })

      assert.equal(code, 0)
      assert.match(stdout.text(), /Cleared meem storage/)
      assert.equal(stderr.text(), "")
    })

    const reopened = new MemoryEngine(new LanceMemoryStore(storagePath), new FakeEmbedder())
    assert.equal(await reopened.hasMemories(), false)
    await reopened.close()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test("clear asks for confirmation unless --yes is provided", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-cli-test-"))
  const storagePath = join(directory, "memory.lancedb")
  const input = new PassThrough()
  const stdout = new CaptureStream()
  const stderr = new CaptureStream()

  try {
    const run = withStoragePath(storagePath, async () => {
      const pending = runCli(["clear"], { stdin: input, stdout, stderr })
      input.end("n\n")
      return pending
    })

    const code = await run
    assert.equal(code, 0)
    assert.match(stdout.text(), /Clear all meem memories/)
    assert.match(stdout.text(), /Aborted\./)
    assert.equal(stderr.text(), "")
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test("package binary runs through its symlink", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-cli-test-"))
  const binary = join(directory, "meem")

  try {
    await symlink(join(process.cwd(), "dist", "cli.js"), binary)
    const { stdout, stderr } = await execFileAsync(process.execPath, [binary, "--help"])

    assert.match(stdout, /Usage:/)
    assert.equal(stderr, "")
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test("inspect renders a keyboard table and changes selected memories", async () => {
  const directory = await mkdtemp(join(tmpdir(), "meem-cli-test-"))
  const storagePath = join(directory, "memory.lancedb")
  const store = new LanceMemoryStore(storagePath)

  try {
    const now = Date.now()
    await store.addMemory({
      id: "promote",
      content: "Promote this memory",
      embedding: [1, 0, 0],
      tier: "short",
      automaticUses: 2,
      searchUses: 1,
      createdAt: new Date(now).toISOString(),
      updatedAt: new Date(now).toISOString(),
    })
    await store.addMemory({
      id: "downgrade",
      content: "Keep this memory",
      embedding: [0, 1, 0],
      tier: "long",
      automaticUses: 1,
      searchUses: 3,
      createdAt: new Date(now - 1_000).toISOString(),
      updatedAt: new Date(now - 1_000).toISOString(),
    })
    await store.addMemory({
      id: "clear",
      content: "Clear this memory",
      embedding: [0, 0, 1],
      tier: "short",
      automaticUses: 0,
      searchUses: 0,
      createdAt: new Date(now - 2_000).toISOString(),
      updatedAt: new Date(now - 2_000).toISOString(),
    })
    await store.close()

    await withStoragePath(storagePath, async () => {
      const input = new TerminalInput()
      const stdout = new TerminalStream()
      const stderr = new CaptureStream()
      const pending = runCli(["inspect"], { stdin: input, stdout, stderr })
      await input.whenRaw()
      input.write(
        "\u001b[<0;3;3M\u001b[C\u001b[D\u001b[<0;3;3M\u001b[<0;85;7M\u001b[<64;3;3M\u001b[<0;62;12M\u007f\u001b[C\u001b[D\u007f\u007f\u001b[C\u001b[Bo\u001b[M o!q",
      )

      assert.equal(await pending, 0)
      input.end()
      assert.match(stdout.text(), /meem memories \| short term/)
      assert.match(stdout.text(), /TEXT.*EXPIRES.*CREATED.*PROMO/)
      assert.match(stdout.text(), /Delete selected memory\? y or del confirm/)
      assert.match(stdout.text(), /\[close\]/)
      assert.match(stdout.text(), /up move \| down move/)
      assert.equal(stderr.text(), "")
    })

    const reopened = new LanceMemoryStore(storagePath)
    const memories = await reopened.listMemories()
    assert.deepEqual(
      memories.map((memory) => [memory.content, memory.tier]).sort(([first], [second]) => first.localeCompare(second)),
      [
        ["Keep this memory", "short"],
        ["Promote this memory", "long"],
      ],
    )
    assert.deepEqual(
      memories
        .map((memory) => [memory.id, memory.automaticUses, memory.searchUses])
        .sort(([first], [second]) => first.localeCompare(second)),
      [
        ["downgrade", 0, 0],
        ["promote", 0, 0],
      ],
    )
    await reopened.close()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test("unsupported CLI command returns usage error", async () => {
  const stdout = new CaptureStream()
  const stderr = new CaptureStream()
  const code = await runCli(["unknown"], { stdin: new PassThrough(), stdout, stderr })

  assert.equal(code, 1)
  assert.equal(stdout.text(), "")
  assert.match(stderr.text(), /Unsupported command: unknown/)
  assert.match(stderr.text(), /Usage:/)
})
