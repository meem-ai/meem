import assert from "node:assert/strict"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { PassThrough, Writable } from "node:stream"
import test from "node:test"

import { runCli } from "../dist/cli.js"
import { LanceMemoryStore, MemoryEngine } from "../dist/index.js"

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

test("unsupported CLI command returns usage error", async () => {
  const stdout = new CaptureStream()
  const stderr = new CaptureStream()
  const code = await runCli(["unknown"], { stdin: new PassThrough(), stdout, stderr })

  assert.equal(code, 1)
  assert.equal(stdout.text(), "")
  assert.match(stderr.text(), /Unsupported command: unknown/)
  assert.match(stderr.text(), /Usage:/)
})
