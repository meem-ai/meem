#!/usr/bin/env node
import { createInterface } from "node:readline/promises"
import type { Readable, Writable } from "node:stream"
import { fileURLToPath } from "node:url"

import { LanceMemoryStore, resolveConfig } from "./index.js"

export const CLI_USAGE = `Usage:
  meem clear [--yes]

Commands:
  clear    Delete all meem memories and automatic insertion records.

Options:
  --yes    Skip confirmation.
  --help   Show this help.`

export interface CliStreams {
  stdin: Readable
  stdout: Writable
  stderr: Writable
}

const write = (stream: Writable, text: string): void => {
  stream.write(text)
}

const confirmClear = async ({ stdin, stdout }: Pick<CliStreams, "stdin" | "stdout">): Promise<boolean> => {
  const readline = createInterface({ input: stdin, output: stdout })
  try {
    const answer = await readline.question("Clear all meem memories and automatic insertion records? [y/N] ")
    return ["y", "yes"].includes(answer.trim().toLowerCase())
  } finally {
    readline.close()
  }
}

export const runCli = async (
  args: string[] = process.argv.slice(2),
  streams: CliStreams = { stdin: process.stdin, stdout: process.stdout, stderr: process.stderr },
): Promise<number> => {
  if (args.length === 0 || args.includes("--help")) {
    write(streams.stdout, `${CLI_USAGE}\n`)
    return 0
  }

  const [command, ...flags] = args
  if (command !== "clear") {
    write(streams.stderr, `Unsupported command: ${command ?? ""}\n${CLI_USAGE}\n`)
    return 1
  }

  const unsupportedFlag = flags.find((flag) => flag !== "--yes")
  if (unsupportedFlag) {
    write(streams.stderr, `Unsupported flag for clear: ${unsupportedFlag}\n${CLI_USAGE}\n`)
    return 1
  }

  const confirmed = flags.includes("--yes") || (await confirmClear(streams))
  if (!confirmed) {
    write(streams.stdout, "Aborted.\n")
    return 0
  }

  const config = await resolveConfig()
  const store = new LanceMemoryStore(config.storagePath)
  await store.clear()
  await store.close()
  write(streams.stdout, `Cleared meem storage at ${config.storagePath}.\n`)
  return 0
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const exitCode = await runCli()
  process.exit(exitCode)
}
