#!/usr/bin/env node
import { realpathSync } from "node:fs"
import { createInterface } from "node:readline/promises"
import type { Readable, Writable } from "node:stream"
import { pathToFileURL } from "node:url"

import {
  DAY_MILLISECONDS,
  LanceMemoryStore,
  type Memory,
  type MemoryTier,
  type ResolvedMeemConfig,
  resolveConfig,
} from "./index.js"

export const CLI_USAGE = `Usage:
  meem clear [--yes]
  meem inspect

Commands:
  clear    Delete all meem memories and automatic insertion records.
  inspect  Interactively view, clear, upgrade, or downgrade memories.

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

type TerminalInput = Readable & { isTTY?: boolean; setRawMode?: (mode: boolean) => void }
type TerminalOutput = Writable & { columns?: number; isTTY?: boolean; rows?: number }

const MEMORY_TIERS: readonly MemoryTier[] = ["short", "long", "lifetime"]
const ESCAPE_SEQUENCES = new Map([
  ["\u001b[A", "up"],
  ["\u001b[B", "down"],
  ["\u001b[C", "right"],
  ["\u001b[D", "left"],
  ["\u001b[3~", "delete"],
])

interface TerminalMouseEvent {
  button: number
  column: number
  pressed: boolean
  row: number
}

interface MouseRegion {
  end: number
  key: string
  row: number
  start: number
}

interface DetailModal {
  memory: Memory
  offset: number
}

interface TerminalLayout {
  legendRegions: MouseRegion[]
  modalCloseRegion?: MouseRegion
  rowStart: number
  visible: Memory[]
  visibleStart: number
}

type TerminalInputEvent = string | TerminalMouseEvent

const nextTier = (tier: MemoryTier): MemoryTier => {
  if (tier === "short") {
    return "long"
  }
  return "lifetime"
}

const previousTier = (tier: MemoryTier): MemoryTier => {
  if (tier === "lifetime") {
    return "long"
  }
  return "short"
}

const clipped = (value: string, width: number): string => {
  if (value.length <= width) {
    return value.padEnd(width)
  }
  if (width < 4) {
    return value.slice(0, width)
  }
  return `${value.slice(0, width - 3)}...`
}

const elapsedDuration = (milliseconds: number): string => {
  const minutes = Math.floor(milliseconds / 60_000)
  if (minutes < 60) {
    return `${minutes}m`
  }
  const hours = Math.floor(minutes / 60)
  if (hours < 24) {
    return `${hours}h`
  }
  return `${Math.floor(hours / 24)}d`
}

const remainingDuration = (milliseconds: number): string => {
  if (milliseconds <= 0) {
    return "expired"
  }
  return elapsedDuration(milliseconds)
}

const createdAt = (timestamp: string, now: number): string => {
  const date = new Date(timestamp)
  const elapsed = now - date.getTime()
  if (elapsed < 60_000) {
    return "now"
  }
  if (elapsed < DAY_MILLISECONDS * 7) {
    return elapsedDuration(elapsed)
  }
  return new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short" }).format(date)
}

const expiration = (memory: Memory, config: ResolvedMeemConfig, now: number): string => {
  const retentionDays = memory.tier === "short" ? config.shortTermRetentionDays : config.longTermRetentionDays
  return remainingDuration(new Date(memory.updatedAt).getTime() + retentionDays * DAY_MILLISECONDS - now)
}

const promotion = (memory: Memory, config: ResolvedMeemConfig): string => {
  const threshold = memory.tier === "short" ? config.shortTermPromotionScore : config.longTermPromotionScore
  const score = memory.automaticUses * config.automaticUseWeight + memory.searchUses * config.searchUseWeight
  return `${Math.min(100, Math.round((score / threshold) * 100))}%`
}

const wrapText = (text: string, width: number): string[] => {
  const words = text.replaceAll(/\s+/g, " ").trim().split(" ")
  const lines: string[] = []
  let line = ""
  for (const word of words) {
    if (line.length > 0 && line.length + word.length + 1 > width) {
      lines.push(line)
      line = word
    } else {
      line = line.length > 0 ? `${line} ${word}` : word
    }
  }
  if (line.length > 0) {
    lines.push(line)
  }
  return lines.length > 0 ? lines : [""]
}

const inspectMemories = async (
  store: LanceMemoryStore,
  config: ResolvedMeemConfig,
  streams: CliStreams,
): Promise<boolean> => {
  const input = streams.stdin as TerminalInput
  const output = streams.stdout as TerminalOutput
  if (!input.isTTY || !input.setRawMode || !output.isTTY) {
    write(streams.stderr, "meem inspect requires an interactive terminal.\n")
    return false
  }

  let memories = await store.listMemories()
  let currentTier: MemoryTier = "short"
  let confirmation = false
  let detail: DetailModal | undefined
  let layout: TerminalLayout = { legendRegions: [], rowStart: 3, visible: [], visibleStart: 0 }
  let message = ""
  let keyBuffer = ""
  let processing = false
  let running = true
  const keyQueue: TerminalInputEvent[] = []
  const selection: Record<MemoryTier, number> = { short: 0, long: 0, lifetime: 0 }
  let complete = (): void => undefined
  const finished = new Promise<void>((resolve) => {
    complete = resolve
  })

  const tierMemories = (): Memory[] =>
    memories
      .filter((memory) => memory.tier === currentTier)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt) || left.id.localeCompare(right.id))

  const selectedMemory = (): Memory | undefined => {
    const rows = tierMemories()
    selection[currentTier] = Math.min(selection[currentTier], Math.max(0, rows.length - 1))
    return rows[selection[currentTier]]
  }

  const render = (): void => {
    const width = Math.max(40, output.columns ?? 80)
    const height = Math.max(6, output.rows ?? 24)
    const narrow = width < 100
    const footerEntries = narrow
      ? [
          [
            ["up move", "up"],
            ["down move", "down"],
            ["left tier", "left"],
            ["right tier", "right"],
            ["enter view", "enter"],
          ],
          [
            ["p promote", "p"],
            ["o demote", "o"],
            ["del remove", "delete"],
            ["q quit", "q"],
          ],
        ]
      : [
          [
            ["up move", "up"],
            ["down move", "down"],
            ["left tier", "left"],
            ["right tier", "right"],
            ["enter view", "enter"],
            ["p promote", "p"],
            ["o demote", "o"],
            ["del remove", "delete"],
            ["q quit", "q"],
          ],
        ]
    const confirmationText = "Delete selected memory? y or del confirm | any other key cancels"
    const footer = confirmation ? [[[confirmationText, ""]]] : footerEntries
    const rowHeight = Math.max(1, height - 2 - footer.length)
    const createdWidth = 7
    const hasRetentionColumns = currentTier !== "lifetime"
    const expiresWidth = 7
    const promotionWidth = 5
    const textWidth = Math.max(12, width - createdWidth - (hasRetentionColumns ? expiresWidth + promotionWidth + 9 : 4))
    const rows = tierMemories()
    const selectedIndex = Math.min(selection[currentTier], Math.max(0, rows.length - 1))
    const start = Math.max(0, Math.min(selectedIndex - Math.floor(rowHeight / 2), rows.length - rowHeight))
    const visible = rows.slice(start, start + rowHeight)
    const now = Date.now()
    const lines = [
      clipped(`meem memories | ${currentTier} term | ${rows.length}${message ? ` | ${message}` : ""}`, width),
      hasRetentionColumns
        ? `  ${clipped("TEXT", textWidth)} ${clipped("EXPIRES", expiresWidth)} ${clipped("CREATED", createdWidth)} ${clipped("PROMO", promotionWidth)}`
        : `  ${clipped("TEXT", textWidth)} ${clipped("CREATED", createdWidth)}`,
    ]
    if (visible.length === 0) {
      lines.push(clipped(`  No ${currentTier}-term memories`, width))
    } else {
      for (const [offset, memory] of visible.entries()) {
        const marker = start + offset === selectedIndex ? ">" : " "
        const content = memory.content.replaceAll(/\s+/g, " ")
        const row = hasRetentionColumns
          ? `${marker} ${clipped(content, textWidth)} ${clipped(expiration(memory, config, now), expiresWidth)} ${clipped(createdAt(memory.createdAt, now), createdWidth)} ${clipped(promotion(memory, config), promotionWidth)}`
          : `${marker} ${clipped(content, textWidth)} ${clipped(createdAt(memory.createdAt, now), createdWidth)}`
        lines.push(row)
      }
    }
    while (lines.length < 2 + rowHeight) {
      lines.push("")
    }
    const legendRegions: MouseRegion[] = []
    for (const entries of footer) {
      const row = lines.length + 1
      let column = 1
      const line = entries
        .map(([label, key]) => {
          const result = label
          if (key) {
            legendRegions.push({ end: column + result.length - 1, key, row, start: column })
          }
          column += result.length + 3
          return result
        })
        .join(" | ")
      lines.push(clipped(line, width))
    }
    layout = { legendRegions, rowStart: 3, visible, visibleStart: start }
    const hasBackdrop = detail !== undefined || confirmation
    let screen = `\u001b[H\u001b[2J${hasBackdrop ? "\u001b[2m" : ""}${lines.join("\n")}${hasBackdrop ? "\u001b[0m" : ""}`
    if (confirmation) {
      screen += `\u001b[${height};1H${clipped(confirmationText, width)}`
    }
    if (detail) {
      const modalWidth = Math.max(24, Math.min(width - 4, 80))
      const innerWidth = modalWidth - 2
      const body = wrapText(detail.memory.content, innerWidth - 2)
      const bodyHeight = Math.max(1, Math.min(body.length, height - 5))
      detail.offset = Math.min(detail.offset, Math.max(0, body.length - bodyHeight))
      const top = Math.max(1, Math.floor((height - bodyHeight - 4) / 2) + 1)
      const left = Math.max(1, Math.floor((width - modalWidth) / 2) + 1)
      const close = "[close]"
      const modalLines = [
        `+${"-".repeat(innerWidth)}+`,
        `|${clipped(detail.memory.id, innerWidth)}|`,
        ...body.slice(detail.offset, detail.offset + bodyHeight).map((line) => `| ${clipped(line, innerWidth - 2)} |`),
        `|${" ".repeat(innerWidth - close.length)}${close}|`,
        `+${"-".repeat(innerWidth)}+`,
      ]
      for (const [offset, line] of modalLines.entries()) {
        screen += `\u001b[${top + offset};${left}H${line}`
      }
      layout.modalCloseRegion = {
        end: left + innerWidth,
        key: "close",
        row: top + bodyHeight + 2,
        start: left + innerWidth - close.length + 1,
      }
    }
    write(output, screen)
  }

  const refreshMemories = async (): Promise<void> => {
    memories = await store.listMemories()
    selectedMemory()
  }

  const moveSelectedMemory = async (tier: MemoryTier): Promise<void> => {
    const memory = selectedMemory()
    if (!memory) {
      return
    }
    if (memory.tier === tier) {
      message = tier === "lifetime" ? "already lifetime" : "already short-term"
      return
    }
    await store.updateMemory({
      ...memory,
      tier,
      automaticUses: 0,
      searchUses: 0,
      updatedAt: new Date().toISOString(),
    })
    message = tier === "lifetime" ? "promoted" : tier === "long" ? "promoted" : "demoted"
    await refreshMemories()
  }

  const finish = (): void => {
    running = false
    complete()
  }

  const handleKey = async (key: string): Promise<void> => {
    if (detail) {
      if (key === "up" || key === "down" || key === "left" || key === "right") {
        detail = undefined
      } else if (key === "enter" || key === "escape" || key === "q" || key === "\u0003") {
        detail = undefined
        render()
        return
      } else {
        render()
        return
      }
    }
    if (confirmation) {
      confirmation = false
      if (key === "y" || key === "delete") {
        const memory = selectedMemory()
        if (memory) {
          await store.deleteMemory(memory.id)
          message = "deleted"
          await refreshMemories()
        }
        render()
        return
      }
      if (key !== "up" && key !== "down" && key !== "left" && key !== "right") {
        render()
        return
      }
    }

    if (key === "q" || key === "\u0003") {
      finish()
      return
    }
    if (key === "up") {
      selection[currentTier] = Math.max(0, selection[currentTier] - 1)
    } else if (key === "down") {
      selection[currentTier] += 1
      selectedMemory()
    } else if (key === "left" || key === "right") {
      const currentIndex = MEMORY_TIERS.indexOf(currentTier)
      const direction = key === "left" ? -1 : 1
      const tier = MEMORY_TIERS[currentIndex + direction]
      if (tier) {
        currentTier = tier
        selectedMemory()
      }
    } else if (key === "delete") {
      if (selectedMemory()) {
        confirmation = true
      }
    } else if (key === "enter") {
      const memory = selectedMemory()
      if (memory) {
        detail = { memory, offset: 0 }
      }
    } else if (key === "p") {
      await moveSelectedMemory(nextTier(currentTier))
    } else if (key === "o") {
      await moveSelectedMemory(previousTier(currentTier))
    }
    render()
  }

  const handleMouse = async (event: TerminalMouseEvent): Promise<void> => {
    if (event.button === 64) {
      await handleKey("up")
      return
    }
    if (event.button === 65) {
      await handleKey("down")
      return
    }
    if (!event.pressed || event.button !== 0) {
      return
    }
    if (detail) {
      const close = layout.modalCloseRegion
      if (close && event.row === close.row && event.column >= close.start && event.column <= close.end) {
        detail = undefined
        render()
      } else {
        detail = undefined
        render()
      }
      return
    }
    if (confirmation) {
      return
    }
    const legend = layout.legendRegions.find(
      (region) => event.row === region.row && event.column >= region.start && event.column <= region.end,
    )
    if (legend) {
      await handleKey(legend.key)
      return
    }
    const rowIndex = event.row - layout.rowStart
    const memory = layout.visible[rowIndex]
    if (!memory) {
      return
    }
    const index = layout.visibleStart + rowIndex
    if (selection[currentTier] === index) {
      detail = { memory, offset: 0 }
    } else {
      selection[currentTier] = index
      message = ""
    }
    render()
  }

  const handleInput = async (event: TerminalInputEvent): Promise<void> => {
    if (typeof event === "string") {
      await handleKey(event)
    } else {
      await handleMouse(event)
    }
  }

  const processKeys = async (): Promise<void> => {
    if (processing) {
      return
    }
    processing = true
    while (running && keyQueue.length > 0) {
      const event = keyQueue.shift()
      if (event) {
        await handleInput(event)
      }
    }
    processing = false
  }

  const queueKeys = (chunk: Buffer | string): void => {
    keyBuffer += chunk.toString()
    while (keyBuffer.length > 0) {
      const match = [...ESCAPE_SEQUENCES.keys()].find((sequence) => keyBuffer.startsWith(sequence))
      if (match) {
        keyQueue.push(ESCAPE_SEQUENCES.get(match) ?? "")
        keyBuffer = keyBuffer.slice(match.length)
        continue
      }
      const mouseMatch = keyBuffer.match(/^\u001b\[<(\d+);(\d+);(\d+)([Mm])/)
      if (mouseMatch) {
        keyQueue.push({
          button: Number(mouseMatch[1]),
          column: Number(mouseMatch[2]),
          pressed: mouseMatch[4] === "M",
          row: Number(mouseMatch[3]),
        })
        keyBuffer = keyBuffer.slice(mouseMatch[0].length)
        continue
      }
      if (keyBuffer.startsWith("\u001b[<")) {
        break
      }
      if (keyBuffer.startsWith("\u001b[M")) {
        if (keyBuffer.length < 6) {
          break
        }
        keyBuffer = keyBuffer.slice(6)
        continue
      }
      if (keyBuffer.startsWith("\u001b[")) {
        const finalIndex = keyBuffer.slice(2).search(/[\x40-\x7e]/)
        if (finalIndex < 0) {
          break
        }
        keyBuffer = keyBuffer.slice(finalIndex + 3)
        continue
      }
      if (keyBuffer.startsWith("\u001bO")) {
        if (keyBuffer.length < 3) {
          break
        }
        keyBuffer = keyBuffer.slice(3)
        continue
      }
      const key = keyBuffer.slice(0, 1)
      keyQueue.push(
        key === "\u007f"
          ? "delete"
          : key === "\r" || key === "\n"
            ? "enter"
            : key === "\u001b"
              ? "escape"
              : key.toLowerCase(),
      )
      keyBuffer = keyBuffer.slice(1)
    }
    void processKeys()
  }

  const resize = (): void => {
    render()
  }

  input.setRawMode(true)
  input.resume()
  input.on("data", queueKeys)
  output.on("resize", resize)
  write(output, "\u001b[?1049h\u001b[?25l\u001b[?1000h\u001b[?1006h")
  render()
  try {
    await finished
    return true
  } finally {
    input.removeListener("data", queueKeys)
    output.removeListener("resize", resize)
    input.setRawMode(false)
    write(output, "\u001b[?1006l\u001b[?1000l\u001b[?25h\u001b[?1049l")
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
  if (command !== "clear" && command !== "inspect") {
    write(streams.stderr, `Unsupported command: ${command ?? ""}\n${CLI_USAGE}\n`)
    return 1
  }

  const unsupportedFlag = flags.find((flag) => command !== "clear" || flag !== "--yes")
  if (unsupportedFlag) {
    write(streams.stderr, `Unsupported flag for ${command}: ${unsupportedFlag}\n${CLI_USAGE}\n`)
    return 1
  }

  if (command === "inspect") {
    const config = await resolveConfig()
    const store = new LanceMemoryStore(config.storagePath)
    try {
      return (await inspectMemories(store, config, streams)) ? 0 : 1
    } finally {
      await store.close()
    }
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

if (process.argv[1] && import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href) {
  const exitCode = await runCli()
  process.exit(exitCode)
}
