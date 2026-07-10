#!/usr/bin/env node
const args = process.argv.slice(2)

console.log("no CLI implemented yet, please configure as opencode plugin")

if (!args.includes("--debug")) {
  process.exit(0)
}

console.log("called with args:", args)
process.exit(0)
