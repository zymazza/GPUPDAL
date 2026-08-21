#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

function platformKey() {
  return `${process.platform}-${process.arch}`;
}

function packagedBinary() {
  return path.join(__dirname, "..", "native", platformKey(), "gpupal");
}

function run(args, options = {}) {
  const environment = options.environment || process.env;
  const exists = options.exists || fs.existsSync;
  const spawn = options.spawn || spawnSync;
  const report = options.report || console.error;
  const forwardSignal = options.forwardSignal ||
    ((signal) => process.kill(process.pid, signal));
  const binary = environment.GPUPAL_BINARY || packagedBinary();

  if (!exists(binary)) {
    report(
      `gpupal: native binary is unavailable for ${platformKey()}. ` +
        "Reinstall from a supported release or set GPUPAL_BINARY to a tested " +
        "GPUPAL executable."
    );
    return 127;
  }

  const result = spawn(binary, args, {
    env: environment,
    stdio: "inherit"
  });
  if (result.error) {
    report(`gpupal: unable to execute ${binary}: ${result.error.message}`);
    return 126;
  }
  if (result.signal) {
    forwardSignal(result.signal);
    return 1;
  }
  return result.status === null ? 1 : result.status;
}

if (require.main === module) {
  process.exit(run(process.argv.slice(2)));
}

module.exports = { packagedBinary, platformKey, run };
