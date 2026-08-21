"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const { run } = require("../bin/gpupdal.js");
const {
  platformKey: installerPlatformKey,
  sha256,
  validateEntry
} = require("../scripts/install.js");

test("launcher forwards arguments and exit status", () => {
  let invocation;
  const status = run(["pipeline", "input.json"], {
    environment: { GPUPDAL_BINARY: "/tmp/native-gpupdal" },
    exists: () => true,
    spawn: (binary, args, options) => {
      invocation = { binary, args, options };
      return { error: null, signal: null, status: 7 };
    }
  });
  assert.equal(status, 7);
  assert.equal(invocation.binary, "/tmp/native-gpupdal");
  assert.deepEqual(invocation.args, ["pipeline", "input.json"]);
  assert.equal(invocation.options.stdio, "inherit");
});

test("launcher fails clearly when no native binary is installed", () => {
  const messages = [];
  const status = run(["version"], {
    environment: { GPUPDAL_BINARY: "/definitely/missing/gpupdal" },
    exists: () => false,
    report: (message) => messages.push(message)
  });
  assert.equal(status, 127);
  assert.match(messages.join("\n"), /native binary is unavailable/);
});

test("installer accepts only immutable GPUPDAL release entries", () => {
  const digest = "a".repeat(64);
  assert.doesNotThrow(() => validateEntry({
    url: "https://github.com/zymazza/GPUPDAL/releases/download/v0.1.0/" +
      "gpupdal-linux-x64.tar.gz",
    sha256: digest
  }));
  assert.throws(() => validateEntry({
    url: "https://example.com/gpupdal.tar.gz",
    sha256: digest
  }), /incomplete or untrusted/);
  assert.throws(() => validateEntry({
    url: "https://github.com/zymazza/GPUPDAL/releases/download/latest/" +
      "gpupdal-linux-x64.tar.gz",
    sha256: "not-a-digest"
  }), /incomplete or untrusted/);
});

test("installer platform selection and SHA-256 are deterministic", () => {
  assert.equal(installerPlatformKey("linux", "x64"), "linux-x64");
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "gpupdal-npm-test-"));
  const fixture = path.join(directory, "fixture");
  try {
    fs.writeFileSync(fixture, "gpupdal\n");
    assert.equal(
      sha256(fixture),
      "10466f0b0d1c65b3e41609eb332b5119dd5e47967a20fbd3f55238f3b5dadedf"
    );
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
