"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const { run } = require("../bin/gpupal.js");
const {
  platformKey: installerPlatformKey,
  sha256,
  validateEntry
} = require("../scripts/install.js");

test("launcher forwards arguments and exit status", () => {
  let invocation;
  const status = run(["pipeline", "input.json"], {
    environment: { GPUPAL_BINARY: "/tmp/native-gpupal" },
    exists: () => true,
    spawn: (binary, args, options) => {
      invocation = { binary, args, options };
      return { error: null, signal: null, status: 7 };
    }
  });
  assert.equal(status, 7);
  assert.equal(invocation.binary, "/tmp/native-gpupal");
  assert.deepEqual(invocation.args, ["pipeline", "input.json"]);
  assert.equal(invocation.options.stdio, "inherit");
});

test("launcher fails clearly when no native binary is installed", () => {
  const messages = [];
  const status = run(["version"], {
    environment: { GPUPAL_BINARY: "/definitely/missing/gpupal" },
    exists: () => false,
    report: (message) => messages.push(message)
  });
  assert.equal(status, 127);
  assert.match(messages.join("\n"), /native binary is unavailable/);
});

test("installer accepts only immutable GPUPAL release entries", () => {
  const digest = "a".repeat(64);
  assert.doesNotThrow(() => validateEntry({
    url: "https://github.com/zymazza/GPUPAL/releases/download/v0.1.0/" +
      "gpupal-linux-x64.tar.gz",
    sha256: digest
  }));
  assert.throws(() => validateEntry({
    url: "https://example.com/gpupal.tar.gz",
    sha256: digest
  }), /incomplete or untrusted/);
  assert.throws(() => validateEntry({
    url: "https://github.com/zymazza/GPUPAL/releases/download/latest/" +
      "gpupal-linux-x64.tar.gz",
    sha256: "not-a-digest"
  }), /incomplete or untrusted/);
});

test("installer platform selection and SHA-256 are deterministic", () => {
  assert.equal(installerPlatformKey("linux", "x64"), "linux-x64");
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "gpupal-npm-test-"));
  const fixture = path.join(directory, "fixture");
  try {
    fs.writeFileSync(fixture, "gpupal\n");
    assert.equal(
      sha256(fixture),
      "9da3e91df58e220a03809a41d5bea6eaa85ed54b6acc3c750d97446c4754bf44"
    );
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
