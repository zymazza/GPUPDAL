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
  validateEntry,
  verifyNativeTree
} = require("../scripts/native.js");

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

test("package accepts only checksummed GPUPDAL native trees", () => {
  const digest = "a".repeat(64);
  assert.doesNotThrow(() => validateEntry({
    directory: "native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64.tar.gz",
    sourceArchiveSha256: digest
  }));
  assert.throws(() => validateEntry({
    directory: "../native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64.tar.gz",
    sourceArchiveSha256: digest
  }), /incomplete or untrusted/);
  assert.throws(() => validateEntry({
    directory: "native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64.tar.gz",
    sourceArchiveSha256: "not-a-digest"
  }), /incomplete or untrusted/);
});

test("package platform selection and SHA-256 are deterministic", () => {
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

test("package verifier checks every staged native file", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "gpupdal-native-test-"));
  const native = path.join(directory, "native", "linux-x64");
  const entry = {
    directory: "native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64.tar.gz",
    sourceArchiveSha256: "a".repeat(64)
  };
  try {
    fs.mkdirSync(native, { recursive: true });
    for (const executable of ["gpupdal", "pdg-engine", "pdal"]) {
      const filename = path.join(native, executable);
      fs.writeFileSync(filename, `${executable}\n`, { mode: 0o755 });
    }
    const sums = ["gpupdal", "pdg-engine", "pdal"].map((filename) =>
      `${sha256(path.join(native, filename))}  ./${filename}`
    );
    fs.writeFileSync(path.join(native, "SHA256SUMS"), `${sums.join("\n")}\n`);
    assert.equal(verifyNativeTree(directory, entry), 3);
    fs.writeFileSync(path.join(native, "unlisted"), "not checksummed\n");
    assert.throws(
      () => verifyNativeTree(directory, entry), /unchecksummed file/
    );
    fs.rmSync(path.join(native, "unlisted"));
    fs.appendFileSync(path.join(native, "gpupdal"), "tampered\n");
    assert.throws(() => verifyNativeTree(directory, entry), /checksum mismatch/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
