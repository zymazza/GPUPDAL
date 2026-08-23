"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {
  executableName,
  nativeEnvironment,
  packagedBinary,
  packagedLayout,
  run
} = require("../bin/gpupdal.js");
const { PACKAGE_SET, packageNames } = require("../scripts/package-set.js");
const {
  platformKey: installerPlatformKey,
  sha256,
  validateEntry,
  verifyNativeTree
} = require("../scripts/native.js");
const { validateSplitPackage } = require("../scripts/validate-split-package.js");

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
    sourceArchive: "gpupdal-0.1.0-linux-x64-cuda13.tar.gz",
    sourceArchiveSha256: digest,
    accelerator: {
      type: "cuda",
      toolkitMajor: 13,
      minimumDriverMajor: 580,
      physicallyQualifiedComputeCapabilities: ["8.9"]
    }
  }, "linux-x64"));
  assert.throws(() => validateEntry({
    directory: "../native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64-cuda13.tar.gz",
    sourceArchiveSha256: digest,
    accelerator: {
      type: "cuda",
      toolkitMajor: 13,
      minimumDriverMajor: 580,
      physicallyQualifiedComputeCapabilities: ["8.9"]
    }
  }, "linux-x64"), /incomplete or untrusted/);
  assert.throws(() => validateEntry({
    directory: "native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64-cuda13.tar.gz",
    sourceArchiveSha256: "not-a-digest",
    accelerator: {
      type: "cuda",
      toolkitMajor: 13,
      minimumDriverMajor: 580,
      physicallyQualifiedComputeCapabilities: ["8.9"]
    }
  }, "linux-x64"), /incomplete or untrusted/);
  assert.throws(() => validateEntry({
    directory: "native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64-cpu.tar.gz",
    sourceArchiveSha256: digest,
    accelerator: { type: "cpu" }
  }, "linux-x64"), /incomplete or untrusted/);
  assert.doesNotThrow(() => validateEntry({
    directory: "native/win32-x64",
    sourceArchive: "gpupdal-0.1.0-win32-x64-cuda13.zip",
    sourceArchiveSha256: digest,
    accelerator: {
      type: "cuda",
      toolkitMajor: 13,
      minimumDriverMajor: 580,
      physicallyQualifiedComputeCapabilities: ["8.9"]
    }
  }, "win32-x64"));
});

test("package platform selection and SHA-256 are deterministic", () => {
  const resolver = (request) =>
    path.join("/packages", request.slice(0, -"/package.json".length),
              "package.json");
  assert.equal(installerPlatformKey("linux", "x64"), "linux-x64");
  assert.equal(installerPlatformKey("win32", "x64"), "win32-x64");
  assert.equal(executableName("linux"), "gpupdal");
  assert.equal(executableName("win32"), "gpupdal.exe");
  assert.match(packagedBinary("win32", "x64", resolver),
               /gpupdal-win32-x64\/native\/win32-x64\/gpupdal\.exe$/);
  assert.deepEqual(packageNames(), [
    "@zymazza/gpupdal-linux-x64",
    "@zymazza/gpupdal-cuda13-linux-x64",
    "@zymazza/gpupdal-win32-x64",
    "@zymazza/gpupdal-cuda13-win32-x64"
  ]);
  assert.equal(PACKAGE_SET["linux-x64"].runtimeFiles.length, 2);
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

test("launcher resolves split native packages and configures runtime search", () => {
  const resolver = (request) =>
    path.join("/packages", request.slice(0, -"/package.json".length),
              "package.json");
  const layout = packagedLayout("linux", "x64", resolver);
  const environment = nativeEnvironment(
    { PATH: "/usr/bin", LD_LIBRARY_PATH: "/existing" }, layout, "linux");
  assert.equal(layout.binary,
    "/packages/@zymazza/gpupdal-linux-x64/native/linux-x64/gpupdal");
  assert.equal(
    environment.LD_LIBRARY_PATH,
    "/packages/@zymazza/gpupdal-linux-x64/native/linux-x64/lib:" +
      "/packages/@zymazza/gpupdal-cuda13-linux-x64/native/linux-x64/lib:" +
      "/existing"
  );
  assert.equal(environment.GDAL_DATA,
    "/packages/@zymazza/gpupdal-linux-x64/native/linux-x64/share/gdal");

  const windowsLayout = packagedLayout("win32", "x64", resolver);
  const windowsEnvironment = nativeEnvironment(
    { Path: "C:\\Windows\\System32" }, windowsLayout, "win32");
  assert.equal(windowsEnvironment.Path,
    "/packages/@zymazza/gpupdal-win32-x64/native/win32-x64;" +
      "/packages/@zymazza/gpupdal-cuda13-win32-x64/native/win32-x64;" +
      "C:\\Windows\\System32");
});

test("package verifier checks every staged native file", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "gpupdal-native-test-"));
  const native = path.join(directory, "native", "linux-x64");
  const entry = {
    directory: "native/linux-x64",
    sourceArchive: "gpupdal-0.1.0-linux-x64-cuda13.tar.gz",
    sourceArchiveSha256: "a".repeat(64),
    accelerator: {
      type: "cuda",
      toolkitMajor: 13,
      minimumDriverMajor: 580,
      physicallyQualifiedComputeCapabilities: ["8.9"]
    }
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

test("Windows native verification requires .exe files without Unix mode bits", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "gpupdal-win-test-"));
  const native = path.join(directory, "native", "win32-x64");
  const entry = {
    directory: "native/win32-x64",
    sourceArchive: "gpupdal-0.1.0-win32-x64-cuda13.zip",
    sourceArchiveSha256: "a".repeat(64),
    accelerator: {
      type: "cuda",
      toolkitMajor: 13,
      minimumDriverMajor: 580,
      physicallyQualifiedComputeCapabilities: ["8.9"]
    }
  };
  try {
    fs.mkdirSync(native, { recursive: true });
    const names = ["gpupdal.exe", "pdg-engine.exe", "pdal.exe"];
    for (const executable of names) {
      fs.writeFileSync(path.join(native, executable), `${executable}\n`, {
        mode: 0o644
      });
    }
    fs.writeFileSync(
      path.join(native, "SHA256SUMS"),
      `${names.map((filename) =>
        `${sha256(path.join(native, filename))}  ./${filename}`
      ).join("\r\n")}\r\n`
    );
    assert.equal(verifyNativeTree(directory, entry), 3);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("split-package verifier rejects missing and modified native files", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "gpupdal-split-test-"));
  const native = path.join(directory, "native", "linux-x64");
  try {
    fs.mkdirSync(native, { recursive: true });
    fs.writeFileSync(path.join(directory, "package.json"), JSON.stringify({
      name: "@zymazza/gpupdal-linux-x64",
      version: "0.1.0"
    }));
    const filename = path.join(native, "gpupdal");
    fs.writeFileSync(filename, "gpupdal\n");
    fs.writeFileSync(path.join(directory, "split-manifest.json"), JSON.stringify({
      schema: 1,
      package: "@zymazza/gpupdal-linux-x64",
      version: "0.1.0",
      platform: "linux-x64",
      role: "native",
      files: { gpupdal: sha256(filename) }
    }));
    assert.equal(validateSplitPackage(directory).verifiedFiles, 1);
    fs.appendFileSync(filename, "modified\n");
    assert.throws(() => validateSplitPackage(directory), /checksum mismatch/);
    fs.writeFileSync(filename, "gpupdal\n");
    fs.writeFileSync(path.join(native, "extra"), "extra\n");
    assert.throws(() => validateSplitPackage(directory), /inventory differs/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
