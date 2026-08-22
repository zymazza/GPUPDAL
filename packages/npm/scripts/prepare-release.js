#!/usr/bin/env node

"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

function usage() {
  console.error(
    "usage: prepare-release.js --version <semver> " +
      "--linux-archive <tar.gz> --windows-archive <zip> " +
      "--output <directory>"
  );
  process.exit(2);
}

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) {
    usage();
  }
  return process.argv[index + 1];
}

function sha256(filename) {
  const hash = crypto.createHash("sha256");
  hash.update(fs.readFileSync(filename));
  return hash.digest("hex");
}

const version = argument("--version");
const linuxArchive = path.resolve(argument("--linux-archive"));
const windowsArchive = path.resolve(argument("--windows-archive"));
const output = path.resolve(argument("--output"));
const packageRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(packageRoot, "../..");
const allowedOutputRoot = path.join(projectRoot, "dist", "npm");

if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$/.test(version)) {
  throw new Error(`invalid release version: ${version}`);
}
if (version.endsWith("-dev")) {
  throw new Error("development archives cannot be staged for npm publication");
}
const expectedLinuxArchive = `gpupdal-${version}-linux-x64-cuda13.tar.gz`;
const expectedWindowsArchive = `gpupdal-${version}-win32-x64-cuda13.zip`;
if (path.basename(linuxArchive) !== expectedLinuxArchive ||
    !fs.statSync(linuxArchive, { throwIfNoEntry: false })?.isFile()) {
  throw new Error(`expected a built archive named ${expectedLinuxArchive}`);
}
if (path.basename(windowsArchive) !== expectedWindowsArchive ||
    !fs.statSync(windowsArchive, { throwIfNoEntry: false })?.isFile()) {
  throw new Error(`expected a built archive named ${expectedWindowsArchive}`);
}
if (!output.startsWith(`${allowedOutputRoot}${path.sep}`)) {
  throw new Error(`release staging output must be below ${allowedOutputRoot}`);
}

fs.rmSync(output, { recursive: true, force: true });
fs.mkdirSync(output, { recursive: true, mode: 0o755 });
fs.cpSync(packageRoot, output, {
  recursive: true,
  filter: (source) => {
    const relative = path.relative(packageRoot, source);
    return relative !== "native";
  }
});

const nativeDirectory = path.join(output, "native", "linux-x64");
fs.mkdirSync(nativeDirectory, { recursive: true, mode: 0o755 });
const extracted = spawnSync("tar", [
  "--extract", "--gzip", "--file", linuxArchive,
  "--directory", nativeDirectory,
  "--no-same-owner", "--no-same-permissions"
], { stdio: "inherit" });
if (extracted.error || extracted.status !== 0) {
  throw extracted.error || new Error(`tar exited with status ${extracted.status}`);
}
for (const executable of ["gpupdal", "pdg-engine", "pdal"]) {
  fs.chmodSync(path.join(nativeDirectory, executable), 0o755);
}

const windowsDirectory = path.join(output, "native", "win32-x64");
fs.mkdirSync(windowsDirectory, { recursive: true, mode: 0o755 });
const unzipProbe = spawnSync("unzip", ["-v"], { stdio: "ignore" });
if (unzipProbe.error || unzipProbe.status !== 0) {
  throw unzipProbe.error ||
    new Error("unzip is required to stage the Windows release archive");
}
const windowsExtracted = spawnSync("unzip", [
  "-q", windowsArchive, "-d", windowsDirectory
], { encoding: "utf8" });
const windowsWarning = windowsExtracted.stderr?.trim() || "";
const knownPowerShellZipWarning =
  /^warning:\s+.+ appears to use backslashes as path separators$/;
const knownPowerShellZipResult =
  windowsExtracted.status === 1 &&
  knownPowerShellZipWarning.test(windowsWarning) &&
  !(windowsExtracted.stdout?.trim()) &&
  (!windowsExtracted.error || windowsExtracted.error.code === "EPERM");
if ((windowsExtracted.error && !knownPowerShellZipResult) ||
    (windowsExtracted.status !== 0 && !knownPowerShellZipResult)) {
  if (windowsExtracted.stdout) {
    process.stdout.write(windowsExtracted.stdout);
  }
  if (windowsExtracted.stderr) {
    process.stderr.write(windowsExtracted.stderr);
  }
  throw windowsExtracted.error ||
    new Error(`unzip exited with status ${windowsExtracted.status}`);
}

// Windows Compress-Archive records DOS paths and file attributes. Info-ZIP
// translates the paths correctly, but its DOS-to-Unix mode mapping can create
// directories without execute/search bits. Normalize the trusted, hash-pinned
// release tree before the package validator walks and checksums every entry.
function normalizeWindowsTree(directory) {
  fs.chmodSync(directory, 0o755);
  for (const name of fs.readdirSync(directory)) {
    const filename = path.join(directory, name);
    const status = fs.lstatSync(filename);
    if (status.isDirectory()) {
      normalizeWindowsTree(filename);
    } else if (status.isFile()) {
      fs.chmodSync(filename, 0o644);
    } else {
      throw new Error(`unsupported Windows archive entry: ${filename}`);
    }
  }
}
normalizeWindowsTree(windowsDirectory);

const metadataPath = path.join(output, "package.json");
const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
metadata.version = version;
fs.writeFileSync(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

const manifestPath = path.join(output, "native-manifest.json");
const manifest = {
  schema: 3,
  version,
  platforms: {
    "linux-x64": {
      directory: "native/linux-x64",
      sourceArchive: expectedLinuxArchive,
      sourceArchiveSha256: sha256(linuxArchive),
      accelerator: {
        type: "cuda",
        toolkitMajor: 13,
        minimumDriverMajor: 580,
        physicallyQualifiedComputeCapabilities: ["8.9"]
      }
    },
    "win32-x64": {
      directory: "native/win32-x64",
      sourceArchive: expectedWindowsArchive,
      sourceArchiveSha256: sha256(windowsArchive),
      accelerator: {
        type: "cuda",
        toolkitMajor: 13,
        minimumDriverMajor: 580,
        physicallyQualifiedComputeCapabilities: ["8.9"]
      }
    }
  }
};
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(`staged gpupdal npm package ${version} at ${output}`);
