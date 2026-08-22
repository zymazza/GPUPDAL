#!/usr/bin/env node

"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

function usage() {
  console.error(
    "usage: prepare-release.js --version <semver> --archive <tar.gz> " +
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
const archive = path.resolve(argument("--archive"));
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
const expectedArchive = `gpupdal-${version}-linux-x64.tar.gz`;
if (path.basename(archive) !== expectedArchive ||
    !fs.statSync(archive, { throwIfNoEntry: false })?.isFile()) {
  throw new Error(`expected a built archive named ${expectedArchive}`);
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
  "--extract", "--gzip", "--file", archive,
  "--directory", nativeDirectory,
  "--no-same-owner", "--no-same-permissions"
], { stdio: "inherit" });
if (extracted.error || extracted.status !== 0) {
  throw extracted.error || new Error(`tar exited with status ${extracted.status}`);
}
for (const executable of ["gpupdal", "pdg-engine", "pdal"]) {
  fs.chmodSync(path.join(nativeDirectory, executable), 0o755);
}

const metadataPath = path.join(output, "package.json");
const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
metadata.version = version;
fs.writeFileSync(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

const manifestPath = path.join(output, "native-manifest.json");
const manifest = {
  schema: 1,
  version,
  platforms: {
    "linux-x64": {
      directory: "native/linux-x64",
      sourceArchive: expectedArchive,
      sourceArchiveSha256: sha256(archive)
    }
  }
};
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(`staged gpupdal npm package ${version} at ${output}`);
