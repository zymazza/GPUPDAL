#!/usr/bin/env node

"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

function platformKey(platform = process.platform, arch = process.arch) {
  return `${platform}-${arch}`;
}

function sha256(filename) {
  const hash = crypto.createHash("sha256");
  hash.update(fs.readFileSync(filename));
  return hash.digest("hex");
}

function executableNames(key) {
  return key.startsWith("win32-")
    ? ["gpupdal.exe", "pdg-engine.exe", "pdal.exe"]
    : ["gpupdal", "pdg-engine", "pdal"];
}

function archivePattern(key) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const extension = key.startsWith("win32-") ? "\\.zip" : "\\.tar\\.gz";
  return new RegExp(
    `^gpupdal-[0-9A-Za-z][0-9A-Za-z._-]*-${escaped}-cuda13${extension}$`
  );
}

function validateEntry(entry, key) {
  if (!entry || entry.directory !== `native/${key}` ||
      typeof entry.sourceArchive !== "string" ||
      !archivePattern(key).test(entry.sourceArchive) ||
      !/^[0-9a-f]{64}$/.test(entry.sourceArchiveSha256 || "") ||
      entry.accelerator?.type !== "cuda" ||
      entry.accelerator?.toolkitMajor !== 13 ||
      entry.accelerator?.minimumDriverMajor !== 580 ||
      !Array.isArray(entry.accelerator?.physicallyQualifiedComputeCapabilities) ||
      entry.accelerator.physicallyQualifiedComputeCapabilities.length !== 1 ||
      entry.accelerator.physicallyQualifiedComputeCapabilities[0] !== "8.9") {
    throw new Error("release manifest entry is incomplete or untrusted");
  }
}

function verifyNativeTree(packageRoot, entry) {
  const root = path.resolve(packageRoot);
  const directory = path.resolve(root, entry.directory);
  const nativeRoot = path.resolve(root, "native");
  if (!directory.startsWith(`${nativeRoot}${path.sep}`)) {
    throw new Error("native directory escapes the package root");
  }
  for (const executable of executableNames(entry.directory.slice("native/".length))) {
    const status = fs.lstatSync(path.join(directory, executable), {
      throwIfNoEntry: false
    });
    const needsExecuteBit = !entry.directory.startsWith("native/win32-");
    if (!status?.isFile() || (needsExecuteBit && (status.mode & 0o111) === 0)) {
      throw new Error(`native package is missing ${executable}`);
    }
  }

  const sumsPath = path.join(directory, "SHA256SUMS");
  if (!fs.lstatSync(sumsPath, { throwIfNoEntry: false })?.isFile()) {
    throw new Error("native SHA256SUMS is missing");
  }
  const lines = fs.readFileSync(sumsPath, "utf8").trimEnd().split(/\r?\n/);
  if (lines.length === 0) {
    throw new Error("native SHA256SUMS is empty");
  }
  const checkedFiles = new Set();
  for (const line of lines) {
    const match = /^([0-9a-f]{64})  (\.\/.+)$/.exec(line);
    if (!match) {
      throw new Error(`invalid native checksum entry: ${line}`);
    }
    const filename = path.resolve(directory, match[2]);
    const relative = `./${path.relative(directory, filename).split(path.sep).join("/")}`;
    if (!filename.startsWith(`${directory}${path.sep}`) ||
        match[2] !== relative || checkedFiles.has(relative) ||
        !fs.lstatSync(filename, { throwIfNoEntry: false })?.isFile()) {
      throw new Error(`invalid native checksum path: ${match[2]}`);
    }
    const actual = sha256(filename);
    if (actual !== match[1]) {
      throw new Error(`native checksum mismatch: ${match[2]}`);
    }
    checkedFiles.add(relative);
  }

  const stagedFiles = new Set();
  const pendingDirectories = [directory];
  while (pendingDirectories.length > 0) {
    const current = pendingDirectories.pop();
    for (const name of fs.readdirSync(current)) {
      const filename = path.join(current, name);
      const status = fs.lstatSync(filename);
      if (status.isDirectory()) {
        pendingDirectories.push(filename);
      } else if (status.isFile()) {
        if (filename !== sumsPath) {
          stagedFiles.add(
            `./${path.relative(directory, filename).split(path.sep).join("/")}`
          );
        }
      } else {
        throw new Error(`unsupported native package entry: ${filename}`);
      }
    }
  }
  if (stagedFiles.size !== checkedFiles.size ||
      [...stagedFiles].some((filename) => !checkedFiles.has(filename))) {
    throw new Error("native package contains an unchecksummed file");
  }
  return checkedFiles.size;
}

module.exports = {
  executableNames,
  platformKey,
  sha256,
  validateEntry,
  verifyNativeTree
};
