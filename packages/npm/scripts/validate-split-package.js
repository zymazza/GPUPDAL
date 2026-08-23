#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { sha256 } = require("./native.js");

function regularFiles(directory) {
  const root = path.resolve(directory);
  const files = [];
  const pending = [root];
  while (pending.length > 0) {
    const current = pending.pop();
    for (const name of fs.readdirSync(current)) {
      const filename = path.join(current, name);
      const status = fs.lstatSync(filename);
      if (status.isDirectory()) {
        pending.push(filename);
      } else if (status.isFile()) {
        files.push(path.relative(root, filename).split(path.sep).join("/"));
      } else {
        throw new Error(`unsupported package entry: ${filename}`);
      }
    }
  }
  return files.sort();
}

function validateSplitPackage(root) {
  const packageRoot = path.resolve(root);
  const metadata = JSON.parse(fs.readFileSync(
    path.join(packageRoot, "package.json"), "utf8"));
  const manifest = JSON.parse(fs.readFileSync(
    path.join(packageRoot, "split-manifest.json"), "utf8"));
  if (manifest.schema !== 1 || manifest.package !== metadata.name ||
      manifest.version !== metadata.version ||
      !["native", "cuda-runtime"].includes(manifest.role) ||
      typeof manifest.platform !== "string" ||
      typeof manifest.files !== "object" || Array.isArray(manifest.files)) {
    throw new Error("split-package metadata is inconsistent");
  }
  const nativeRoot = path.join(packageRoot, "native", manifest.platform);
  const actualFiles = regularFiles(nativeRoot);
  const declaredFiles = Object.keys(manifest.files).sort();
  if (actualFiles.join("\n") !== declaredFiles.join("\n")) {
    throw new Error("split package native-file inventory differs from manifest");
  }
  for (const relative of declaredFiles) {
    const digest = manifest.files[relative];
    if (!/^[0-9a-f]{64}$/.test(digest) ||
        sha256(path.join(nativeRoot, relative)) !== digest) {
      throw new Error(`split package checksum mismatch: ${relative}`);
    }
  }
  return { manifest, verifiedFiles: declaredFiles.length };
}

if (require.main === module) {
  const result = validateSplitPackage(path.resolve(__dirname, ".."));
  console.log(
    `${result.manifest.package}@${result.manifest.version} ` +
      `verified ${result.verifiedFiles} native files`
  );
}

module.exports = { regularFiles, validateSplitPackage };
