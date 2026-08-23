#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { PACKAGE_SET } = require("./package-set.js");
const { sha256 } = require("./native.js");
const { validateSplitPackage } = require("./validate-split-package.js");

function validateReleaseSet(root) {
  const rootPackage = path.resolve(root);
  const stagingRoot = path.dirname(rootPackage);
  const metadata = JSON.parse(fs.readFileSync(
    path.join(rootPackage, "package.json"), "utf8"));
  const manifest = JSON.parse(fs.readFileSync(
    path.join(rootPackage, "native-manifest.json"), "utf8"));
  if (manifest.schema !== 4 || manifest.version !== metadata.version) {
    throw new Error("root package manifest is not a staged split release");
  }

  let verifiedFiles = 0;
  for (const [key, expected] of Object.entries(PACKAGE_SET)) {
    const entry = manifest.platforms?.[key];
    if (!entry || entry.nativePackage !== expected.nativePackage ||
        entry.runtimePackage !== expected.runtimePackage) {
      throw new Error(`root package has an invalid ${key} package mapping`);
    }
    for (const packageName of [entry.nativePackage, entry.runtimePackage]) {
      if (metadata.optionalDependencies?.[packageName] !== metadata.version) {
        throw new Error(`root package does not pin ${packageName}`);
      }
    }

    const nativeRoot = path.join(stagingRoot, entry.nativePackage);
    const runtimeRoot = path.join(stagingRoot, entry.runtimePackage);
    const nativeResult = validateSplitPackage(nativeRoot);
    const runtimeResult = validateSplitPackage(runtimeRoot);
    if (nativeResult.manifest.role !== "native" ||
        runtimeResult.manifest.role !== "cuda-runtime" ||
        nativeResult.manifest.platform !== key ||
        runtimeResult.manifest.platform !== key) {
      throw new Error(`split-package roles are invalid for ${key}`);
    }

    const sumsPath = path.join(nativeRoot, "native", key, "SHA256SUMS");
    const sums = fs.readFileSync(sumsPath, "utf8").trimEnd().split(/\r?\n/);
    const owners = new Set();
    for (const line of sums) {
      const match = /^([0-9a-f]{64})  (\.\/.+)$/.exec(line);
      if (!match) {
        throw new Error(`invalid ${key} checksum entry: ${line}`);
      }
      const relative = match[2].slice(2);
      const nativeFile = path.join(nativeRoot, "native", key, relative);
      const runtimeFile = path.join(runtimeRoot, "native", key, relative);
      const candidates = [nativeFile, runtimeFile].filter((filename) =>
        fs.lstatSync(filename, { throwIfNoEntry: false })?.isFile());
      if (candidates.length !== 1 || owners.has(relative) ||
          sha256(candidates[0]) !== match[1]) {
        throw new Error(`split release does not uniquely preserve ${key}/${relative}`);
      }
      owners.add(relative);
    }
    if (owners.size !== entry.payloadFiles) {
      throw new Error(`split release payload count differs for ${key}`);
    }
    verifiedFiles += owners.size;
  }

  console.log(
    `gpupdal npm release set ${metadata.version} is ready ` +
      `(5 packages, ${verifiedFiles} native payload files verified)`
  );
  return verifiedFiles;
}

if (require.main === module) {
  validateReleaseSet(path.resolve(__dirname, ".."));
}

module.exports = { validateReleaseSet };
