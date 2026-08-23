#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { PACKAGE_SET } = require("./package-set.js");

const root = path.resolve(__dirname, "..");
const metadata = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const manifest = JSON.parse(fs.readFileSync(
  path.join(root, "native-manifest.json"), "utf8"));

if (metadata.version.includes("development") || metadata.version.endsWith("-dev") ||
    manifest.schema !== 4 || manifest.version !== metadata.version ||
    fs.existsSync(path.join(root, "native"))) {
  throw new Error("npm launcher package is not a staged split release");
}
const supported = Object.keys(PACKAGE_SET).sort();
if (Object.keys(manifest.platforms || {}).sort().join(",") !==
    supported.join(",")) {
  throw new Error("npm launcher manifest must declare every supported platform");
}
for (const [key, expected] of Object.entries(PACKAGE_SET)) {
  const entry = manifest.platforms[key];
  if (entry.nativePackage !== expected.nativePackage ||
      entry.runtimePackage !== expected.runtimePackage ||
      !/^[0-9a-f]{64}$/.test(entry.sourceArchiveSha256 || "") ||
      !Number.isInteger(entry.payloadFiles) || entry.payloadFiles < 1) {
    throw new Error(`npm launcher manifest has an invalid ${key} entry`);
  }
  for (const packageName of [entry.nativePackage, entry.runtimePackage]) {
    if (metadata.optionalDependencies?.[packageName] !== metadata.version) {
      throw new Error(`npm launcher does not pin ${packageName}`);
    }
  }
}
console.log(
  `gpupdal npm launcher ${metadata.version} is ready to pack ` +
    `(4 platform support packages pinned)`
);
