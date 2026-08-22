#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { validateEntry, verifyNativeTree } = require("./native.js");

const root = path.resolve(__dirname, "..");
const metadata = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const manifest = JSON.parse(fs.readFileSync(
  path.join(root, "native-manifest.json"), "utf8"));

if (metadata.version.includes("development") || metadata.version.endsWith("-dev") ||
    manifest.schema !== 3 || manifest.version !== metadata.version) {
  throw new Error("npm package and native manifest need the same release version");
}
const supported = ["linux-x64", "win32-x64"];
if (Object.keys(manifest.platforms).sort().join(",") !== supported.join(",")) {
  throw new Error("npm package must contain the complete supported platform set");
}
let verifiedFiles = 0;
for (const key of supported) {
  const entry = manifest.platforms[key];
  validateEntry(entry, key);
  verifiedFiles += verifyNativeTree(root, entry);
}
console.log(
  `gpupdal npm package ${metadata.version} is ready to pack ` +
    `(${verifiedFiles} native files verified)`
);
