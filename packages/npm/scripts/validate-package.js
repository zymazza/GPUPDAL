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
    manifest.schema !== 2 || manifest.version !== metadata.version) {
  throw new Error("npm package and native manifest need the same release version");
}
const entry = manifest.platforms["linux-x64"];
validateEntry(entry);
const verifiedFiles = verifyNativeTree(root, entry);
console.log(
  `gpupdal npm package ${metadata.version} is ready to pack ` +
    `(${verifiedFiles} native files verified)`
);
