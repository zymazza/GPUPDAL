#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { validateEntry } = require("./install.js");

const root = path.resolve(__dirname, "..");
const metadata = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const manifest = JSON.parse(fs.readFileSync(
  path.join(root, "native-manifest.json"), "utf8"));

if (metadata.version.includes("development") || manifest.version !== metadata.version) {
  throw new Error("npm package and native manifest need the same release version");
}
validateEntry(manifest.platforms["linux-x64"]);
console.log(`gpupal npm package ${metadata.version} is ready to pack`);
