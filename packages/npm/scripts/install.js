#!/usr/bin/env node

"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const https = require("node:https");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const packageRoot = path.resolve(__dirname, "..");

function platformKey(platform = process.platform, arch = process.arch) {
  return `${platform}-${arch}`;
}

function sha256(filename) {
  const hash = crypto.createHash("sha256");
  hash.update(fs.readFileSync(filename));
  return hash.digest("hex");
}

function download(url, destination, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) {
      reject(new Error("too many release-asset redirects"));
      return;
    }
    const request = https.get(url, { headers: { "User-Agent": "gpupdal-npm" } },
      (response) => {
        if (response.statusCode >= 300 && response.statusCode < 400 &&
            response.headers.location) {
          response.resume();
          resolve(download(response.headers.location, destination, redirects + 1));
          return;
        }
        if (response.statusCode !== 200) {
          response.resume();
          reject(new Error(`release download returned HTTP ${response.statusCode}`));
          return;
        }
        const output = fs.createWriteStream(destination, { mode: 0o600 });
        response.pipe(output);
        output.on("finish", () => output.close(resolve));
        output.on("error", reject);
      });
    request.on("error", reject);
  });
}

function validateEntry(entry) {
  if (!entry || typeof entry.url !== "string" ||
      !entry.url.startsWith("https://github.com/zymazza/GPUPDAL/releases/download/") ||
      !/^[0-9a-f]{64}$/.test(entry.sha256 || "")) {
    throw new Error("release manifest entry is incomplete or untrusted");
  }
}

async function install() {
  if (process.env.GPUPDAL_SKIP_DOWNLOAD === "1") {
    console.warn("gpupdal: native download skipped by GPUPDAL_SKIP_DOWNLOAD=1");
    return;
  }

  const manifest = JSON.parse(fs.readFileSync(
    path.join(packageRoot, "native-manifest.json"), "utf8"));
  const key = platformKey();
  const entry = manifest.platforms[key];
  validateEntry(entry);

  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "gpupdal-install-"));
  const archive = path.join(temporary, "gpupdal.tar.gz");
  const destination = path.join(packageRoot, "native", key);
  try {
    await download(entry.url, archive);
    const actual = sha256(archive);
    if (actual !== entry.sha256) {
      throw new Error(`release checksum mismatch: expected ${entry.sha256}, got ${actual}`);
    }
    fs.rmSync(destination, { recursive: true, force: true });
    fs.mkdirSync(destination, { recursive: true, mode: 0o755 });
    const extracted = spawnSync("tar", [
      "--extract", "--gzip", "--file", archive,
      "--directory", destination,
      "--no-same-owner", "--no-same-permissions"
    ], { stdio: "inherit" });
    if (extracted.error || extracted.status !== 0) {
      throw extracted.error || new Error(`tar exited with status ${extracted.status}`);
    }
    for (const executable of ["gpupdal", "pdg-engine", "pdal"]) {
      const filename = path.join(destination, executable);
      if (!fs.statSync(filename, { throwIfNoEntry: false })?.isFile()) {
        throw new Error(`release archive is missing ${executable}`);
      }
      fs.chmodSync(filename, 0o755);
    }
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

if (require.main === module) {
  install().catch((error) => {
    console.error(`gpupdal: native installation failed: ${error.message}`);
    process.exit(1);
  });
}

module.exports = { install, platformKey, sha256, validateEntry };
