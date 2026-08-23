#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { PACKAGE_SET } = require("../scripts/package-set.js");

function platformKey(platform = process.platform, arch = process.arch) {
  return `${platform}-${arch}`;
}

function executableName(platform = process.platform) {
  return platform === "win32" ? "gpupdal.exe" : "gpupdal";
}

function resolvePackageRoot(packageName, resolver = require.resolve) {
  return path.dirname(resolver(`${packageName}/package.json`));
}

function packagedLayout(platform = process.platform, arch = process.arch,
                        resolver = require.resolve) {
  const key = `${platform}-${arch}`;
  const packages = PACKAGE_SET[key];
  if (!packages) {
    throw new Error(`unsupported platform ${key}`);
  }
  const nativeRoot = path.join(
    resolvePackageRoot(packages.nativePackage, resolver), "native", key);
  const runtimeRoot = path.join(
    resolvePackageRoot(packages.runtimePackage, resolver), "native", key);
  return {
    binary: path.join(nativeRoot, executableName(platform)),
    key,
    nativeRoot,
    runtimeRoot,
    runtimeSearchDirectory: packages.runtimeSearchDirectory
  };
}

function packagedBinary(platform = process.platform, arch = process.arch,
                        resolver = require.resolve) {
  return packagedLayout(platform, arch, resolver).binary;
}

function nativeEnvironment(environment, layout, platform = process.platform) {
  const configured = { ...environment };
  const delimiter = platform === "win32" ? ";" : ":";
  if (platform === "win32") {
    const pathKey = Object.keys(configured).find((name) =>
      name.toLowerCase() === "path") || "PATH";
    configured[pathKey] = [
      layout.nativeRoot,
      path.resolve(layout.runtimeRoot, layout.runtimeSearchDirectory),
      configured[pathKey]
    ].filter(Boolean).join(delimiter);
  } else {
    configured.LD_LIBRARY_PATH = [
      path.join(layout.nativeRoot, "lib"),
      path.resolve(layout.runtimeRoot, layout.runtimeSearchDirectory),
      configured.LD_LIBRARY_PATH
    ].filter(Boolean).join(delimiter);
  }

  const defaults = {
    GDAL_DATA: path.join(layout.nativeRoot, "share", "gdal"),
    PROJ_DATA: path.join(layout.nativeRoot, "share", "proj"),
    CURL_CA_BUNDLE: path.join(layout.nativeRoot, "share", "certs", "cacert.pem"),
    SSL_CERT_FILE: path.join(layout.nativeRoot, "share", "certs", "cacert.pem")
  };
  for (const [name, value] of Object.entries(defaults)) {
    if (!configured[name]) {
      configured[name] = value;
    }
  }
  return configured;
}

function run(args, options = {}) {
  const environment = options.environment || process.env;
  const exists = options.exists || fs.existsSync;
  const spawn = options.spawn || spawnSync;
  const report = options.report || console.error;
  const forwardSignal = options.forwardSignal ||
    ((signal) => process.kill(process.pid, signal));
  let binary = environment.GPUPDAL_BINARY;
  let spawnEnvironment = environment;
  if (!binary) {
    try {
      const layout = packagedLayout(
        options.platform || process.platform,
        options.arch || process.arch,
        options.resolve || require.resolve);
      binary = layout.binary;
      spawnEnvironment = nativeEnvironment(
        environment, layout, options.platform || process.platform);
    } catch (error) {
      report(
        `gpupdal: native support packages are unavailable for ${platformKey(
          options.platform || process.platform,
          options.arch || process.arch)}: ` +
          `${error.message}. Reinstall gpupdal from npm.`
      );
      return 127;
    }
  }

  if (!exists(binary)) {
    report(
      `gpupdal: native binary is unavailable for ${platformKey()}. ` +
        "Reinstall from a supported release or set GPUPDAL_BINARY to a tested " +
        "GPUPDAL executable."
    );
    return 127;
  }

  const result = spawn(binary, args, {
    env: spawnEnvironment,
    stdio: "inherit"
  });
  if (result.error) {
    report(`gpupdal: unable to execute ${binary}: ${result.error.message}`);
    return 126;
  }
  if (result.signal) {
    forwardSignal(result.signal);
    return 1;
  }
  return result.status === null ? 1 : result.status;
}

if (require.main === module) {
  process.exit(run(process.argv.slice(2)));
}

module.exports = {
  executableName,
  nativeEnvironment,
  packagedBinary,
  packagedLayout,
  platformKey,
  resolvePackageRoot,
  run
};
