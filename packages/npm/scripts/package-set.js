#!/usr/bin/env node

"use strict";

const PACKAGE_SET = Object.freeze({
  "linux-x64": Object.freeze({
    os: "linux",
    cpu: "x64",
    nativePackage: "gpupdal-linux-x64",
    runtimePackage: "gpupdal-cuda13-linux-x64",
    runtimeFiles: Object.freeze([
      "lib/libnvrtc.so.13",
      "lib/libnvrtc-builtins.so.13.3"
    ]),
    runtimeSearchDirectory: "lib"
  }),
  "win32-x64": Object.freeze({
    os: "win32",
    cpu: "x64",
    nativePackage: "gpupdal-win32-x64",
    runtimePackage: "gpupdal-cuda13-win32-x64",
    runtimeFiles: Object.freeze([
      "nvrtc64_130_0.dll",
      "nvrtc-builtins64_133.dll"
    ]),
    runtimeSearchDirectory: "."
  })
});

function packageNames() {
  return Object.values(PACKAGE_SET).flatMap((entry) => [
    entry.nativePackage,
    entry.runtimePackage
  ]);
}

module.exports = { PACKAGE_SET, packageNames };
