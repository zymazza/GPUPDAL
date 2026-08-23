#!/usr/bin/env node

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { PACKAGE_SET, packageNames } = require("./package-set.js");
const { sha256, validateEntry, verifyNativeTree } = require("./native.js");
const { regularFiles } = require("./validate-split-package.js");
const { validateReleaseSet } = require("./validate-release-set.js");

function usage() {
  console.error(
    "usage: prepare-release.js --version <semver> " +
      "--linux-archive <tar.gz> --windows-archive <zip> " +
      "--output <directory>"
  );
  process.exit(2);
}

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) {
    usage();
  }
  return process.argv[index + 1];
}

function writeJson(filename, value) {
  fs.writeFileSync(filename, `${JSON.stringify(value, null, 2)}\n`);
}

function normalizeWindowsTree(directory) {
  fs.chmodSync(directory, 0o755);
  for (const name of fs.readdirSync(directory)) {
    const filename = path.join(directory, name);
    const status = fs.lstatSync(filename);
    if (status.isDirectory()) {
      normalizeWindowsTree(filename);
    } else if (status.isFile()) {
      fs.chmodSync(filename, 0o644);
    } else {
      throw new Error(`unsupported Windows archive entry: ${filename}`);
    }
  }
}

function extractLinux(archive, directory) {
  const result = spawnSync("tar", [
    "--extract", "--gzip", "--file", archive,
    "--directory", directory,
    "--no-same-owner", "--no-same-permissions"
  ], { stdio: "inherit" });
  if (result.error || result.status !== 0) {
    throw result.error || new Error(`tar exited with status ${result.status}`);
  }
  for (const executable of ["gpupdal", "pdg-engine", "pdal"]) {
    fs.chmodSync(path.join(directory, executable), 0o755);
  }
}

function extractWindows(archive, directory) {
  const probe = spawnSync("unzip", ["-v"], { stdio: "ignore" });
  if (probe.error || probe.status !== 0) {
    throw probe.error || new Error("unzip is required to stage Windows");
  }
  const result = spawnSync("unzip", ["-q", archive, "-d", directory], {
    encoding: "utf8"
  });
  const warning = result.stderr?.trim() || "";
  const knownPowerShellZipResult =
    result.status === 1 &&
    /^warning:\s+.+ appears to use backslashes as path separators$/.test(warning) &&
    !(result.stdout?.trim()) &&
    (!result.error || result.error.code === "EPERM");
  if ((result.error && !knownPowerShellZipResult) ||
      (result.status !== 0 && !knownPowerShellZipResult)) {
    if (result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);
    throw result.error || new Error(`unzip exited with status ${result.status}`);
  }
  normalizeWindowsTree(directory);
}

function packageMetadata(name, version, description, spec, dependencies = {}) {
  return {
    name,
    version,
    description,
    license: "BSD-3-Clause",
    repository: {
      type: "git",
      url: "git+https://github.com/zymazza/GPUPDAL.git"
    },
    homepage: "https://github.com/zymazza/GPUPDAL#readme",
    os: [spec.os],
    cpu: [spec.cpu],
    files: [
      "native/",
      "scripts/",
      "LICENSE.txt",
      "README.md",
      "split-manifest.json",
      "CUDA-EULA.txt",
      "NOTICE",
      "THIRD_PARTY_LICENSES.md"
    ],
    scripts: { prepack: "node scripts/validate-split-package.js" },
    dependencies,
    publishConfig: { access: "public", provenance: false },
    keywords: ["gpupdal", "pdal", "cuda", "native"]
  };
}

function copyIfFile(source, destination) {
  if (fs.lstatSync(source, { throwIfNoEntry: false })?.isFile()) {
    fs.copyFileSync(source, destination);
  }
}

function nativeFileManifest(nativeRoot) {
  return Object.fromEntries(regularFiles(nativeRoot).map((relative) => [
    relative,
    sha256(path.join(nativeRoot, relative))
  ]));
}

function finishSupportPackage(packageRoot, metadata, manifest, readme,
                              npmSourceRoot) {
  fs.mkdirSync(path.join(packageRoot, "scripts"), { recursive: true });
  fs.copyFileSync(path.join(npmSourceRoot, "LICENSE.txt"),
                  path.join(packageRoot, "LICENSE.txt"));
  fs.copyFileSync(path.join(npmSourceRoot, "scripts", "native.js"),
                  path.join(packageRoot, "scripts", "native.js"));
  fs.copyFileSync(
    path.join(npmSourceRoot, "scripts", "validate-split-package.js"),
    path.join(packageRoot, "scripts", "validate-split-package.js"));
  fs.writeFileSync(path.join(packageRoot, "README.md"), `${readme.trim()}\n`);
  writeJson(path.join(packageRoot, "package.json"), metadata);
  writeJson(path.join(packageRoot, "split-manifest.json"), manifest);
}

function stagePlatform({ key, spec, archive, archiveName, archiveDigest,
                         outputRoot, version, npmSourceRoot }) {
  const nativePackageRoot = path.join(outputRoot, spec.nativePackage);
  const runtimePackageRoot = path.join(outputRoot, spec.runtimePackage);
  for (const directory of [nativePackageRoot, runtimePackageRoot]) {
    fs.rmSync(directory, { recursive: true, force: true });
  }
  const nativeRoot = path.join(nativePackageRoot, "native", key);
  const runtimeRoot = path.join(runtimePackageRoot, "native", key);
  fs.mkdirSync(nativeRoot, { recursive: true, mode: 0o755 });
  fs.mkdirSync(runtimeRoot, { recursive: true, mode: 0o755 });
  if (key === "linux-x64") {
    extractLinux(archive, nativeRoot);
  } else {
    extractWindows(archive, nativeRoot);
  }

  const entry = {
    directory: `native/${key}`,
    sourceArchive: archiveName,
    sourceArchiveSha256: archiveDigest,
    accelerator: {
      type: "cuda",
      toolkitMajor: 13,
      minimumDriverMajor: 580,
      physicallyQualifiedComputeCapabilities: ["8.9"]
    }
  };
  validateEntry(entry, key);
  const payloadFiles = verifyNativeTree(nativePackageRoot, entry);

  for (const relative of spec.runtimeFiles) {
    const source = path.join(nativeRoot, relative);
    const destination = path.join(runtimeRoot, relative);
    if (!fs.lstatSync(source, { throwIfNoEntry: false })?.isFile()) {
      throw new Error(`release archive is missing split runtime file ${relative}`);
    }
    fs.mkdirSync(path.dirname(destination), { recursive: true, mode: 0o755 });
    fs.renameSync(source, destination);
  }

  const eula = key === "linux-x64"
    ? path.join(nativeRoot, "licenses", "source", "cuda", "EULA.txt")
    : path.join(nativeRoot, "licenses", "system", "nvidia-cuda-toolkit", "EULA.txt");
  copyIfFile(eula, path.join(runtimePackageRoot, "CUDA-EULA.txt"));
  copyIfFile(path.join(nativeRoot, "NOTICE"),
             path.join(runtimePackageRoot, "NOTICE"));
  copyIfFile(path.join(nativeRoot, "THIRD_PARTY_LICENSES.md"),
             path.join(runtimePackageRoot, "THIRD_PARTY_LICENSES.md"));

  const nativeManifest = {
    schema: 1,
    package: spec.nativePackage,
    version,
    platform: key,
    role: "native",
    sourceArchive: archiveName,
    sourceArchiveSha256: archiveDigest,
    files: nativeFileManifest(nativeRoot)
  };
  const runtimeManifest = {
    schema: 1,
    package: spec.runtimePackage,
    version,
    platform: key,
    role: "cuda-runtime",
    sourceArchive: archiveName,
    sourceArchiveSha256: archiveDigest,
    files: nativeFileManifest(runtimeRoot)
  };
  finishSupportPackage(
    nativePackageRoot,
    packageMetadata(
      spec.nativePackage, version,
      `GPUPDAL ${key} native runtime (installed automatically by gpupdal)`,
      spec, { [spec.runtimePackage]: version }),
    nativeManifest,
    `# ${spec.nativePackage}\n\nInternal native support package for \`gpupdal\`. ` +
      `Install \`gpupdal\`, not this package directly.`,
    npmSourceRoot);
  finishSupportPackage(
    runtimePackageRoot,
    packageMetadata(
      spec.runtimePackage, version,
      `GPUPDAL CUDA 13 JIT runtime for ${key} (installed automatically)`,
      spec),
    runtimeManifest,
    `# ${spec.runtimePackage}\n\nInternal CUDA runtime support package for ` +
      `\`gpupdal\`. Install \`gpupdal\`, not this package directly.`,
    npmSourceRoot);

  return { ...entry, nativePackage: spec.nativePackage,
    runtimePackage: spec.runtimePackage, payloadFiles };
}

const version = argument("--version");
const linuxArchive = path.resolve(argument("--linux-archive"));
const windowsArchive = path.resolve(argument("--windows-archive"));
const output = path.resolve(argument("--output"));
const packageRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(packageRoot, "../..");
const allowedOutputRoot = path.join(projectRoot, "dist", "npm");
const outputRoot = path.dirname(output);

if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$/.test(version) ||
    version.endsWith("-dev")) {
  throw new Error(`invalid release version: ${version}`);
}
if (output !== path.join(allowedOutputRoot, "gpupdal")) {
  throw new Error(`release launcher output must be ${allowedOutputRoot}/gpupdal`);
}
const expected = {
  "linux-x64": `gpupdal-${version}-linux-x64-cuda13.tar.gz`,
  "win32-x64": `gpupdal-${version}-win32-x64-cuda13.zip`
};
for (const [archive, name] of [
  [linuxArchive, expected["linux-x64"]],
  [windowsArchive, expected["win32-x64"]]
]) {
  if (path.basename(archive) !== name ||
      !fs.lstatSync(archive, { throwIfNoEntry: false })?.isFile()) {
    throw new Error(`expected a built archive named ${name}`);
  }
}

for (const directory of [output, ...packageNames().map((name) =>
  path.join(outputRoot, name))]) {
  fs.rmSync(directory, { recursive: true, force: true });
}
fs.mkdirSync(output, { recursive: true, mode: 0o755 });
fs.cpSync(packageRoot, output, {
  recursive: true,
  filter: (source) => {
    const relative = path.relative(packageRoot, source);
    return relative !== "native" && relative !== "test";
  }
});

const metadataPath = path.join(output, "package.json");
const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
metadata.version = version;
metadata.optionalDependencies = Object.fromEntries(
  packageNames().map((name) => [name, version]));
writeJson(metadataPath, metadata);

const platforms = {};
platforms["linux-x64"] = stagePlatform({
  key: "linux-x64",
  spec: PACKAGE_SET["linux-x64"],
  archive: linuxArchive,
  archiveName: expected["linux-x64"],
  archiveDigest: sha256(linuxArchive),
  outputRoot,
  version,
  npmSourceRoot: packageRoot
});
platforms["win32-x64"] = stagePlatform({
  key: "win32-x64",
  spec: PACKAGE_SET["win32-x64"],
  archive: windowsArchive,
  archiveName: expected["win32-x64"],
  archiveDigest: sha256(windowsArchive),
  outputRoot,
  version,
  npmSourceRoot: packageRoot
});
writeJson(path.join(output, "native-manifest.json"), {
  schema: 4,
  version,
  platforms
});

validateReleaseSet(output);
console.log(`staged gpupdal npm release set ${version} below ${outputRoot}`);
