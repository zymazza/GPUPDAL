param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$SourceRevision,
    [string]$SourceRoot = "C:\gpupdal\src",
    [string]$BuildDirectory = "C:\gpupdal\build-cuda",
    [string]$CondaEnvironment = "C:\Miniforge3\envs\pdal-build",
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    [string]$OutputDirectory = "C:\gpupdal\dist"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ($Version -notmatch '^\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$') {
    throw "A semantic version is required"
}
if ($SourceRevision -notmatch '^[0-9a-f]{40}$') {
    throw "SourceRevision must be a full Git commit"
}

$artifactName = "gpupdal-$Version-win32-x64-cuda13"
$work = Join-Path ([IO.Path]::GetTempPath()) "$artifactName-work"
$bundle = Join-Path $work $artifactName
$extract = Join-Path $work "extracted"
$binaryDirectory = Join-Path $BuildDirectory "bin"
$condaBinaryDirectory = Join-Path $CondaEnvironment "Library\bin"
$cudaBinaryDirectory = Join-Path $CudaRoot "bin"
$cudaRuntimeDirectory = Join-Path $cudaBinaryDirectory "x64"
$cmakeCache = Join-Path $BuildDirectory "CMakeCache.txt"
$vcRuntimeDirectory = Get-ChildItem "C:\BuildTools\VC\Redist\MSVC" -Directory |
    Where-Object {
        $_.Name -match '^\d+\.\d+\.\d+$' -and
        (Test-Path (Join-Path $_.FullName "x64\Microsoft.VC143.CRT") `
            -PathType Container
        )
    } | Sort-Object Name -Descending | Select-Object -First 1 |
    ForEach-Object { Join-Path $_.FullName "x64\Microsoft.VC143.CRT" }
if (-not $vcRuntimeDirectory) {
    throw "The app-local Visual C++ x64 redistributable directory is missing"
}
$vcRuntimeVersion = Split-Path `
    (Split-Path (Split-Path $vcRuntimeDirectory -Parent) -Parent) -Leaf

if (-not (Test-Path $cmakeCache -PathType Leaf)) {
    throw "Configured build cache is missing: $cmakeCache"
}
function Get-CMakeCacheValue([string]$Name) {
    $escaped = [Regex]::Escape($Name)
    $line = Get-Content $cmakeCache | Where-Object {
        $_ -match "^${escaped}:[^=]+="
    } | Select-Object -Last 1
    if (-not $line) { return $null }
    return $line.Substring($line.IndexOf('=') + 1)
}
foreach ($required in @{
        WITH_PDG = "ON"
        GPUPDAL_ENABLE_CUDA = "ON"
        PDG_CUDA_ARCHITECTURES = "all"
        PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES = "ON"
    }.GetEnumerator()) {
    $actual = Get-CMakeCacheValue $required.Key
    if ($actual -ne $required.Value) {
        throw "Release build requires $($required.Key)=$($required.Value); found $actual"
    }
}
foreach ($identity in @{
        GPUPDAL_VERSION = $Version
        GPUPDAL_SOURCE_REVISION = $SourceRevision
    }.GetEnumerator()) {
    $actual = Get-CMakeCacheValue $identity.Key
    if ($actual -ne $identity.Value) {
        throw "Release identity mismatch for $($identity.Key): expected $($identity.Value), found $actual"
    }
}
$buildType = Get-CMakeCacheValue "CMAKE_BUILD_TYPE"
if ($buildType -notin @("Release", "RelWithDebInfo")) {
    throw "Windows release bundle requires an optimized build; found CMAKE_BUILD_TYPE=$buildType"
}
$enabledPlugins = @(Get-Content $cmakeCache | Where-Object {
        $_ -match '^BUILD_PLUGIN_[A-Z0-9_]+:BOOL=ON$'
    } | ForEach-Object { $_.Substring(0, $_.IndexOf(':')) })
if ($enabledPlugins.Count -gt 0) {
    throw "Windows release bundle requires optional plugins off: $($enabledPlugins -join ', ')"
}
if (Test-Path (Join-Path $SourceRoot ".git")) {
    $sourceHead = (& git -C $SourceRoot rev-parse HEAD 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0 -or $sourceHead.Trim() -ne $SourceRevision) {
        throw "SourceRoot Git revision does not match SourceRevision"
    }
    $sourceStatus = (& git -C $SourceRoot status --porcelain `
        --untracked-files=normal 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0 -or $sourceStatus) {
        throw "Refusing to package a dirty Windows source tree"
    }
}

Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $bundle, $OutputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path `
    (Join-Path $bundle "share"), `
    (Join-Path $bundle "licenses\source"), `
    (Join-Path $bundle "licenses\system") | Out-Null

$dumpbin = Get-ChildItem "C:\BuildTools\VC\Tools\MSVC" -Filter dumpbin.exe `
    -Recurse -File | Where-Object { $_.FullName -match 'Hostx64\\x64\\dumpbin\.exe$' } |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $dumpbin) { throw "64-bit dumpbin.exe was not found" }

$requiredBuildFiles = @(
    "gpupdal.exe",
    "pdg-engine.exe",
    "pdal.exe",
    "pdalcpp.dll",
    "pdg-verify.py",
    "pdg-benchmark-reference.py"
)
foreach ($name in $requiredBuildFiles) {
    $source = Join-Path $binaryDirectory $name
    if (-not (Test-Path $source -PathType Leaf)) {
        throw "Required build output is missing: $source"
    }
    Copy-Item -Force $source (Join-Path $bundle $name)
}

foreach ($name in @("LICENSE.txt", "NOTICE", "ORIGIN.md", "THIRD_PARTY_LICENSES.md")) {
    $source = Join-Path $SourceRoot $name
    if (-not (Test-Path $source -PathType Leaf)) { throw "Missing $source" }
    Copy-Item -Force $source (Join-Path $bundle $name)
}
Copy-Item -Recurse -Force (Join-Path $SourceRoot "licenses\*") `
    (Join-Path $bundle "licenses\source")

$condaOwners = @{}
$condaPackages = @{}
Get-ChildItem (Join-Path $CondaEnvironment "conda-meta") -Filter *.json -File |
    Sort-Object Name | ForEach-Object {
        $metadata = Get-Content $_.FullName -Raw | ConvertFrom-Json
        $package = [PSCustomObject]@{
            Name = [string]$metadata.name
            Version = [string]$metadata.version
            License = [string]$metadata.license
            Source = [string]$metadata.link.source
        }
        $condaPackages[$package.Name] = $package
        foreach ($relative in $metadata.files) {
            $key = ([string]$relative).Replace('/', '\').ToLowerInvariant()
            $condaOwners[$key] = $package
        }
    }

function Get-CondaPackageForPath([string]$Path) {
    $prefix = [IO.Path]::GetFullPath($CondaEnvironment).TrimEnd('\') + '\'
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }
    $relative = $full.Substring($prefix.Length).ToLowerInvariant()
    return $condaOwners[$relative]
}

function Add-CondaDataDependency([string]$Path, [string]$Kind) {
    $package = Get-CondaPackageForPath $Path
    if (-not $package) {
        throw "No Conda package owns bundled $Kind file: $Path"
    }
    $usedCondaPackages[$package.Name] = $package
    $key = "$Kind|$($package.Name)"
    if ($recordedDataPackages.ContainsKey($key)) { return }
    $recordedDataPackages[$key] = $true
    $relative = [IO.Path]::GetFullPath($Path).Substring(
        [IO.Path]::GetFullPath($CondaEnvironment).TrimEnd('\').Length + 1)
    $dependencies.Add([PSCustomObject]@{
        soname = "[$Kind]"
        source_path = "conda/" + $relative.Replace('\', '/')
        package = $package.Name
        version = $package.Version
        license_declared = $package.License
    })
}

$dependencies = New-Object System.Collections.Generic.List[object]
$usedCondaPackages = @{}
$recordedDataPackages = @{}
function Add-DependencyRecord([string]$Name, [string]$Source) {
    $package = Get-CondaPackageForPath $Source
    if ($package) {
        $usedCondaPackages[$package.Name] = $package
        $relative = [IO.Path]::GetFullPath($Source).Substring(
            [IO.Path]::GetFullPath($CondaEnvironment).TrimEnd('\').Length + 1)
        $dependencies.Add([PSCustomObject]@{
            soname = $Name
            source_path = "conda/" + $relative.Replace('\', '/')
            package = $package.Name
            version = $package.Version
            license_declared = $package.License
        })
        return
    }
    if ([IO.Path]::GetFullPath($Source).StartsWith(
            [IO.Path]::GetFullPath($CudaRoot),
            [StringComparison]::OrdinalIgnoreCase)) {
        $relative = [IO.Path]::GetFullPath($Source).Substring(
            [IO.Path]::GetFullPath($CudaRoot).TrimEnd('\').Length + 1)
        $dependencies.Add([PSCustomObject]@{
            soname = $Name
            source_path = "cuda/" + $relative.Replace('\', '/')
            package = "NVIDIA CUDA Toolkit"
            version = "13.3"
            license_declared = "NVIDIA CUDA Toolkit EULA"
        })
        return
    }
    if ([IO.Path]::GetFullPath($Source).StartsWith(
            [IO.Path]::GetFullPath($vcRuntimeDirectory),
            [StringComparison]::OrdinalIgnoreCase)) {
        $dependencies.Add([PSCustomObject]@{
            soname = $Name
            source_path = "visual-cpp-runtime/$Name"
            package = "Microsoft Visual C++ Runtime"
            version = $vcRuntimeVersion
            license_declared = "LicenseRef-Microsoft-Visual-Cpp-Runtime"
        })
        return
    }
    $dependencies.Add([PSCustomObject]@{
        soname = $Name
        source_path = "build-output/bin/$Name"
        package = "GPUPDAL"
        version = $Version
        license_declared = "BSD-3-Clause"
    })
}

$searchDirectories = @(
    $binaryDirectory,
    $condaBinaryDirectory,
    $cudaRuntimeDirectory,
    $cudaBinaryDirectory,
    $vcRuntimeDirectory
) | Where-Object { Test-Path $_ -PathType Container }
$systemDll = '^(?i)(api-ms-win-|ext-ms-win-|kernel32|user32|advapi32|bcrypt|crypt32|dnsapi|gdi32|iphlpapi|mpr|netapi32|normaliz|nsi|ole32|oleaut32|powrprof|psapi|rpcrt4|secur32|shell32|shlwapi|userenv|version|winhttp|winmm|wintrust|ws2_32|wtsapi32|ntdll|nvcuda)\.dll$'
$queue = [System.Collections.Generic.Queue[string]]::new()
$seen = @{}
$copied = @{}

foreach ($name in @("gpupdal.exe", "pdg-engine.exe", "pdal.exe", "pdalcpp.dll")) {
    $source = Join-Path $binaryDirectory $name
    $queue.Enqueue($source)
    $copied[$name.ToLowerInvariant()] = $source
    Add-DependencyRecord $name $source
}

$cuobjdump = Join-Path $cudaBinaryDirectory "cuobjdump.exe"
if (-not (Test-Path $cuobjdump -PathType Leaf)) {
    throw "CUDA release verification requires cuobjdump.exe"
}
$engine = Join-Path $binaryDirectory "pdg-engine.exe"
$cubinText = (& $cuobjdump --list-elf $engine 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cuobjdump --list-elf failed for $engine" }
$ptxText = (& $cuobjdump --list-ptx $engine 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) { throw "cuobjdump --list-ptx failed for $engine" }
$actualCubins = @([Regex]::Matches($cubinText, '\.sm_([0-9]+)\.cubin') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object { [int]$_ } -Unique)
$actualPtx = @([Regex]::Matches($ptxText, '\.sm_([0-9]+)\.ptx') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object { [int]$_ } -Unique)
$expectedCubins = @("75", "80", "86", "87", "88", "89", "90", "100", "103", "110", "120", "121")
if ((Compare-Object $actualCubins $expectedCubins) -or
        (Compare-Object $actualPtx @("120"))) {
    throw "CUDA 13.3 fatbin image mismatch: cubins=$($actualCubins -join ','), ptx=$($actualPtx -join ',')"
}

function Find-Dependency([string]$Name) {
    foreach ($directory in $searchDirectories) {
        $candidate = Join-Path $directory $Name
        if (Test-Path $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Copy-RuntimeDependency([string]$Source, [string]$Name) {
    $key = $Name.ToLowerInvariant()
    if (-not $copied.ContainsKey($key)) {
        Copy-Item -Force $Source (Join-Path $bundle $Name)
        $copied[$key] = $Source
        Add-DependencyRecord $Name $Source
    }
    $queue.Enqueue($Source)
}

# NVRTC is loaded dynamically, so it will not appear in dumpbin's import table.
$nvrtc = @(Get-ChildItem $cudaRuntimeDirectory -Filter "nvrtc64_*.dll" -File |
    Sort-Object Name)
$builtins = @(Get-ChildItem $cudaRuntimeDirectory `
    -Filter "nvrtc-builtins64_*.dll" -File |
    Sort-Object Name)
if ($nvrtc.Count -ne 1 -or $builtins.Count -ne 1) {
    throw "Expected one CUDA 13.3 NVRTC DLL and one matching builtins DLL"
}
Copy-RuntimeDependency $nvrtc[0].FullName $nvrtc[0].Name
Copy-RuntimeDependency $builtins[0].FullName $builtins[0].Name

while ($queue.Count -gt 0) {
    $file = $queue.Dequeue()
    $identity = [IO.Path]::GetFullPath($file).ToLowerInvariant()
    if ($seen.ContainsKey($identity)) { continue }
    $seen[$identity] = $true
    $output = & $dumpbin /DEPENDENTS $file 2>&1
    if ($LASTEXITCODE -ne 0) { throw "dumpbin failed for $file" }
    foreach ($line in $output) {
        if ($line -notmatch '^\s+([A-Za-z0-9_.+\-]+\.dll)\s*$') { continue }
        $name = $Matches[1]
        if ($name -match $systemDll) { continue }
        $dependency = Find-Dependency $name
        if (-not $dependency) {
            $systemPath = Join-Path $env:SystemRoot "System32\$name"
            if (Test-Path $systemPath -PathType Leaf) { continue }
            throw "Unresolved non-system runtime dependency $name required by $file"
        }
        Copy-RuntimeDependency $dependency $name
    }
}

foreach ($directory in @("gdal", "proj", "pdal")) {
    $source = Join-Path $CondaEnvironment "Library\share\$directory"
    if (Test-Path $source -PathType Container) {
        Get-ChildItem $source -Recurse -File | ForEach-Object {
            Add-CondaDataDependency $_.FullName "data:$directory"
        }
        Copy-Item -Recurse -Force $source (Join-Path $bundle "share\$directory")
    }
}
$certificateCandidates = @(
    (Join-Path $CondaEnvironment "Library\ssl\cacert.pem"),
    (Join-Path $CondaEnvironment "Library\etc\ssl\certs\ca-bundle.crt"),
    (Join-Path $CondaEnvironment "Library\bin\curl-ca-bundle.crt")
)
$certificate = $certificateCandidates | Where-Object { Test-Path $_ -PathType Leaf } |
    Select-Object -First 1
if ($certificate) {
    Add-CondaDataDependency $certificate "ca-certificates"
    New-Item -ItemType Directory -Force -Path (Join-Path $bundle "share\certs") |
        Out-Null
    Copy-Item -Force $certificate (Join-Path $bundle "share\certs\cacert.pem")
}

$missingLicenses = New-Object System.Collections.Generic.List[string]
foreach ($package in ($usedCondaPackages.Values | Sort-Object Name)) {
    $destination = Join-Path $bundle ("licenses\system\" + $package.Name)
    $licenseRoot = Join-Path $package.Source "info\licenses"
    if (Test-Path $licenseRoot -PathType Container) {
        Copy-Item -Recurse -Force $licenseRoot $destination
    } else {
        $licenseFiles = Get-ChildItem (Join-Path $package.Source "info") `
            -Filter "*license*" -File -ErrorAction SilentlyContinue
        if ($licenseFiles.Count -gt 0) {
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
            $licenseFiles | ForEach-Object { Copy-Item -Force $_.FullName $destination }
        } elseif ($package.Name -eq "libsqlite" -and
                $package.License -eq "blessing") {
            $sqliteNotice = Join-Path $SourceRoot `
                "licenses\SQLite-Public-Domain-Notice.txt"
            if (-not (Test-Path $sqliteNotice -PathType Leaf)) {
                throw "Pinned SQLite public-domain notice is missing"
            }
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
            Copy-Item -Force $sqliteNotice `
                (Join-Path $destination "PUBLIC-DOMAIN-NOTICE.txt")
        } else {
            $missingLicenses.Add($package.Name)
        }
    }
}
if ($missingLicenses.Count -gt 0) {
    throw "Missing license files for Conda runtime packages: $($missingLicenses -join ', ')"
}
$cudaEula = @(
    (Join-Path $CudaRoot "EULA.txt"),
    (Join-Path $CudaRoot "LICENSE")
) | Where-Object { Test-Path $_ -PathType Leaf } | Select-Object -First 1
if (-not $cudaEula) { throw "CUDA EULA is missing" }
$cudaEulaHash = (Get-FileHash $cudaEula -Algorithm SHA256).Hash.ToLowerInvariant()
if ($cudaEulaHash -ne
        "088381bc2d891e719a2a9398645b00bb45f3b24231473a8283ac7e3e66b8a028") {
    throw "CUDA 13.3 EULA hash mismatch: $cudaEulaHash"
}
New-Item -ItemType Directory -Force -Path `
    (Join-Path $bundle "licenses\system\nvidia-cuda-toolkit") | Out-Null
Copy-Item -Force $cudaEula `
    (Join-Path $bundle "licenses\system\nvidia-cuda-toolkit\EULA.txt")
$ccclLicense = Join-Path $SourceRoot "licenses\CCCL-3.4.0.txt"
if (-not (Test-Path $ccclLicense -PathType Leaf)) {
    throw "Pinned CCCL 3.4.0 combined license is missing"
}
$ccclLicenseHash = (Get-FileHash $ccclLicense -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ccclLicenseHash -ne
        "f96f51edda77fb9897de29d924d615e6bd153f4969db85fc6b7a22d7a624631e") {
    throw "Pinned CCCL 3.4.0 combined license hash mismatch"
}
New-Item -ItemType Directory -Force -Path `
    (Join-Path $bundle "licenses\system\nvidia-cccl") | Out-Null
Copy-Item -Force $ccclLicense `
    (Join-Path $bundle "licenses\system\nvidia-cccl\LICENSE")
$dependencies.Add([PSCustomObject]@{
    soname = "[compiled-header-library]"
    source_path = "cuda/include/cccl"
    package = "NVIDIA CCCL"
    version = "3.4.0"
    license_declared = "NOASSERTION (combined upstream license included)"
})
$vcLicenseDirectory = Join-Path $bundle "licenses\system\microsoft-vc-runtime"
New-Item -ItemType Directory -Force -Path $vcLicenseDirectory | Out-Null
@(
    "Microsoft Visual C++ Runtime",
    "Redistributable binaries copied from the licensed Visual Studio Build Tools installation.",
    "Version: $vcRuntimeVersion",
    "Upstream licensing information: https://visualstudio.microsoft.com/license-terms/"
) | Set-Content -Encoding UTF8 (Join-Path $vcLicenseDirectory "README.txt")

$dependencyMap = Join-Path $bundle "RUNTIME_DEPENDENCIES.tsv"
"soname`tsource_path`tpackage`tversion`tlicense_declared" |
    Set-Content -Encoding UTF8 $dependencyMap
$dependencies | Sort-Object soname, package -Unique | ForEach-Object {
    "$($_.soname)`t$($_.source_path)`t$($_.package)`t$($_.version)`t$($_.license_declared)" |
        Add-Content -Encoding UTF8 $dependencyMap
}

$compilerVersion = (& $dumpbin /? 2>&1 | Select-Object -First 1)
@(
    "release_baseline=Windows Server 2022 / Visual Studio 2022 / CUDA 13.3",
    "cmake_build_type=$buildType",
    "compiler=$compilerVersion",
    "release_variant=cuda13",
    "cuda_enabled=ON",
    "cuda_toolkit=13.3",
    "cuda_eula_sha256=$cudaEulaHash",
    "cccl=3.4.0",
    "cccl_license_sha256=$ccclLicenseHash",
    "cuda_architectures=all",
    "cuda_cubins=$($actualCubins -join ',')",
    "cuda_ptx=$($actualPtx -join ',')",
    "cuda_driver_policy=not bundled; NVIDIA driver 580 or newer required",
    "embedded_path_policy=canonical C:\gpupdal source/build roots may appear in optimized debug records; personal and noncanonical roots are rejected",
    "physically_qualified_profile=SM 89; NVIDIA L4; qualification report required",
    "source_revision=$SourceRevision"
) | Set-Content -Encoding UTF8 (Join-Path $bundle "BUILD-ENVIRONMENT.txt")

$python = Join-Path $CondaEnvironment "python.exe"
if (-not (Test-Path $python -PathType Leaf)) { throw "Conda Python is missing" }
$createdEpoch = [DateTimeOffset]::Parse("2026-08-22T00:00:00Z").ToUnixTimeSeconds()
& $python (Join-Path $SourceRoot "scripts\release\generate_spdx_sbom.py") `
    --root $bundle --version $Version --artifact-name $artifactName `
    --commit $SourceRevision --created-epoch $createdEpoch `
    --output (Join-Path $bundle "SBOM.spdx.json")
if ($LASTEXITCODE -ne 0) { throw "SPDX SBOM generation failed" }

$canonicalSourceRoot = [IO.Path]::GetFullPath("C:\gpupdal\src").TrimEnd('\')
$canonicalBuildDirectory = `
    [IO.Path]::GetFullPath("C:\gpupdal\build-cuda").TrimEnd('\')
$actualSourceRoot = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$actualBuildDirectory = [IO.Path]::GetFullPath($BuildDirectory).TrimEnd('\')
$sensitivePaths = @($CondaEnvironment, "C:\Users\")
if ($actualSourceRoot -ne $canonicalSourceRoot) {
    $sensitivePaths += $actualSourceRoot
}
if ($actualBuildDirectory -ne $canonicalBuildDirectory) {
    $sensitivePaths += $actualBuildDirectory
}
# RelWithDebInfo PE/CodeView records may contain the deliberately neutral
# C:\gpupdal roots. They carry no user identity and the two clean-machine gates
# prove they are not runtime dependencies. Any personal or noncanonical path
# still fails closed.
foreach ($file in Get-ChildItem $bundle -Recurse -File) {
    $bytes = [IO.File]::ReadAllBytes($file.FullName)
    $text = [Text.Encoding]::ASCII.GetString($bytes)
    foreach ($sensitivePath in $sensitivePaths) {
        if ($text.IndexOf($sensitivePath,
                [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            throw "Bundle embeds private path $sensitivePath in $($file.FullName)"
        }
    }
}

$checksumPath = Join-Path $bundle "SHA256SUMS"
$checksumLines = Get-ChildItem $bundle -Recurse -File |
    Where-Object { $_.FullName -ne $checksumPath } |
    ForEach-Object {
        $relative = $_.FullName.Substring($bundle.Length + 1).Replace('\', '/')
        $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  ./$relative"
    } | Sort-Object
$checksumLines | Set-Content -Encoding ASCII $checksumPath

$artifact = Join-Path $OutputDirectory "$artifactName.zip"
Remove-Item -Force $artifact -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $bundle "*") -DestinationPath $artifact `
    -CompressionLevel Optimal
$artifactHash = (Get-FileHash $artifact -Algorithm SHA256).Hash.ToLowerInvariant()
"$artifactHash  $artifactName.zip" | Set-Content -Encoding ASCII "$artifact.sha256"

New-Item -ItemType Directory -Force -Path $extract | Out-Null
Expand-Archive -Path $artifact -DestinationPath $extract -Force
$listedFiles = [System.Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase)
$extractedSums = Join-Path $extract "SHA256SUMS"
foreach ($line in Get-Content $extractedSums) {
    if ($line -notmatch '^([0-9a-f]{64})  \./(.+)$') { throw "Invalid checksum: $line" }
    $path = Join-Path $extract $Matches[2].Replace('/', '\')
    if (-not (Test-Path $path -PathType Leaf)) { throw "Missing checksummed file: $path" }
    if (-not $listedFiles.Add($Matches[2].Replace('\', '/'))) {
        throw "Duplicate checksum entry: $($Matches[2])"
    }
    $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Matches[1]) { throw "Checksum mismatch: $path" }
}
$archiveFiles = @(Get-ChildItem $extract -Recurse -File | Where-Object {
        $_.FullName -ne $extractedSums
    } | ForEach-Object {
        $_.FullName.Substring($extract.Length + 1).Replace('\', '/')
    })
if ($archiveFiles.Count -ne $listedFiles.Count) {
    throw "Archive contains unchecksummed or phantom files"
}
foreach ($relative in $archiveFiles) {
    if (-not $listedFiles.Contains($relative)) {
        throw "Archive file is not checksummed: $relative"
    }
}

$controlledEnvironmentPattern = `
    '^(PDG_|PDAL_|GPUPDAL_|GDAL_|PROJ_|CONDA|_CE_CONDA|CUDA_|CUDACXX$|CMAKE_|CC$|CXX$|VCPKG_|HOME$|USERPROFILE$|APPDATA$|LOCALAPPDATA$|SSL_CERT_FILE$|CURL_CA_BUNDLE$)'
$savedEnvironment = @{}
Get-ChildItem Env: | Where-Object {
    $_.Name -match $controlledEnvironmentPattern
} | ForEach-Object {
    $savedEnvironment[$_.Name] = $_.Value
    Remove-Item "Env:$($_.Name)"
}
$temporaryProfile = Join-Path $work "profile"
$temporaryAppData = Join-Path $temporaryProfile "AppData\Roaming"
$temporaryLocalData = Join-Path $temporaryProfile "AppData\Local"
New-Item -ItemType Directory -Force -Path $temporaryProfile,
    $temporaryAppData, $temporaryLocalData | Out-Null
$env:HOME = $temporaryProfile
$env:USERPROFILE = $temporaryProfile
$env:APPDATA = $temporaryAppData
$env:LOCALAPPDATA = $temporaryLocalData
$oldPath = $env:PATH
try {
    $env:PATH = "$extract;$env:SystemRoot\System32;$env:SystemRoot"
    Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
    & (Join-Path $extract "gpupdal.exe") --version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Extracted gpupdal --version failed" }
    & (Join-Path $extract "gpupdal.exe") --drivers |
        Set-Content -Encoding UTF8 (Join-Path $work "gpupdal-drivers.txt")
    if ($LASTEXITCODE -ne 0) { throw "Extracted gpupdal --drivers failed" }
    & (Join-Path $extract "pdal.exe") --drivers |
        Set-Content -Encoding UTF8 (Join-Path $work "pdal-drivers.txt")
    if ($LASTEXITCODE -ne 0) { throw "Extracted pdal --drivers failed" }
    $difference = Compare-Object `
        (Get-Content (Join-Path $work "gpupdal-drivers.txt")) `
        (Get-Content (Join-Path $work "pdal-drivers.txt"))
    if ($difference) { throw "Extracted gpupdal/pdal driver catalogs differ" }
} finally {
    $env:PATH = $oldPath
    Get-ChildItem Env: | Where-Object {
        $_.Name -match $controlledEnvironmentPattern
    } | ForEach-Object {
        Remove-Item "Env:$($_.Name)"
    }
    foreach ($name in $savedEnvironment.Keys) {
        Set-Item "Env:$name" $savedEnvironment[$name]
    }
}

Write-Output "Created $artifact"
Write-Output "Created $artifact.sha256"
Write-Output "files=$((Get-ChildItem $bundle -Recurse -File).Count)"
Write-Output "runtime_dependencies=$($dependencies.Count)"
Write-Output "sha256=$artifactHash"
