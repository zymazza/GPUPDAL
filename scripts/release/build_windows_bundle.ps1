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

$dependencies = New-Object System.Collections.Generic.List[object]
$usedCondaPackages = @{}
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
        $dependencies.Add([PSCustomObject]@{
            soname = $Name
            source_path = "cuda/bin/$Name"
            package = "NVIDIA CUDA Toolkit"
            version = "13.3"
            license_declared = "NVIDIA CUDA Toolkit EULA"
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

$searchDirectories = @($binaryDirectory, $condaBinaryDirectory, $cudaBinaryDirectory)
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
            throw "Unresolved non-system runtime dependency $name required by $file"
        }
        Copy-RuntimeDependency $dependency $name
    }
}

# NVRTC is loaded dynamically, so it will not appear in dumpbin's import table.
$nvrtc = Get-ChildItem $cudaBinaryDirectory -Filter "nvrtc64_*.dll" -File |
    Sort-Object Name
$builtins = Get-ChildItem $cudaBinaryDirectory -Filter "nvrtc-builtins64_*.dll" -File |
    Sort-Object Name
if ($nvrtc.Count -ne 1 -or $builtins.Count -ne 1) {
    throw "Expected one CUDA 13.3 NVRTC DLL and one matching builtins DLL"
}
Copy-RuntimeDependency $nvrtc[0].FullName $nvrtc[0].Name
Copy-RuntimeDependency $builtins[0].FullName $builtins[0].Name

foreach ($directory in @("gdal", "proj", "pdal")) {
    $source = Join-Path $CondaEnvironment "Library\share\$directory"
    if (Test-Path $source -PathType Container) {
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
        } else {
            $missingLicenses.Add($package.Name)
        }
    }
}
if ($missingLicenses.Count -gt 0) {
    throw "Missing license files for Conda runtime packages: $($missingLicenses -join ', ')"
}
$cudaEula = Join-Path $CudaRoot "EULA.txt"
if (-not (Test-Path $cudaEula -PathType Leaf)) { throw "CUDA EULA is missing" }
New-Item -ItemType Directory -Force -Path `
    (Join-Path $bundle "licenses\system\nvidia-cuda-toolkit") | Out-Null
Copy-Item -Force $cudaEula `
    (Join-Path $bundle "licenses\system\nvidia-cuda-toolkit\EULA.txt")

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
    "compiler=$compilerVersion",
    "release_variant=cuda13",
    "cuda_enabled=ON",
    "cuda_toolkit=13.3",
    "cuda_architectures=89",
    "cuda_driver_policy=not bundled; NVIDIA driver 580 or newer required",
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
foreach ($line in Get-Content (Join-Path $extract "SHA256SUMS")) {
    if ($line -notmatch '^([0-9a-f]{64})  \./(.+)$') { throw "Invalid checksum: $line" }
    $path = Join-Path $extract $Matches[2].Replace('/', '\')
    if (-not (Test-Path $path -PathType Leaf)) { throw "Missing checksummed file: $path" }
    $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Matches[1]) { throw "Checksum mismatch: $path" }
}

$oldPath = $env:PATH
$oldCudaPath = $env:CUDA_PATH
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
    if ($null -eq $oldCudaPath) {
        Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
    } else {
        $env:CUDA_PATH = $oldCudaPath
    }
}

Write-Output "Created $artifact"
Write-Output "Created $artifact.sha256"
Write-Output "files=$((Get-ChildItem $bundle -Recurse -File).Count)"
Write-Output "runtime_dependencies=$($dependencies.Count)"
Write-Output "sha256=$artifactHash"
