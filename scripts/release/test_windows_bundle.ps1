param(
    [Parameter(Mandatory = $true)]
    [string]$BundlePath,
    [Parameter(Mandatory = $true)]
    [string]$FixtureRoot,
    [string]$NpmTarball,
    [string]$NpmCommand,
    [switch]$RequireCuda,
    [switch]$RequireNoDriver
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not (Test-Path $BundlePath -PathType Leaf)) {
    throw "Missing file: $BundlePath"
}
if (-not (Test-Path $FixtureRoot -PathType Container)) {
    throw "Missing fixture root: $FixtureRoot"
}
if ($RequireCuda -and $RequireNoDriver) {
    throw "RequireCuda and RequireNoDriver are mutually exclusive"
}

$work = Join-Path ([IO.Path]::GetTempPath()) `
    ("gpupdal-windows-package-test-" + [Guid]::NewGuid().ToString("N"))
$bundle = Join-Path $work "bundle"
$reports = Join-Path $work "reports"
New-Item -ItemType Directory -Force -Path $bundle, $reports | Out-Null
Expand-Archive -Path $BundlePath -DestinationPath $bundle -Force

$sums = Join-Path $bundle "SHA256SUMS"
if (-not (Test-Path $sums -PathType Leaf)) { throw "SHA256SUMS is missing" }
$listedFiles = [System.Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase)
foreach ($line in Get-Content $sums) {
    if ($line -notmatch '^([0-9a-f]{64})  \./(.+)$') {
        throw "Invalid checksum line: $line"
    }
    $filename = Join-Path $bundle $Matches[2].Replace('/', '\')
    if (-not (Test-Path $filename -PathType Leaf)) {
        throw "Checksummed file is missing: $filename"
    }
    if (-not $listedFiles.Add($Matches[2].Replace('\', '/'))) {
        throw "Duplicate checksum entry: $($Matches[2])"
    }
    $actual = (Get-FileHash $filename -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Matches[1]) { throw "Checksum mismatch: $filename" }
}
$archiveFiles = @(Get-ChildItem $bundle -Recurse -File | Where-Object {
        $_.FullName -ne $sums
    } | ForEach-Object {
        $_.FullName.Substring($bundle.Length + 1).Replace('\', '/')
    })
if ($archiveFiles.Count -ne $listedFiles.Count) {
    throw "Archive contains unchecksummed or phantom files"
}
foreach ($relative in $archiveFiles) {
    if (-not $listedFiles.Contains($relative)) {
        throw "Archive file is not checksummed: $relative"
    }
}

$forbidden = @("cl.exe", "conda.exe", "nvcc.exe", "cmake.exe")
foreach ($program in $forbidden) {
    if (Get-Command $program -ErrorAction SilentlyContinue) {
        throw "Clean-machine PATH unexpectedly contains $program"
    }
}
if ($RequireNoDriver -and (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw "Driverless machine unexpectedly contains nvidia-smi.exe"
}
if ($RequireNoDriver -and
        (Test-Path (Join-Path $env:SystemRoot "System32\nvcuda.dll") -PathType Leaf)) {
    throw "Driverless machine unexpectedly contains nvcuda.dll"
}

$gpupdal = Join-Path $bundle "gpupdal.exe"
$pdal = Join-Path $bundle "pdal.exe"
$fixture = Join-Path $FixtureRoot "simple.las"
$pipeline = Join-Path $FixtureRoot "assign-ferry-fused.json"
$reprojectionPipeline = Join-Path $FixtureRoot "reprojection-fallback.json"
foreach ($path in @($gpupdal, $pdal, $fixture, $pipeline,
        $reprojectionPipeline)) {
    if (-not (Test-Path $path -PathType Leaf)) { throw "Missing test input: $path" }
}

function Invoke-ExactRole(
    [string]$Role,
    [string]$Executable,
    [bool]$ForceCuda,
    [string]$PipelineSource = $pipeline,
    [bool]$UseBundledData = $false
) {
    $directory = Join-Path $reports $Role
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    Copy-Item -Force $fixture (Join-Path $directory "input.las")
    Copy-Item -Force $PipelineSource (Join-Path $directory "pipeline.json")
    $stdout = Join-Path $directory "stdout.bin"
    $stderr = Join-Path $directory "stderr.bin"
    if ($ForceCuda) {
        $env:PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM = "1"
    } else {
        Remove-Item Env:PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM `
            -ErrorAction SilentlyContinue
    }
    foreach ($name in @("GDAL_DATA", "PROJ_DATA", "CURL_CA_BUNDLE",
            "SSL_CERT_FILE")) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    if ($UseBundledData) {
        $env:GDAL_DATA = Join-Path $bundle "share\gdal"
        $env:PROJ_DATA = Join-Path $bundle "share\proj"
        $certificate = Join-Path $bundle "share\certs\cacert.pem"
        if (Test-Path $certificate -PathType Leaf) {
            $env:CURL_CA_BUNDLE = $certificate
            $env:SSL_CERT_FILE = $certificate
        }
    }
    $process = Start-Process -FilePath $Executable -WorkingDirectory $directory `
        -ArgumentList @("pipeline", "pipeline.json") -NoNewWindow -Wait `
        -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Remove-Item Env:PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM -ErrorAction SilentlyContinue
    foreach ($name in @("GDAL_DATA", "PROJ_DATA", "CURL_CA_BUNDLE",
            "SSL_CERT_FILE")) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    return [PSCustomObject]@{ Directory = $directory; ExitCode = $process.ExitCode }
}

function Get-ExactInventory([string]$Directory) {
    $inventory = @{}
    Get-ChildItem $Directory -Recurse -File | ForEach-Object {
        $relative = $_.FullName.Substring($Directory.Length + 1).Replace('\', '/')
        $inventory[$relative] = [PSCustomObject]@{
            Bytes = $_.Length
            Sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    return $inventory
}

$controlledEnvironmentPattern = `
    '^(PDG_|PDAL_|GPUPDAL_|GDAL_|PROJ_|CONDA|_CE_CONDA|CUDA_|CUDACXX$|CMAKE_|CC$|CXX$|VCPKG_|HOME$|USERPROFILE$|APPDATA$|LOCALAPPDATA$|SSL_CERT_FILE$|CURL_CA_BUNDLE$|NPM_CONFIG_)'
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
$env:NPM_CONFIG_CACHE = Join-Path $work "npm-cache"
$priorPath = $env:PATH
try {
    $env:PATH = "$bundle;$env:SystemRoot\System32;$env:SystemRoot"
    Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue

    & $gpupdal --version
    if ($LASTEXITCODE -ne 0) { throw "gpupdal --version failed" }
    $gpupdalDrivers = & $gpupdal --drivers
    if ($LASTEXITCODE -ne 0) { throw "gpupdal --drivers failed" }
    $pdalDrivers = & $pdal --drivers
    if ($LASTEXITCODE -ne 0) { throw "pdal --drivers failed" }
    if (Compare-Object $gpupdalDrivers $pdalDrivers) {
        throw "gpupdal and bundled pdal driver catalogs differ"
    }

    & $gpupdal doctor
    $doctorStatus = $LASTEXITCODE
    if ($RequireCuda -and $doctorStatus -ne 0) {
        throw "gpupdal doctor did not find the required CUDA device"
    }
    if ($RequireNoDriver -and $doctorStatus -eq 0) {
        throw "gpupdal doctor unexpectedly found CUDA on the driverless host"
    }

    $oracleRun = Invoke-ExactRole "oracle" $pdal $false
    $candidateRun = Invoke-ExactRole "candidate" $gpupdal ([bool]$RequireCuda)
    if ($oracleRun.ExitCode -ne $candidateRun.ExitCode) {
        throw "Extracted package exit status differs: oracle=$($oracleRun.ExitCode), candidate=$($candidateRun.ExitCode)"
    }
    $oracleInventory = Get-ExactInventory $oracleRun.Directory
    $candidateInventory = Get-ExactInventory $candidateRun.Directory
    $oracleNames = @($oracleInventory.Keys | Sort-Object)
    $candidateNames = @($candidateInventory.Keys | Sort-Object)
    if (Compare-Object $oracleNames $candidateNames) {
        throw "Extracted package artifact names differ"
    }
    foreach ($name in $oracleNames) {
        $expected = $oracleInventory[$name]
        $actual = $candidateInventory[$name]
        if ($expected.Bytes -ne $actual.Bytes -or
            $expected.Sha256 -ne $actual.Sha256) {
            throw "Extracted package artifact differs byte-for-byte: $name"
        }
    }

    # The direct driverless route must initialize the bundle before spawning
    # pdal.exe. The oracle receives explicit data paths; the candidate starts
    # with all GDAL/PROJ controls absent and must discover its sibling data.
    $reprojectionOracle = Invoke-ExactRole "reprojection-oracle" $pdal $false `
        $reprojectionPipeline $true
    $reprojectionCandidate = Invoke-ExactRole "reprojection-candidate" `
        $gpupdal $false $reprojectionPipeline $false
    if ($reprojectionOracle.ExitCode -ne $reprojectionCandidate.ExitCode -or
            $reprojectionOracle.ExitCode -ne 0) {
        throw "Bundled reprojection fallback status differs"
    }
    $reprojectionOracleInventory = Get-ExactInventory `
        $reprojectionOracle.Directory
    $reprojectionCandidateInventory = Get-ExactInventory `
        $reprojectionCandidate.Directory
    $reprojectionNames = @($reprojectionOracleInventory.Keys | Sort-Object)
    if (Compare-Object $reprojectionNames `
            @($reprojectionCandidateInventory.Keys | Sort-Object)) {
        throw "Bundled reprojection fallback artifact names differ"
    }
    foreach ($name in $reprojectionNames) {
        $expected = $reprojectionOracleInventory[$name]
        $actual = $reprojectionCandidateInventory[$name]
        if ($expected.Bytes -ne $actual.Bytes -or
            $expected.Sha256 -ne $actual.Sha256) {
            throw "Bundled reprojection fallback differs byte-for-byte: $name"
        }
    }

    if ($NpmTarball) {
        if (-not (Test-Path $NpmTarball -PathType Leaf)) {
            throw "Missing npm tarball: $NpmTarball"
        }
        if (-not $NpmCommand -or -not (Test-Path $NpmCommand -PathType Leaf)) {
            throw "A valid NpmCommand is required with NpmTarball"
        }
        $nodeDirectory = Split-Path $NpmCommand -Parent
        if (Test-Path (Join-Path $nodeDirectory "node.exe") -PathType Leaf) {
            $env:PATH = "$nodeDirectory;$env:PATH"
        }
        $project = Join-Path $work "npm-project"
        New-Item -ItemType Directory -Force -Path $project | Out-Null
        '{"name":"gpupdal-clean-install","private":true,"version":"0.0.0"}' |
            Set-Content -Encoding ASCII (Join-Path $project "package.json")
        Push-Location $project
        try {
            & $NpmCommand install --ignore-scripts --no-audit --no-fund $NpmTarball
            if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
            $shim = Join-Path $project "node_modules\.bin\gpupdal.cmd"
            if (-not (Test-Path $shim -PathType Leaf)) { throw "npm shim is missing" }
            & $shim --version
            if ($LASTEXITCODE -ne 0) { throw "npm gpupdal shim failed" }
            $installed = Join-Path $project `
                "node_modules\gpupdal\native\win32-x64\gpupdal.exe"
            if (-not (Test-Path $installed -PathType Leaf)) {
                throw "npm Windows payload is missing"
            }
            $npmRun = Invoke-ExactRole "npm-candidate" $shim `
                ([bool]$RequireCuda)
            if ($npmRun.ExitCode -ne $oracleRun.ExitCode) {
                throw "npm candidate exit status differs from bundled pdal"
            }
            $npmInventory = Get-ExactInventory $npmRun.Directory
            $npmNames = @($npmInventory.Keys | Sort-Object)
            if (Compare-Object $oracleNames $npmNames) {
                throw "npm candidate artifact names differ from bundled pdal"
            }
            foreach ($name in $oracleNames) {
                $expected = $oracleInventory[$name]
                $actual = $npmInventory[$name]
                if ($expected.Bytes -ne $actual.Bytes -or
                    $expected.Sha256 -ne $actual.Sha256) {
                    throw "npm candidate artifact differs byte-for-byte: $name"
                }
            }
            & $NpmCommand uninstall --ignore-scripts --no-audit --no-fund gpupdal
            if ($LASTEXITCODE -ne 0) { throw "npm uninstall failed" }
            if ((Test-Path $shim) -or (Test-Path (Join-Path $project "node_modules\gpupdal"))) {
                throw "npm uninstall left the GPUPDAL command or payload behind"
            }

            $globalPrefix = Join-Path $work "npm-global"
            & $NpmCommand install --global --prefix $globalPrefix `
                --ignore-scripts --no-audit --no-fund $NpmTarball
            if ($LASTEXITCODE -ne 0) { throw "global npm install failed" }
            $globalShim = Join-Path $globalPrefix "gpupdal.cmd"
            if (-not (Test-Path $globalShim -PathType Leaf)) {
                throw "global npm gpupdal shim is missing"
            }
            $globalRun = Invoke-ExactRole "npm-global-candidate" $globalShim `
                ([bool]$RequireCuda)
            if ($globalRun.ExitCode -ne $oracleRun.ExitCode) {
                throw "global npm candidate exit status differs from bundled pdal"
            }
            $globalInventory = Get-ExactInventory $globalRun.Directory
            if (Compare-Object $oracleNames `
                    @($globalInventory.Keys | Sort-Object)) {
                throw "global npm candidate artifact names differ from bundled pdal"
            }
            foreach ($name in $oracleNames) {
                $expected = $oracleInventory[$name]
                $actual = $globalInventory[$name]
                if ($expected.Bytes -ne $actual.Bytes -or
                    $expected.Sha256 -ne $actual.Sha256) {
                    throw "global npm candidate differs byte-for-byte: $name"
                }
            }
            & $NpmCommand uninstall --global --prefix $globalPrefix `
                --ignore-scripts --no-audit --no-fund gpupdal
            if ($LASTEXITCODE -ne 0) { throw "global npm uninstall failed" }
            if ((Test-Path $globalShim) -or
                    (Test-Path (Join-Path $globalPrefix "node_modules\gpupdal"))) {
                throw "global npm uninstall left the command or payload behind"
            }
        } finally {
            Pop-Location
        }
    }
} finally {
    $env:PATH = $priorPath
    Get-ChildItem Env: | Where-Object {
        $_.Name -match $controlledEnvironmentPattern
    } | ForEach-Object {
        Remove-Item "Env:$($_.Name)"
    }
    foreach ($name in $savedEnvironment.Keys) {
        Set-Item "Env:$name" $savedEnvironment[$name]
    }
}

Write-Output "Windows package qualification passed: $BundlePath"
