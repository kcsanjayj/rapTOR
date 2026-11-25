$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$binary = Join-Path $projectRoot "target\release\raptor.exe"

if (-not (Test-Path $binary)) {
    Write-Error "Build rapTOR first with 'cargo build --release'. Expected binary: $binary"
    exit 1
}

$request = "Compare Rust and Go for a CLI tool"

Write-Host "`n1. Deterministic multi-step run with visible parallel work"
& $binary run $request --sim --latency-ms 100 --no-memory
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n2. Injected transient failures and retry recovery"
& $binary run $request --sim --latency-ms 0 --chaos-transient 1.0 --seed 3 --no-memory
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n3. Weak first drafts caught and repaired by the critic"
& $binary run $request --sim --latency-ms 0 --chaos-degrade 1.0 --seed 3 --no-memory
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
