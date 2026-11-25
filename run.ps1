param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Request
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Rust Cargo was not found. Install the Rust toolchain, then run this script again."
    exit 1
}

Write-Host "Building rapTOR..."
& cargo build --release
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$binary = Join-Path $projectRoot "target\release\raptor.exe"
if (-not (Test-Path $binary)) {
    Write-Error "Build completed but the rapTOR executable was not found at $binary."
    exit 1
}

Write-Host "`nChoose the offline simulator or local Ollama, then select or download a model."
& $binary setup
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Request -and $Request.Count -gt 0) {
    $requestText = $Request -join " "
    & $binary run $requestText --no-memory
    exit $LASTEXITCODE
}

& $binary chat
exit $LASTEXITCODE
