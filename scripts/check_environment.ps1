[CmdletBinding()]
param(
    [string]$SC2Path = $(if ($env:SC2PATH) { $env:SC2PATH } else { 'C:\linux_project\TStarBot2-dev-open\sc2_runtime' }),
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$SC2Path = [System.IO.Path]::GetFullPath($SC2Path)
$map = Join-Path $SC2Path 'Maps\Ladder2017Season4\AbyssalReefLE.SC2Map'
if (-not (Test-Path -LiteralPath $SC2Path -PathType Container)) {
    throw "SC2 runtime not found: $SC2Path"
}
if (-not (Test-Path -LiteralPath $map -PathType Leaf)) {
    throw "AbyssalReef map not found: $map"
}

$env:SC2PATH = $SC2Path
& (Join-Path $PSScriptRoot 'Test-PyTorchEnvironment.ps1') -Python $Python
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed.' }
Write-Host "SC2 runtime: $SC2Path" -ForegroundColor Green
Write-Host "SC2 map: $map" -ForegroundColor Green
