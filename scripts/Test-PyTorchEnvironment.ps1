[CmdletBinding()]
param(
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$Python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }

& $Python -m pytest -q -m 'not sc2'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch test suite failed.' }

& $Python -m tools.probe_sc2_compat
if ($LASTEXITCODE -ne 0) { throw 'Tencent PySC2 compatibility probe failed.' }

Write-Host 'PyTorch and Tencent PySC2 environment checks passed.' -ForegroundColor Green
