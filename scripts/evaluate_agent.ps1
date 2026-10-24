[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ModelPath,
    [ValidateSet('1','2','3','4','5','6','7','8','9','A')]
    [string]$Difficulty = '1',
    [ValidateRange(1, 1000)]
    [int]$Episodes = 1,
    [switch]$Deterministic,
    [string]$SC2Path = $(if ($env:SC2PATH) { $env:SC2PATH } else { 'C:\linux_project\TStarBot2-dev-open\sc2_runtime' }),
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe'),
    [string]$LogPath = (Join-Path $PSScriptRoot '..\results\smoke_tests\ppo_evaluation.log')
)

$ErrorActionPreference = 'Stop'
$Python = [System.IO.Path]::GetFullPath($Python)
$SC2Path = [System.IO.Path]::GetFullPath($SC2Path)
$ModelPath = [System.IO.Path]::GetFullPath($ModelPath)
$LogPath = [System.IO.Path]::GetFullPath($LogPath)
$map = Join-Path $SC2Path 'Maps\Ladder2017Season4\AbyssalReefLE.SC2Map'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw "Checkpoint not found: $ModelPath" }
if (-not (Test-Path -LiteralPath $map -PathType Leaf)) { throw "AbyssalReef map not found: $map" }
if (Get-Process SC2_x64 -ErrorAction SilentlyContinue) {
    throw 'SC2_x64 is already running; close it before evaluation.'
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
$env:SC2PATH = $SC2Path
$arguments = @(
    '-m', 'sc2learner.bin.evaluate', '--agent=ppo',
    "--difficulty=$Difficulty", "--num_episodes=$Episodes",
    "--model_path=$ModelPath", '--device=cpu'
)
if ($Deterministic) { $arguments += '--deterministic' }
try {
    & $Python @arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { throw "PPO evaluation failed; see $LogPath" }
}
finally {
    Get-Process SC2_x64 -ErrorAction SilentlyContinue | Stop-Process -Force
}
Write-Host "Evaluation log: $LogPath" -ForegroundColor Green
