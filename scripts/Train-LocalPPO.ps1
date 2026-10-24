[CmdletBinding()]
param(
    [ValidateRange(1, 1000000)]
    [int]$Updates = 10,
    [ValidateRange(8, 65536)]
    [int]$RolloutSteps = 128,
    [string]$Device = 'cpu',
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe'),
    [string]$Checkpoint = (Join-Path $PSScriptRoot '..\results\pytorch_local\ppo.pt')
)

$ErrorActionPreference = 'Stop'
$Python = [System.IO.Path]::GetFullPath($Python)
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$checkpointDirectory = Split-Path -Parent $Checkpoint
New-Item -ItemType Directory -Force -Path $checkpointDirectory | Out-Null

& $Python -m sc2learner.bin.train_ppo_local `
    --updates $Updates `
    --rollout-steps $RolloutSteps `
    --device $Device `
    --seed 42 `
    --checkpoint $Checkpoint
if ($LASTEXITCODE -ne 0) { throw 'Local PPO training failed.' }

Write-Host "Local PPO checkpoint: $Checkpoint" -ForegroundColor Green
