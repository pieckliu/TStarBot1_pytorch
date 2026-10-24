[CmdletBinding()]
param(
    [string]$SC2Path = $(
        if ($env:SC2PATH) { $env:SC2PATH }
        else { 'C:\linux_project\TStarBot2-dev-open\sc2_runtime' }
    ),
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\results\pytorch_sc2_smoke'),
    [ValidateRange(1024, 65533)]
    [int]$PortA = 15800,
    [ValidateRange(1025, 65534)]
    [int]$PortB = 15801
)

$ErrorActionPreference = 'Stop'
$Python = [System.IO.Path]::GetFullPath($Python)
$SC2Path = [System.IO.Path]::GetFullPath($SC2Path)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $SC2Path -PathType Container)) {
    throw "SC2 runtime not found: $SC2Path"
}
if ($PortA -eq $PortB) { throw 'PortA and PortB must be different.' }

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$env:SC2PATH = $SC2Path
$existingSc2 = @(Get-Process SC2_x64 -ErrorAction SilentlyContinue)
if ($existingSc2.Count -gt 0) {
    throw 'SC2_x64 is already running; close it before this smoke test.'
}

$common = @(
    '-m', 'sc2learner.bin.train_ppo',
    '--policy=mlp',
    '--unroll_length=8',
    '--batch_size=1',
    '--difficulties=1',
    '--device=cpu',
    "--port_A=$PortA",
    "--port_B=$PortB",
    '--print_interval=1',
    '--save_interval=1'
)
$learnerArgs = $common + @(
    '--job_name=learner',
    '--max_learner_updates=1',
    "--save_dir=$OutputDirectory"
)
$actorArgs = $common + @(
    '--job_name=actor',
    '--learner_ip=localhost',
    '--max_actor_unrolls=1'
)

$learner = $null
$actor = $null
try {
    $learner = Start-Process -FilePath $Python -ArgumentList $learnerArgs `
        -RedirectStandardOutput (Join-Path $OutputDirectory 'learner.stdout.log') `
        -RedirectStandardError (Join-Path $OutputDirectory 'learner.stderr.log') `
        -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 5
    $actor = Start-Process -FilePath $Python -ArgumentList $actorArgs `
        -RedirectStandardOutput (Join-Path $OutputDirectory 'actor.stdout.log') `
        -RedirectStandardError (Join-Path $OutputDirectory 'actor.stderr.log') `
        -WindowStyle Hidden -PassThru

    if (-not $actor.WaitForExit(180000)) { throw 'SC2 actor timed out after 180 seconds.' }
    if (-not $learner.WaitForExit(60000)) { throw 'PPO learner timed out after actor completion.' }
    $actor.WaitForExit()
    $learner.WaitForExit()
    if ($actor.ExitCode -ne 0) {
        throw "SC2 actor failed with exit code $($actor.ExitCode). See actor.stderr.log."
    }
    if ($learner.ExitCode -ne 0) {
        throw "PPO learner failed with exit code $($learner.ExitCode). See learner.stderr.log."
    }
}
finally {
    if ($actor -and -not $actor.HasExited) { Stop-Process -Id $actor.Id -Force }
    if ($learner -and -not $learner.HasExited) { Stop-Process -Id $learner.Id -Force }
    Get-Process SC2_x64 -ErrorAction SilentlyContinue | Stop-Process -Force
}

$checkpoint = Join-Path $OutputDirectory 'ppo_1.pt'
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "Training exited without the expected checkpoint: $checkpoint"
}
Write-Host "Real-SC2 PPO smoke test passed: $checkpoint" -ForegroundColor Green
