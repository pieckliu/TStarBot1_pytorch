[CmdletBinding()]
param(
    [string]$SC2Path = $(if ($env:SC2PATH) { $env:SC2PATH } else { 'C:\linux_project\TStarBot2-dev-open\sc2_runtime' }),
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\results\smoke_tests\ppo'),
    [int]$PortA = 15800,
    [int]$PortB = 15801
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'Train-RealSC2Smoke.ps1') `
    -SC2Path $SC2Path -Python $Python -OutputDirectory $OutputDirectory `
    -PortA $PortA -PortB $PortB
