[CmdletBinding()]
param(
    [ValidateSet('1','2','3','4','5','6','7','8','9','A')]
    [string]$Difficulty = '1',
    [string]$SC2Path = $(if ($env:SC2PATH) { $env:SC2PATH } else { 'C:\linux_project\TStarBot2-dev-open\sc2_runtime' }),
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe'),
    [string]$LogPath = (Join-Path $PSScriptRoot '..\results\smoke_tests\macro_actions.log')
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_random_agent.ps1') `
    -Difficulty $Difficulty -Episodes 1 -SC2Path $SC2Path `
    -Python $Python -LogPath $LogPath
Write-Host 'Inspect the action distribution at the end of the log for executed macro actions.' -ForegroundColor Green
