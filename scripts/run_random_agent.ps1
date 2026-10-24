[CmdletBinding()]
param(
    [ValidateSet('1','2','3','4','5','6','7','8','9','A')]
    [string]$Difficulty = '1',
    [ValidateRange(1, 1000)]
    [int]$Episodes = 1,
    [string]$SC2Path = $(if ($env:SC2PATH) { $env:SC2PATH } else { 'C:\linux_project\TStarBot2-dev-open\sc2_runtime' }),
    [string]$Python = (Join-Path $PSScriptRoot '..\tstarbot1_pytorch_env\Scripts\python.exe'),
    [string]$LogPath = (Join-Path $PSScriptRoot '..\results\smoke_tests\random_agent.log')
)

$ErrorActionPreference = 'Stop'
$Python = [System.IO.Path]::GetFullPath($Python)
$SC2Path = [System.IO.Path]::GetFullPath($SC2Path)
$LogPath = [System.IO.Path]::GetFullPath($LogPath)
$map = Join-Path $SC2Path 'Maps\Ladder2017Season4\AbyssalReefLE.SC2Map'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $map -PathType Leaf)) { throw "AbyssalReef map not found: $map" }
$existing = @(Get-Process SC2_x64 -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0) { throw 'SC2_x64 is already running; close it before this smoke test.' }

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
$env:SC2PATH = $SC2Path
try {
    # Windows PowerShell converts native stderr piped through 2>&1 into
    # ErrorRecord objects. Gym writes its deprecation notice to stderr, so an
    # ErrorActionPreference of Stop would abort a healthy SC2 run. Temporarily
    # allow native stderr through Tee-Object and judge success by the process
    # exit code instead.
    $savedErrorActionPreference = $ErrorActionPreference
    $hasNativePreference = Test-Path Variable:\PSNativeCommandUseErrorActionPreference
    if ($hasNativePreference) {
        $savedNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        $ErrorActionPreference = 'Continue'
        & $Python -m sc2learner.bin.evaluate --agent=random "--difficulty=$Difficulty" "--num_episodes=$Episodes" 2>&1 |
            Tee-Object -FilePath $LogPath
        $pythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $savedNativePreference
        }
    }
    if ($pythonExitCode -ne 0) {
        throw "Random-agent evaluation failed with exit code $pythonExitCode; see $LogPath"
    }
}
finally {
    Get-Process SC2_x64 -ErrorAction SilentlyContinue | Stop-Process -Force
}
Write-Host "Random-agent log: $LogPath" -ForegroundColor Green
