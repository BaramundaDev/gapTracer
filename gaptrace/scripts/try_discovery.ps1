param(
    [string]$Data = 'data/reactivation-demo',
    [ValidateSet('local', 'connector', 'prepare')]
    [string]$Engine = 'local'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.research-venv/Scripts/python.exe'
Push-Location $projectRoot
try {
    Write-Host 'Newly enabled research demonstration' -ForegroundColor Green
    Write-Host 'Both papers are invented engineering controls. This is not a validated scientific discovery.'
    if (Test-Path -LiteralPath (Join-Path $Data 'discovery.json')) {
        & $python -m gaptrace.cli discover --resume $Data --engine $Engine --max-attempts 1
    }
    else {
        & $python -m gaptrace.cli discover 'Aurora simulation memory bottleneck' --source evaluation/examples/reactivation-demo-papers.json --data $Data --papers 2 --engine $Engine --max-attempts 1
    }
    if ($LASTEXITCODE -ne 0) { throw 'Discovery did not complete. Its prepared sources and checkpoints were retained.' }
    Write-Host "`nRun state: $Data/discovery.json"
    Write-Host "Knowledge base: $Data/corpus"
}
finally { Pop-Location }
