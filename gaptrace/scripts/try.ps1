param(
    [string]$Data = 'data/terminal-demo',
    [string]$Question = 'What accuracy does the replay buffer achieve on Split CIFAR-100 in papers A and B?',
    [switch]$NoAsk,
    [switch]$NoShell
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.research-venv/Scripts/python.exe'
$previousCostMap = $env:LITELLM_LOCAL_MODEL_COST_MAP

function Invoke-GapTrace {
    param([string[]]$CliArguments)
    Write-Host "`n> gaptrace $($CliArguments -join ' ')" -ForegroundColor Cyan
    & $python -m gaptrace.cli @CliArguments
    if ($LASTEXITCODE -ne 0) { throw "GapTrace command failed (exit $LASTEXITCODE)." }
}

Push-Location $projectRoot
try {
    # Local inference does not need LiteLLM's remote price catalogue.
    $env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
    if (-not (Test-Path -LiteralPath $python)) {
        throw 'Project environment missing. See docs/TERMINAL.md for setup.'
    }
    Write-Host 'GapTrace terminal walkthrough' -ForegroundColor Green
    Write-Host 'The demo uses three invented papers. Gap cards are candidates requiring review.'
    & $python -c 'import gaptrace, tokenizers; print(gaptrace.__file__)'
    if ($LASTEXITCODE -ne 0) { throw 'Install the project environment using docs/TERMINAL.md.' }
    if (-not $NoAsk) {
        & $python -c 'import litellm'
        if ($LASTEXITCODE -ne 0) { throw 'Install the models extra using docs/TERMINAL.md.' }
    }
    $dataRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Data))
    if (-not (Test-Path -LiteralPath (Join-Path $dataRoot 'passages.jsonl'))) {
        if ((Test-Path -LiteralPath $dataRoot) -and (Get-ChildItem -LiteralPath $dataRoot -Force | Select-Object -First 1)) {
            throw "Refusing to write a demo into the nonempty directory $dataRoot. Choose a new -Data path."
        }
        Invoke-GapTrace -CliArguments @('demo', '--data', $dataRoot)
    }
    Invoke-GapTrace -CliArguments @('status', '--data', $dataRoot)
    Invoke-GapTrace -CliArguments @('search', 'replay buffer', '--data', $dataRoot)
    Invoke-GapTrace -CliArguments @('fact', '--method', 'replay', '--data', $dataRoot)
    Invoke-GapTrace -CliArguments @('gaps', '--data', $dataRoot)
    $cardJson = & $python -m gaptrace.cli gaps --json --data $dataRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not read gap candidates.' }
    $cards = @((($cardJson -join "`n") | ConvertFrom-Json))
    if ($cards.Count -gt 0) {
        Invoke-GapTrace -CliArguments @('trail', $cards[0].id, '--data', $dataRoot)
    }
    Invoke-GapTrace -CliArguments @('passage', 'a-1', '--data', $dataRoot)
    if (-not $NoAsk) {
        Write-Host "`nAsking Qwen locally. The first request may load the model; please wait." -ForegroundColor Yellow
        Invoke-GapTrace -CliArguments @('ask', $Question, '--local-only', '--json', '--data', $dataRoot)
        $runs = Join-Path $dataRoot 'runs'
        $latest = Get-ChildItem -LiteralPath $runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
            $trailPath = Join-Path $latest.FullName 'trail.jsonl'
            Write-Host "`nRun files: $($latest.FullName)" -ForegroundColor Green
            if (Test-Path -LiteralPath $trailPath) {
                Write-Host 'Answer validation (source checks, removed claims, calls and latency):'
                Get-Content -LiteralPath $trailPath | ForEach-Object {
                    $entry = $_ | ConvertFrom-Json
                    if ($entry.kind -in @('answer_validation', 'answer_validation_summary')) { $_ }
                }
            }
        }
    }
    Write-Host "`nCorpus files: $dataRoot"
    Write-Host 'Next commands: status | gaps | search "replay" | ask "What limitations are reported?" --local-only | help | exit'
    if (-not $NoShell) {
        Invoke-GapTrace -CliArguments @('shell', '--data', $dataRoot)
    }
}
finally {
    $env:LITELLM_LOCAL_MODEL_COST_MAP = $previousCostMap
    Pop-Location
}
