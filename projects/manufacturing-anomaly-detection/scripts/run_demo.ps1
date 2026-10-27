$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'

Push-Location $projectRoot
try {
    python -m unittest discover -s tests -v
    python -m crowe_anomaly.train
    python -m crowe_anomaly.score `
        --input data/manufacturing_telemetry.csv `
        --output artifacts/all_scored.csv
    Write-Host 'Demo complete. Review artifacts/metrics.json and artifacts/stakeholder_summary.md.'
}
finally {
    Pop-Location
}

