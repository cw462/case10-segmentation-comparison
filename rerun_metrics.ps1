param([string]$Python = 'python', [string]$WorkDirectory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$case10Work = (Resolve-Path -LiteralPath $WorkDirectory).Path
$case10Python = $Python
foreach ($structure in @('pulmonary_artery', 'pulmonary_vein', 'airway', 'lung')) {
    & $case10Python -B (Join-Path $case10Work 'run_metric_sets.py') $case10Work $structure
    if ($LASTEXITCODE -ne 0) { throw "Metrics failed: $structure" }
}
& $case10Python -B (Join-Path $case10Work 'build_report.py') $case10Work
if ($LASTEXITCODE -ne 0) { throw 'Report export failed' }
