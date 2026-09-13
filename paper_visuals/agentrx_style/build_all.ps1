param(
    [string]$Python = 'python',
    [string]$Tectonic = ''
)
$ErrorActionPreference = 'Stop'
if (-not $Tectonic) {
    $figureTexCommand = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($figureTexCommand) { $Tectonic = $figureTexCommand.Source }
    else {
        $figureTexCandidate = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Paper-Figure-Toolkit\runtime\tectonic-0.17.0\tectonic.exe'
        if (Test-Path -LiteralPath $figureTexCandidate) { $Tectonic = $figureTexCandidate }
        else { throw 'Pass -Tectonic with the path to tectonic.exe (tested: 0.17.0).' }
    }
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python 'plot_results.py'
    if ($LASTEXITCODE -ne 0) { throw 'Data plotting failed.' }
    & $Tectonic -X compile 'fig01_pipeline.tex' --keep-logs
    if ($LASTEXITCODE -ne 0) { throw 'TikZ compilation failed.' }
    & $Python -c "from pathlib import Path; from build_artifacts import export_pdf; export_pdf(Path('fig01_pipeline.pdf'))"
    if ($LASTEXITCODE -ne 0) { throw 'Diagram export failed.' }
    & $Python 'build_artifacts.py' --tectonic $Tectonic
    if ($LASTEXITCODE -ne 0) { throw 'Table, code or atlas build failed.' }
    & $Python 'verify_artifacts.py'
    if ($LASTEXITCODE -ne 0) { throw 'Artifact verification failed.' }
}
finally { Pop-Location }
