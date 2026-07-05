$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "No existe .venv. Ejecutando setup_streamlit.ps1 primero..."
    & (Join-Path $ProjectRoot "setup_streamlit.ps1")
}

Set-Location $ProjectRoot
& $VenvPython -m streamlit run "app_streamlit.py"
