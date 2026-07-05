$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BundledPython = "C:\Users\matig\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

function Resolve-Python {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return $pyCmd.Source
    }

    if (Test-Path $BundledPython) {
        return $BundledPython
    }

    throw "No se encontro Python. Instala Python 3.10+ o agrega python.exe al PATH."
}

Set-Location $ProjectRoot

if (-not (Test-Path $VenvPython)) {
    $Python = Resolve-Python
    Write-Host "Creando entorno virtual con: $Python"
    & $Python -m venv ".venv"
}

Write-Host "Actualizando pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Instalando dependencias de requirements.txt..."
& $VenvPython -m pip install -r "requirements.txt"

Write-Host ""
Write-Host "Listo. Ejecuta el dashboard con:"
Write-Host ".\run_streamlit.ps1"
