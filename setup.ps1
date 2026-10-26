param([string]$BasePython = "python")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

& $BasePython -c "import sys; assert sys.version_info[:2] == (3,12), 'Use Python 3.12 for the tested environment'"
if ($LASTEXITCODE -ne 0) { throw "Python version check failed." }
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $BasePython -m venv --without-pip --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the project environment." }
}
$CoursePython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
foreach ($CourseScript in @("download_resources.py", "setup_packages.py", "apply_compatibility.py", "prepare_data.py", "check_environment.py")) {
    & $CoursePython -u $CourseScript
    if ($LASTEXITCODE -ne 0) { throw "Stopped because $CourseScript failed." }
}
Write-Host "Setup complete. Run the smoke test, then the full experiment matrix."
