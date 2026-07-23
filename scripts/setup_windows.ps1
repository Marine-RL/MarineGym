[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$IsaacSimPath = $env:ISAACSIM_PATH
)

$ErrorActionPreference = "Stop"

if (-not $IsaacSimPath) {
    throw "Set ISAACSIM_PATH or pass -IsaacSimPath with the Isaac Sim 4.1.0 installation directory."
}

$IsaacSimPath = (Resolve-Path -LiteralPath $IsaacSimPath).Path
$PythonLauncher = Join-Path $IsaacSimPath "python.bat"
$Experience = Join-Path $IsaacSimPath "apps\omni.isaac.sim.python.kit"
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $PythonLauncher -PathType Leaf)) {
    throw "Isaac Sim python launcher not found: $PythonLauncher"
}
if (-not (Test-Path -LiteralPath $Experience -PathType Leaf)) {
    throw "Isaac Sim 4.1 Python experience not found: $Experience"
}

$env:ISAACSIM_PATH = $IsaacSimPath
$env:EXP_PATH = Join-Path $IsaacSimPath "apps"

Write-Host "Installing MarineGym into the Isaac Sim Python environment."
& $PythonLauncher -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonLauncher -m pip install -e $RepositoryRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonLauncher -c "from marinegym._platform import resolve_experience_path; print(resolve_experience_path())"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "MarineGym is installed. Keep ISAACSIM_PATH set when using scripts\marinegym.ps1."
