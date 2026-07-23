[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet("train", "evaluate")]
    [string]$Command = "train",

    [Parameter(Mandatory = $false)]
    [string]$IsaacSimPath = $env:ISAACSIM_PATH,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MarineGymArgs
)

$ErrorActionPreference = "Stop"

if (-not $IsaacSimPath) {
    throw "Set ISAACSIM_PATH or pass -IsaacSimPath with the Isaac Sim 4.1.0 installation directory."
}

$IsaacSimPath = (Resolve-Path -LiteralPath $IsaacSimPath).Path
$PythonLauncher = Join-Path $IsaacSimPath "python.bat"
$TargetScript = Join-Path $PSScriptRoot "$Command.py"

if (-not (Test-Path -LiteralPath $PythonLauncher -PathType Leaf)) {
    throw "Isaac Sim python launcher not found: $PythonLauncher"
}

$env:ISAACSIM_PATH = $IsaacSimPath
$env:EXP_PATH = Join-Path $IsaacSimPath "apps"

& $PythonLauncher $TargetScript @MarineGymArgs
exit $LASTEXITCODE
