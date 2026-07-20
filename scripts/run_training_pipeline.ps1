[CmdletBinding()]
param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick",

    [string]$VenvPath = ".venv",

    [string]$PythonLauncher = "py",

    [string]$ConfigPath = "configs/experiments.json",

    [ValidateRange(0, 10000)]
    [int]$OptunaTrials = 0,

    [switch]$EnableMlflow,

    [switch]$StartInfrastructure,

    [switch]$RunTests,

    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $Arguments"
    }
}

$projectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)
$resolvedVenv = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot $VenvPath)
)
$projectPrefix = $projectRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar

if (-not $resolvedVenv.StartsWith(
    $projectPrefix,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "VenvPath must resolve inside the project directory."
}

$venvPython = Join-Path $resolvedVenv "Scripts\python.exe"
$pipelineCommand = Join-Path $resolvedVenv "Scripts\obesity-training-pipeline.exe"

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($SkipInstall) {
            throw "The virtual environment does not exist and -SkipInstall was used."
        }
        Write-Host "[1/6] Creating virtual environment at $resolvedVenv"
        Invoke-Checked -Command $PythonLauncher -Arguments @(
            "-m", "venv", $resolvedVenv
        )
    }
    else {
        Write-Host "[1/6] Reusing virtual environment at $resolvedVenv"
    }

    if (-not $SkipInstall) {
        $requirements = if ($Mode -eq "full") {
            "requirements-modeling.txt"
        }
        else {
            "requirements-training.txt"
        }
        Write-Host "[2/6] Installing dependencies from $requirements"
        Invoke-Checked -Command $venvPython -Arguments @(
            "-m", "pip", "install", "--requirement", $requirements
        )
        Invoke-Checked -Command $venvPython -Arguments @(
            "-m", "pip", "install", "--no-deps", "--editable", "."
        )
    }
    else {
        Write-Host "[2/6] Dependency installation skipped"
    }

    if ($StartInfrastructure) {
        Write-Host "[3/6] Starting Docker Compose infrastructure"
        Invoke-Checked -Command "docker" -Arguments @("compose", "up", "-d")
    }
    else {
        Write-Host "[3/6] Infrastructure startup skipped"
    }

    if ($RunTests) {
        Write-Host "[4/6] Running unit tests"
        Invoke-Checked -Command $venvPython -Arguments @(
            "-m", "pytest", "-q", "tests/unit"
        )
    }
    else {
        Write-Host "[4/6] Unit tests skipped"
    }

    Write-Host "[5/6] Initializing dataset and running $Mode training"
    $pipelineArguments = @("--mode", $Mode)
    if ($Mode -eq "full") {
        $pipelineArguments += @("--config", $ConfigPath)
    }
    if ($OptunaTrials -gt 0) {
        $pipelineArguments += @("--optuna-trials", $OptunaTrials.ToString())
    }
    if ($EnableMlflow) {
        $pipelineArguments += "--enable-mlflow"
    }
    Invoke-Checked -Command $pipelineCommand -Arguments $pipelineArguments

    Write-Host "[6/6] Pipeline completed successfully"
    Write-Host "Artifacts are available under artifacts/runs/."
}
finally {
    Pop-Location
}
