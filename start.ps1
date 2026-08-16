param(
    [ValidateRange(1, 65535)]
    [int]$Port = 5050,

    [switch]$Restart,

    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$envFile = Join-Path $projectRoot '.env'
$localUrl = "http://127.0.0.1:$Port/matches"

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Warning "No .env file found. Live fixtures and OpenAI research will remain disabled."
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) {
            continue
        }

        $parts = $trimmed.Split('=', 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            Write-Warning "Skipped invalid environment variable name: $name"
            continue
        }

        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

function Test-Configured {
    param([string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    return -not [string]::IsNullOrWhiteSpace($value)
}

function Get-ListeningProcessIds {
    param([int]$LocalPort)

    return @(
        Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
}

Push-Location $projectRoot
try {
    Import-DotEnv -Path $envFile

    $python = $null
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) {
        $python = $venvPython
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $python = $pythonCommand.Source
        }
    }

    if (-not $python) {
        throw "Python was not found. Install Python 3.11+ or create .venv in this project."
    }

    $listeners = Get-ListeningProcessIds -LocalPort $Port
    if ($listeners.Count -gt 0) {
        if ($Restart) {
            Write-Host "Stopping the existing local server on port $Port..." -ForegroundColor Yellow
            foreach ($processId in $listeners) {
                Stop-Process -Id $processId -Force
            }
        } else {
            Write-Host "Matchup Matrix is already running at $localUrl" -ForegroundColor Green
            Write-Host "Use .\start.ps1 -Restart to restart it." -ForegroundColor DarkGray
            if (-not $NoBrowser) {
                Start-Process $localUrl
            }
            exit 0
        }
    }

    $env:APP_ENV = 'development'
    $env:PORT = [string]$Port

    Write-Host "Matchup Matrix local environment" -ForegroundColor Cyan
    Write-Host "  CricketData: $(if (Test-Configured 'CRICKETDATA_API_KEY') { 'configured' } else { 'not configured' })"
    Write-Host "  OpenAI:      $(if (Test-Configured 'OPENAI_API_KEY') { 'configured' } else { 'not configured' })"
    Write-Host "  Model:       $(if (Test-Configured 'OPENAI_RESEARCH_MODEL') { $env:OPENAI_RESEARCH_MODEL } else { 'default' })"
    Write-Host "  URL:         $localUrl"
    Write-Host "Press Ctrl+C to stop the server." -ForegroundColor DarkGray

    if (-not $NoBrowser) {
        Start-Process $localUrl
    }

    $runCode = "import os; from app import app; app.run(host='127.0.0.1', port=int(os.environ['PORT']), debug=False, use_reloader=False)"
    & $python -c $runCode
    if ($LASTEXITCODE -ne 0) {
        throw "The Flask server exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}