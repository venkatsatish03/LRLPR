param(
  [switch]$Reload
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$arguments = @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000")

if ($Reload) {
  $arguments += "--reload"
}

& $python @arguments
