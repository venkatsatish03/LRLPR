$modelPath = Join-Path $PSScriptRoot "..\models\license_plate_detector.pt"
$resolvedPath = Resolve-Path -Path $modelPath -ErrorAction SilentlyContinue

if ($resolvedPath) {
  Write-Host "Model found: $resolvedPath"
  exit 0
}

Write-Host "Model missing."
Write-Host "Put your trained YOLOv8 license plate weights here:"
Write-Host (Join-Path (Resolve-Path "$PSScriptRoot\..") "models\license_plate_detector.pt")
exit 1
