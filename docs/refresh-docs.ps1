$ErrorActionPreference = 'Stop'

# Find workspace root (where README.md is)
$workspaceRoot = if ($PSScriptRoot -match '.+\\docs$') {
    Split-Path $PSScriptRoot -Parent
} else {
    $PSScriptRoot
}

$docsDir = Join-Path $workspaceRoot "docs"
$configPath = Join-Path $docsDir "Doxyfile"
$buildDir = Join-Path $docsDir "build/html"

Write-Host "Workspace root: $workspaceRoot"
Write-Host "Docs dir: $docsDir"
Write-Host "Running doxygen from: $workspaceRoot"

# Create build directory if needed
$null = New-Item -ItemType Directory -Path $buildDir -Force

# Run doxygen from workspace root
Push-Location $workspaceRoot
doxygen $configPath
Pop-Location

$source = Join-Path $buildDir "index.html"
$dest   = Join-Path $docsDir "index.html"

if (-not (Test-Path $source)) {
    throw "Doxygen did not produce $source"
}

Write-Host "Copying $source -> $dest"
Copy-Item -Path $source -Destination $dest -Force

$content = Get-Content -Path $dest -Raw
if ($content -notmatch '<base href="build/html/"') {
    Write-Host "Injecting <base> for relocated assets"
    $updated = $content -replace '(<head>\s*)', '$1<base href="build/html/" />`n'
    Set-Content -Path $dest -Value $updated -Encoding utf8
} else {
    Write-Host "<base> already present; leaving as-is"
}

Write-Host "Done. Open docs/index.html to view."
