# Install DAC Player for the current user and associate .dac files with it (no admin rights needed).
#   powershell -ExecutionPolicy Bypass -File packaging\install-windows.ps1 -BuildDir build -Model dac.gguf
#   powershell -ExecutionPolicy Bypass -File packaging\install-windows.ps1 -Uninstall
param(
    [string]$BuildDir = "build",
    [string]$Model = "dac.gguf",
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
$dest = Join-Path $env:LOCALAPPDATA "Programs\DAC Player"

if ($Uninstall) {
    if (Test-Path "$dest\dac-player.exe") {
        Start-Process "$dest\dac-player.exe" -ArgumentList "--unregister", "--quiet" -Wait
    }
    Remove-Item -Recurse -Force $dest -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $env:LOCALAPPDATA "DacPlayer") -ErrorAction SilentlyContinue
    Write-Host "DAC Player was removed."
    return
}

New-Item -ItemType Directory -Force $dest | Out-Null
foreach ($exe in "dac-player.exe", "dac-native.exe") {
    $src = Get-ChildItem -Path $BuildDir -Recurse -Filter $exe | Select-Object -First 1
    if (-not $src) { throw "$exe not found under $BuildDir - build the project first" }
    Copy-Item $src.FullName $dest -Force
}
Copy-Item $Model (Join-Path $dest "dac.gguf") -Force

# MSVC runtime + OpenMP DLLs next to the executables (if Visual Studio's redistributables are present)
$redist = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64" -Directory -ErrorAction SilentlyContinue |
          Sort-Object FullName -Descending | Select-Object -First 1
if ($redist) {
    Get-ChildItem $redist.FullName -Recurse -Include msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll, vcomp140.dll |
        Where-Object { $_.FullName -notmatch "onecore|debug" } | ForEach-Object { Copy-Item $_.FullName $dest -Force }
}

$p = Start-Process (Join-Path $dest "dac-player.exe") -ArgumentList "--register", "--quiet" -Wait -PassThru
if ($p.ExitCode -ne 0) { throw "registering the .dac association failed" }
Write-Host "DAC Player installed to $dest; .dac files now open with it."
