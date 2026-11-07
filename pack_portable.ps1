# ==============================================================================
# SCRIPT DONG GOI VIMAX STUDIO THANH FILE NEN PORTABLE .ZIP
# ==============================================================================
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$outputZip = Join-Path $root "ViMax_Studio_Portable.zip"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   BAT DAU DONG GOI VIMAX STUDIO (PORTABLE PACKAGE)       " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $outputZip) {
    Write-Host "Dang xoa file zip cu..." -ForegroundColor Yellow
    Remove-Item -Force $outputZip
}

# Danh sach cac thu muc va file can dong goi
$includeItems = @(
    "ViMax_Studio.exe",
    "run_web.bat",
    "HUONG_DAN_SU_DUNG.md",
    "readme.md",
    "bin",
    ".venv",
    "web",
    "agents",
    "agent_runtime",
    "configs",
    "interfaces",
    "pipelines",
    "prompts",
    "tools",
    "utils",
    "assets"
)

Write-Host "Dang nen cac thanh phan vao: $outputZip ..." -ForegroundColor Green
Write-Host "(Qua trinh nay mat khoang 1-2 phut do nen thu vien Python)..." -ForegroundColor Gray

# Dung tar hoac Compress-Archive
tar.exe -a -c -f $outputZip $includeItems

if (Test-Path $outputZip) {
    $sizeMB = [math]::Round((Get-Item $outputZip).Length / 1MB, 2)
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "   DONG GOI THANH CONG!                                   " -ForegroundColor Green
    Write-Host "   File: $outputZip                                       " -ForegroundColor Green
    Write-Host "   Dung luong file: $sizeMB MB                            " -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "Ban co the upload file zip nay len Google Drive/OneDrive  " -ForegroundColor Yellow
    Write-Host "de chia se cho ban be. Nguoi nhan chi can giai nen va bam " -ForegroundColor Yellow
    Write-Host "dup vao 'ViMax_Studio.exe' la dung duoc ngay!             " -ForegroundColor Yellow
} else {
    Write-Host "Co loi xay ra trong qua trinh nen file." -ForegroundColor Red
}
