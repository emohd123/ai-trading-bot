# Push this project to GitHub
# Run: .\scripts\push-to-github.ps1
# Or from repo root: powershell -File scripts\push-to-github.ps1

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $repoRoot

$username = Read-Host "Enter your GitHub username"
$reponame = Read-Host "Enter repository name (e.g. ai-trading-bot)"
if ([string]::IsNullOrWhiteSpace($username) -or [string]::IsNullOrWhiteSpace($reponame)) {
    Write-Host "Username and repo name are required." -ForegroundColor Red
    exit 1
}

$url = "https://github.com/$username/$reponame.git"
Write-Host "Setting origin to $url" -ForegroundColor Cyan
git remote set-url origin $url

Write-Host "Pushing main to origin..." -ForegroundColor Cyan
git push -u origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "Done. Your code is on GitHub: https://github.com/$username/$reponame" -ForegroundColor Green
} else {
    Write-Host "Push failed. Create the repo first: https://github.com/new (name: $reponame)" -ForegroundColor Yellow
    Write-Host "Then run this script again." -ForegroundColor Yellow
}
