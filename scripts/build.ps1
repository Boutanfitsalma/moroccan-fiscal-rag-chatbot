# build.ps1 - PowerShell Docker Build Script with Progress Tracking

Write-Host "🚀 Starting Moroccan Fiscal RAG Docker Build..." -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw
    }
} catch {
    Write-Host "❌ Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Show build progress with timestamps
Write-Host "⏰ Build started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
Write-Host ""

# Build with progress output and no cache for clean build
Write-Host "🔨 Building Docker image with progress tracking..." -ForegroundColor Green

# Start docker build with progress tracking
$process = Start-Process -FilePath "docker" -ArgumentList "build", "--progress=plain", "--no-cache", "-t", "moroccan-fiscal-rag", "." -NoNewWindow -PassThru -RedirectStandardOutput "build_output.log" -RedirectStandardError "build_error.log"

# Monitor the build process
while (-not $process.HasExited) {
    if (Test-Path "build_output.log") {
        $content = Get-Content "build_output.log" -Tail 10 -ErrorAction SilentlyContinue
        if ($content) {
            foreach ($line in $content) {
                if ($line -match "^\s*$") { continue }
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $line" -ForegroundColor White
            }
        }
    }
    Start-Sleep -Seconds 2
}

# Wait for process to complete
$process.WaitForExit()

# Clean up log files
Remove-Item "build_output.log" -ErrorAction SilentlyContinue
Remove-Item "build_error.log" -ErrorAction SilentlyContinue

# Check if build was successful
if ($process.ExitCode -eq 0) {
    Write-Host ""
    Write-Host "✅ Docker build completed successfully!" -ForegroundColor Green
    Write-Host "⏰ Build finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🚢 To start the application, run:" -ForegroundColor Cyan
    Write-Host "   docker-compose up -d" -ForegroundColor White
    Write-Host ""
    Write-Host "🌐 The API will be available at: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📋 API documentation: http://localhost:8000/docs" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    Write-Host "⏰ Build failed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
    exit 1
}