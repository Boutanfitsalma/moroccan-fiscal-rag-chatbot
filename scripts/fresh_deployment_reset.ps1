# Fresh Deployment Reset Script
# This script resets the application to a fresh deployment state

Write-Host "🏗️  Moroccan Fiscal RAG - Fresh Deployment Reset" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker ps | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check if Docker is available
if (-not (Test-DockerRunning)) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker is running" -ForegroundColor Green

# Step 1: Stop and remove existing containers
Write-Host "`n🛑 Stopping existing containers..." -ForegroundColor Yellow
try {
    docker-compose down --remove-orphans
    Write-Host "✅ Containers stopped" -ForegroundColor Green
}
catch {
    Write-Host "⚠️  No containers to stop or error occurred" -ForegroundColor Yellow
}

# Step 2: Remove Docker volumes and images to ensure clean state
Write-Host "`n🧹 Cleaning Docker resources..." -ForegroundColor Yellow
try {
    # Remove project-specific volumes
    docker volume ls --filter name=moroccan_fiscal_rag -q | ForEach-Object { docker volume rm $_ }
    
    # Remove project images to force rebuild
    docker image ls --filter reference=moroccan_fiscal_rag-* -q | ForEach-Object { docker image rm $_ -f }
    
    Write-Host "✅ Docker resources cleaned" -ForegroundColor Green
}
catch {
    Write-Host "⚠️  Some Docker resources could not be removed (may not exist)" -ForegroundColor Yellow
}

# Step 3: Run Python script to clean application data
Write-Host "`n🧹 Cleaning application data..." -ForegroundColor Yellow
try {
    & "C:\Users\yassi\anaconda3\envs\moroccan_fiscal_rag\python.exe" reset_fresh_state.py
    Write-Host "✅ Application data cleaned" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to clean application data: $_" -ForegroundColor Red
    Write-Host "Continuing with Docker rebuild..." -ForegroundColor Yellow
}

# Step 4: Rebuild and start containers
Write-Host "`n🔄 Rebuilding and starting containers..." -ForegroundColor Yellow
try {
    docker-compose up -d --build --force-recreate
    Write-Host "✅ Containers rebuilt and started" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to start containers: $_" -ForegroundColor Red
    exit 1
}

# Step 5: Wait for containers to be healthy
Write-Host "`n⏳ Waiting for containers to be healthy..." -ForegroundColor Yellow
$maxAttempts = 30
$attempts = 0

do {
    Start-Sleep -Seconds 2
    $attempts++
    $status = docker ps --format "table {{.Names}}\t{{.Status}}" | Select-String -Pattern "healthy"
    
    if ($status.Count -ge 2) {
        Write-Host "✅ Both containers are healthy" -ForegroundColor Green
        break
    }
    
    Write-Host "⏳ Attempt $attempts/$maxAttempts - Waiting for health checks..." -ForegroundColor Yellow
    
} while ($attempts -lt $maxAttempts)

if ($attempts -ge $maxAttempts) {
    Write-Host "⚠️  Containers may not be fully healthy yet, but continuing..." -ForegroundColor Yellow
}

# Step 6: Display status
Write-Host "`n📊 Container Status:" -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Step 7: Test endpoints
Write-Host "`n🧪 Testing endpoints..." -ForegroundColor Yellow

try {
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
    if ($healthResponse.status -eq "healthy") {
        Write-Host "✅ API health check passed" -ForegroundColor Green
    }
}
catch {
    Write-Host "❌ API health check failed: $_" -ForegroundColor Red
}

try {
    $webResponse = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 10 -UseBasicParsing
    if ($webResponse.StatusCode -eq 200) {
        Write-Host "✅ Web UI accessible" -ForegroundColor Green
    }
}
catch {
    Write-Host "❌ Web UI not accessible: $_" -ForegroundColor Red
}

# Step 8: Summary
Write-Host "`n🎯 Fresh Deployment Reset Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray
Write-Host "📋 Status Summary:" -ForegroundColor Cyan
Write-Host "   • Vector database: Reset to empty state" -ForegroundColor White
Write-Host "   • Docker containers: Rebuilt from scratch" -ForegroundColor White
Write-Host "   • Upload logic: Updated to save JSON files" -ForegroundColor White
Write-Host "   • Original data: Preserved for re-indexing" -ForegroundColor White

Write-Host "`n🌐 Access Points:" -ForegroundColor Cyan
Write-Host "   • API Server: http://localhost:8000" -ForegroundColor White
Write-Host "   • Web UI: http://localhost:3000" -ForegroundColor White
Write-Host "   • API Docs: http://localhost:8000/docs" -ForegroundColor White

Write-Host "`n🧪 Ready to Test:" -ForegroundColor Cyan
Write-Host "   1. Upload a PDF document via Web UI" -ForegroundColor White
Write-Host "   2. JSON file will be saved in data/ folder" -ForegroundColor White
Write-Host "   3. Document will be indexed into vector database" -ForegroundColor White
Write-Host "   4. Test queries about uploaded content" -ForegroundColor White

Write-Host "`n🚀 Application is ready for fresh deployment testing!" -ForegroundColor Green