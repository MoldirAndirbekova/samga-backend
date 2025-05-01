# PowerShell script for setting up hand tracking functionality in SAMGA backend
Write-Host "Setting up hand tracking for SAMGA backend..." -ForegroundColor Green

# Install required Python packages
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install opencv-python==4.8.0.74
pip install numpy==1.24.3

# Update the requirements.txt file
$requirementsPath = "requirements.txt"
$requirements = Get-Content $requirementsPath -ErrorAction SilentlyContinue
$hasOpenCV = $requirements -contains "opencv-python==4.8.0.74"
$hasNumpy = $requirements -contains "numpy==1.24.3"

if (-not $hasOpenCV) {
    Write-Host "Adding OpenCV to requirements.txt" -ForegroundColor Yellow
    Add-Content -Path $requirementsPath -Value "opencv-python==4.8.0.74"
}

if (-not $hasNumpy) {
    Write-Host "Adding numpy to requirements.txt" -ForegroundColor Yellow
    Add-Content -Path $requirementsPath -Value "numpy==1.24.3"
}

Write-Host "Setup complete! Make sure to restart the backend server." -ForegroundColor Green
Write-Host "To test the hand tracking, navigate to the ping pong game in the frontend app." -ForegroundColor Cyan 