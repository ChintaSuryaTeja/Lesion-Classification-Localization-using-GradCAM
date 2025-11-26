Write-Host "Starting Skin Lesion Classification Demo..." -ForegroundColor Green
Write-Host ""
Write-Host "Make sure you have activated your virtual environment!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Running: python -m streamlit run app.py" -ForegroundColor Cyan
Write-Host ""

# Check if the model file exists
if (Test-Path "best_multimodal_effb3.pth") {
    Write-Host "Model file found. Starting demo..." -ForegroundColor Green
    echo "" | C:/Users/jumpi/OneDrive/Desktop/DLV/venv/Scripts/python.exe -m streamlit run app.py
} else {
    Write-Host "Error: Model file 'best_multimodal_effb3.pth' not found!" -ForegroundColor Red
    Write-Host "Please make sure the trained model is in the project directory." -ForegroundColor White
    pause
}