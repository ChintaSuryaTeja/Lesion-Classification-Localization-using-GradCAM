@echo off
REM Setup script for Lesion Classification project (Windows)

echo 🔬 Setting up Skin Lesion Classification ^& Localization project...

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.7+ first.
    pause
    exit /b 1
)

echo ✅ Python found!

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment  
echo ⚡ Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

echo.
echo 🎉 Setup complete! 
echo.
echo 📋 Next steps:
echo 1. Download the trained model file 'best_multimodal_effb3.pth'
echo 2. Place it in the project root directory
echo 3. Run the demo: streamlit run app.py
echo.
echo 💡 For detailed instructions, see README.md
echo.
pause