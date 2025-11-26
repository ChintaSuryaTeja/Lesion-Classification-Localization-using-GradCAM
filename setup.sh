#!/bin/bash
# Setup script for Lesion Classification project

echo "🔬 Setting up Skin Lesion Classification & Localization project..."

# Check Python version
python_version=$(python --version 2>&1)
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete! "
echo ""
echo "📋 Next steps:"
echo "1. Download the trained model file 'best_multimodal_effb3.pth'"
echo "2. Place it in the project root directory"
echo "3. Run the demo: streamlit run app.py"
echo ""
echo "💡 For detailed instructions, see README.md"