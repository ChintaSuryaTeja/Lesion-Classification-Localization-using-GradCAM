# 🔬 Skin Lesion Classification & Localization using Grad-CAM

A comprehensive deep learning project for automated skin lesion classification with explainable AI visualization using Grad-CAM. This project combines multimodal data (dermoscopic images + clinical features) for accurate diagnosis and provides interpretable results through attention heatmaps.

## 🌟 Features

- 🔬 **Multimodal Classification**: Combines dermoscopic images, MONET clinical features, and patient metadata
- 🔥 **Grad-CAM Visualization**: Explainable AI showing which image regions influence predictions
- 📊 **Interactive Demo**: Streamlit web application for real-time classification and visualization
- 🎯 **11-Class Classification**: Comprehensive skin lesion type detection
- 📈 **High Performance**: EfficientNet-B3 backbone with multimodal fusion architecture

## 📋 Supported Lesion Classes

| Class | Full Name | Description |
|-------|-----------|-------------|
| **AKIEC** | Actinic Keratoses | Precancerous skin lesions |
| **BCC** | Basal Cell Carcinoma | Most common skin cancer |
| **BEN_OTH** | Benign Other | Non-cancerous lesions |
| **BKL** | Benign Keratosis-like | Harmless skin growths |
| **DF** | Dermatofibroma | Benign fibrous nodules |
| **INF** | Inflammatory | Inflammatory conditions |
| **MAL_OTH** | Malignant Other | Other skin cancers |
| **MEL** | Melanoma | Most dangerous skin cancer |
| **NV** | Melanocytic Nevus | Common moles |
| **SCCKA** | Squamous Cell Carcinoma | Second most common skin cancer |
| **VASC** | Vascular Lesions | Blood vessel-related marks |

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ 
- CUDA-compatible GPU (recommended) or CPU
- Git (for cloning)

### 1. Clone Repository
```bash
git clone https://github.com/ChintaSuryaTeja/Lesion-Classification-Localization-using-GradCAM.git
cd Lesion-Classification-Localization-using-GradCAM
```

### 2. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
**Important**: You need to obtain the trained model file `best_multimodal_effb3.pth` and place it in the project root directory. This file is not included in the repository due to size constraints.

### 5. Run the Streamlit Demo

#### Option A: Using Scripts (Windows)
```cmd
# Batch file
run_demo.bat

# Or PowerShell
.\run_demo.ps1
```

#### Option B: Direct Command
```bash
streamlit run app.py
```

#### Option C: Using Python Module
```bash
python -m streamlit run app.py
```

### 6. Access the Application
- The app will automatically open in your browser
- Default URL: `http://localhost:8501`
- Upload dermoscopic images and view real-time classification results!

## How to Use

1. **Upload Image**: Click "Choose a dermoscopic image..." and select a JPG, JPEG, or PNG file
2. **Wait for Analysis**: The model will process the image and generate predictions
3. **View Results**: 
   - See the original image and Grad-CAM overlay
   - Check the predicted class and confidence level
   - Review probability scores for all classes
4. **Interpret Grad-CAM**: Red areas in the overlay indicate regions that most influenced the prediction

## Model Architecture

The demo uses a **MultimodalEfficientNetB3** model that combines:
- **Image features** from EfficientNet-B3 backbone
- **MONET clinical features** (7 morphological descriptors)
- **Metadata** (age, sex, skin tone, anatomical site)

*Note: In this demo, default values are used for MONET and metadata features since they're not available from image upload alone.*

## Important Disclaimers

⚠️ **FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY**

- This is NOT a medical device or diagnostic tool
- Do NOT use for actual medical diagnosis
- Always consult qualified healthcare professionals
- The model uses simplified default clinical features in demo mode

## Technical Requirements

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- Web browser with JavaScript enabled
- Trained model file (`best_multimodal_effb3.pth`)

## 🔧 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| 🚫 **Model file not found** | Download `best_multimodal_effb3.pth` and place in project root |
| ⚡ **CUDA errors** | App automatically falls back to CPU processing |
| 💾 **Memory errors** | Use smaller images or restart application |
| 📦 **Import errors** | Ensure virtual environment is activated: `pip install -r requirements.txt` |
| 🌐 **Streamlit not found** | Run: `pip install streamlit` in activated environment |
| 🔗 **Port already in use** | Streamlit will automatically use next available port |

### 🆘 Getting Help

1. ✅ **Check Prerequisites**: Virtual environment activated, all dependencies installed
2. 📁 **Verify Files**: Model file present and accessible  
3. 🖥️ **Check Console**: Review terminal/console for detailed error messages
4. 🐛 **Debug Mode**: Add `--logger.level debug` to streamlit command
5. 💬 **Report Issues**: Create GitHub issue with error details and system info

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Lesion-Classification-Localization-using-GradCAM.git
cd Lesion-Classification-Localization-using-GradCAM

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Contribution Areas
- 🎨 **UI/UX Improvements**: Enhance Streamlit interface
- 🧠 **Model Enhancements**: Implement new architectures or training techniques  
- 📊 **Visualization**: Add new interpretation methods (ScoreCAM, LayerCAM)
- 🔬 **Medical Features**: Integrate additional clinical data modalities
- 📚 **Documentation**: Improve guides, add tutorials
- 🐛 **Bug Fixes**: Report and fix issues

### Pull Request Process
1. 🔀 Create feature branch: `git checkout -b feature/amazing-feature`
2. 💻 Make changes and test thoroughly
3. 📝 Update documentation if needed
4. ✅ Ensure code follows project style
5. 📤 Submit pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MILK10k Dataset**: International Skin Imaging Collaboration (ISIC)
- **EfficientNet**: Google Research team
- **Grad-CAM**: Selvaraju et al. (2017)
- **Streamlit**: Amazing framework for ML web apps

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{lesion-classification-gradcam-2025,
  title={Skin Lesion Classification and Localization using Grad-CAM},
  author={Chinta Surya Teja},
  year={2025},
  url={https://github.com/ChintaSuryaTeja/Lesion-Classification-Localization-using-GradCAM}
}
```

---

**⚠️ Medical Disclaimer**: This software is for research and educational purposes only. It is not intended for medical diagnosis or clinical decision-making. Always consult qualified healthcare professionals for medical advice.

## 🏗️ Project Structure

```
Lesion-Classification-Localization-using-GradCAM/
├── 📱 Frontend & Demo
│   ├── app.py                 # Streamlit web application
│   ├── run_demo.bat          # Windows batch launcher
│   └── run_demo.ps1          # PowerShell launcher
│
├── 🧠 Model & Training
│   ├── model.py              # MultimodalEfficientNetB3 architecture
│   ├── train.py              # Training script
│   └── best_multimodal_effb3.pth  # Trained model weights (download required)
│
├── 📊 Analysis & Notebooks
│   ├── train.ipynb           # Training notebook
│   ├── dataset.ipynb         # Data preprocessing
│   └── gradCam.ipynb         # Grad-CAM analysis
│
├── 📁 Data & Outputs
│   ├── dataset/              # MILK10k dataset (not included)
│   ├── gradcam_outputs/      # Generated visualizations
│   ├── scorecam_outputs/     # Alternative CAM outputs
│   └── *.csv                 # Processed dataset files
│
├── ⚙️ Configuration
│   ├── requirements.txt      # Python dependencies
│   ├── .gitignore           # Git ignore rules
│   └── README.md            # This documentation
```

## 🧬 Model Architecture

The **MultimodalEfficientNetB3** combines three data streams:

### 🖼️ Image Branch
- **Backbone**: EfficientNet-B3 (ImageNet pretrained)
- **Input**: 300×300 RGB dermoscopic images
- **Output**: 512-dimensional image features

### 🏥 Clinical Branch (MONET)
- **Features**: 7 morphological descriptors
  - Ulceration/crust, Hair, Vasculature, Erythema
  - Pigmentation, Gel/fluid, Skin markings
- **Architecture**: MLP (7 → 128 features)

### 👤 Metadata Branch  
- **Features**: Patient demographics & lesion location
  - Age, Sex, Skin tone, Anatomical site
- **Architecture**: MLP (4 → 32 features)

### 🔗 Fusion Layer
- **Input**: Concatenated features (512 + 128 + 32 = 672)
- **Output**: 11-class probability distribution
- **Loss**: Class-weighted CrossEntropy (handles imbalanced data)

## 🎯 Training Details

- **Dataset**: MILK10k (5,000+ training samples)
- **Optimization**: AdamW (lr=2e-4)
- **Regularization**: Mixed precision training, dropout, data augmentation
- **Hardware**: CUDA GPU recommended
- **Epochs**: 20 with early stopping

## Model Performance

The model was trained on the MILK10k dataset and uses:
- Multimodal fusion architecture
- Class-weighted loss for imbalanced data
- Mixed precision training
- Grad-CAM for interpretability

For detailed model information, see the training scripts and notebooks in the project.