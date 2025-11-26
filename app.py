import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import io
import base64

from model import MultimodalEfficientNetB3

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Skin Lesion Classification Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CONSTANTS
# ========================================
CHECKPOINT_PATH = "best_multimodal_effb3.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONET_DIM = 7
IMAGE_SIZE = 300

# Class names and descriptions
CLASS_NAMES = {
    0: "AKIEC", 1: "BCC", 2: "BEN_OTH", 3: "BKL", 4: "DF",
    5: "INF", 6: "MAL_OTH", 7: "MEL", 8: "NV", 9: "SCCKA", 10: "VASC"
}

CLASS_DESCRIPTIONS = {
    "AKIEC": "Actinic Keratoses - Precancerous skin lesions caused by sun damage",
    "BCC": "Basal Cell Carcinoma - Most common form of skin cancer",
    "BEN_OTH": "Benign Other - Non-cancerous skin lesions",
    "BKL": "Benign Keratosis-like Lesions - Harmless skin growths",
    "DF": "Dermatofibroma - Benign fibrous skin nodules",
    "INF": "Inflammatory - Inflammatory skin conditions",
    "MAL_OTH": "Malignant Other - Other types of skin cancer",
    "MEL": "Melanoma - Most dangerous form of skin cancer",
    "NV": "Melanocytic Nevus - Common moles",
    "SCCKA": "Squamous Cell Carcinoma - Second most common skin cancer",
    "VASC": "Vascular Lesions - Blood vessel-related skin marks"
}

# ========================================
# GRAD-CAM IMPLEMENTATION
# ========================================
# Use the exact same GradCAM implementation as your working notebook
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor, monet, meta):
        # Forward pass
        logits = self.model(img_tensor, monet, meta)
        pred_class = logits.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        logits[0, pred_class].backward()

        # Get gradients and activations (same as notebook)
        gradients = self.gradients[0]       # [C,H,W]
        activations = self.activations[0]   # [C,H,W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1,2))  # GAP

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(DEVICE)
        for w, act in zip(weights, activations):
            cam += w * act

        # ReLU and normalize (same as notebook)
        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        return cam, pred_class, probabilities

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = MultimodalEfficientNetB3(monet_dim=MONET_DIM).to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Set to eval mode initially, but we'll switch to train mode for gradcam
        model.eval()
        
        # Ensure all parameters can compute gradients
        for param in model.parameters():
            param.requires_grad = True
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ========================================
# IMAGE PREPROCESSING
# ========================================
def preprocess_image(image):
    """Preprocess the input image"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(DEVICE)

# ========================================
# MOCK FEATURES (since we don't have real MONET/meta data)
# ========================================
def get_default_features():
    """Generate default MONET and metadata features"""
    # Default MONET features (normalized values)
    monet_features = torch.tensor([
        [0.5, 0.3, 0.2, 0.4, 0.6, 0.3, 0.4]  # Mock values
    ], dtype=torch.float32).to(DEVICE)
    
    # Default metadata: [age, sex, skin_tone, site]
    meta_features = torch.tensor([
        [50.0, 1.0, 3.0, 5.0]  # Mock values
    ], dtype=torch.float32).to(DEVICE)
    
    return monet_features, meta_features

# ========================================
# VISUALIZATION FUNCTIONS
# ========================================
def overlay_gradcam(image, cam, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image - same as notebook"""
    # Convert PIL to numpy if needed
    if not isinstance(image, np.ndarray):
        orig = np.array(image)
    else:
        orig = image.copy()
    
    # Resize CAM to image size (same as notebook)
    cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
    
    # Create heatmap (exact same as notebook)
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    # Note: notebook doesn't convert BGR to RGB, but we need it for display
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay (same blending as notebook: 0.6 + 0.4)
    overlay = (orig * 0.6 + heatmap * 0.4).astype(np.uint8)
    
    return overlay

def create_prediction_chart(probabilities):
    """Create a bar chart of class probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_labels = [CLASS_NAMES[i] for i in range(len(probabilities))]
    colors = plt.cm.viridis(probabilities / max(probabilities))
    
    bars = ax.barh(class_labels, probabilities, color=colors)
    ax.set_xlabel('Probability')
    ax.set_title('Classification Probabilities')
    ax.set_xlim(0, 1)
    
    # Add probability values on bars
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig

# ========================================
# MAIN APP
# ========================================
def main():
    # Title and description
    st.title("🔬 Skin Lesion Classification Demo")
    st.markdown("""
    This demo uses a multimodal deep learning model to classify skin lesions from dermoscopic images.
    The model combines image features with clinical descriptors (MONET) and metadata for accurate classification.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Initialize Grad-CAM using the exact same approach as the working notebook
    try:
        # Use the same target layer as your notebook: model.image_backbone.features[-1]
        target_layer = model.image_backbone.features[-1]
        st.info(f"Using target layer: {type(target_layer).__name__} (same as notebook)")
        gradcam = GradCAM(model, target_layer)
        
    except Exception as e:
        st.error(f"Error initializing Grad-CAM: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a dermoscopic image (JPG, JPEG, PNG)
    2. The model will classify the lesion type
    3. View the Grad-CAM visualization to see which areas influenced the prediction
    4. Check the confidence scores for all classes
    """)
    
    st.sidebar.header("ℹ️ About the Model")
    st.sidebar.markdown("""
    - **Architecture**: Multimodal EfficientNet-B3
    - **Input Modalities**: 
      - Dermoscopic images
      - MONET clinical features
      - Patient metadata
    - **Classes**: 11 skin lesion types
    - **Training**: MILK10k dataset
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear dermoscopic image for analysis"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded dermoscopic image", use_container_width=True)
        
        # Process image
        with st.spinner("Analyzing image..."):
            # Preprocess
            img_tensor = preprocess_image(image)
            monet_features, meta_features = get_default_features()
            
            # Generate Grad-CAM using the exact same method as your working notebook
            try:
                cam, pred_class, probabilities = gradcam.generate(
                    img_tensor, monet_features, meta_features
                )
                st.success("✅ Grad-CAM generated successfully (notebook method)")
                
            except Exception as e:
                st.error(f"Grad-CAM failed: {str(e)}")
                # Fallback to basic prediction
                with torch.no_grad():
                    logits = model(img_tensor, monet_features, meta_features)
                    pred_class = logits.argmax(dim=1).item()
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                # Create a dummy CAM
                cam = np.random.rand(50, 50)
                st.warning("Using fallback visualization")
            
            # Debug info
            st.write(f"🔍 CAM shape: {cam.shape}, range: [{cam.min():.3f}, {cam.max():.3f}]")
            
            # Convert image to numpy for visualization
            img_np = np.array(image)
            
            # Create Grad-CAM overlay (same as notebook)
            gradcam_overlay = overlay_gradcam(img_np, cam)
        
        with col2:
            st.subheader("🔥 Grad-CAM Visualization")
            st.image(gradcam_overlay, caption="Areas influencing the prediction (red = high influence)", use_container_width=True)
        
        # Display results
        st.markdown("---")
        
        # Prediction results
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("🎯 Classification Result")
            
            predicted_class = CLASS_NAMES[pred_class]
            confidence = probabilities[pred_class]
            
            # Create metric display
            st.metric(
                label="Predicted Class",
                value=predicted_class,
                delta=f"Confidence: {confidence:.1%}"
            )
            
            # Class description
            st.info(f"**{predicted_class}**: {CLASS_DESCRIPTIONS[predicted_class]}")
            
            # Confidence level indicator
            if confidence > 0.8:
                st.success("High confidence prediction")
            elif confidence > 0.6:
                st.warning("Medium confidence prediction")
            else:
                st.error("Low confidence prediction - please consult a dermatologist")
        
        with col4:
            st.subheader("📊 All Class Probabilities")
            
            # Create probability chart
            fig = create_prediction_chart(probabilities)
            st.pyplot(fig)
        
        # Additional information
        st.markdown("---")
        st.subheader("⚠️ Important Disclaimer")
        st.warning("""
        **This is a research demonstration and should NOT be used for medical diagnosis.**
        
        - Always consult qualified healthcare professionals for medical advice
        - This model is trained on research data and may not generalize to all cases
        - The demo uses simplified default values for clinical features
        - For actual clinical use, proper MONET features and patient metadata would be required
        """)
        
        # Technical details (expandable)
        with st.expander("🔧 Technical Details"):
            st.markdown(f"""
            **Model Information:**
            - Device: {DEVICE}
            - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}
            - Predicted class index: {pred_class}
            - Max probability: {max(probabilities):.4f}
            
            **Feature Information:**
            - MONET features: 7 clinical descriptors (using default values in demo)
            - Metadata: Age, sex, skin tone, anatomical site (using default values in demo)
            - Image features: Extracted by EfficientNet-B3 backbone
            """)

if __name__ == "__main__":
    main()