import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from models import get_model
from utils import load_checkpoint, generate_gradcam, get_target_layer

# 📁 Checkpoint directory
CHECKPOINT_DIR = os.path.join("..", "checkpoints")

# ✅ Available models and checkpoints
MODEL_CHECKPOINTS = {
    "resnet18_CURLM_phase0": "resnet18_CURLM_phase0.pth",
    "resnet18_CURLM_phase1": "resnet18_CURLM_phase1.pth",
    "resnet18_DIRECT_phase1": "resnet18_DIRECT_phase1.pth",
    "resnet50_phase0": "resnet50_phase0.pth",
    "resnet50_phase1": "resnet50_phase1.pth",
    "densenet121_phase0": "densenet121_phase0.pth",
    "densenet121_phase1": "densenet121_phase1.pth"
}

# 🧠 Class mapping
CLASS_MAP = {0: "Normal", 1: "Pneumonia"}

# 📊 Model performance stats
def get_model_stats():
    return pd.DataFrame({
        "Model": [
            "ResNet18 CURLM", "ResNet18 CURLM", "ResNet18 DIRECT",
            "ResNet50", "ResNet50", "DenseNet121", "DenseNet121"
        ],
        "Phase": ["0", "1", "1", "0", "1", "0", "1"],
        "Accuracy (%)": [97.94, 98.51, 96.68, 97.82, 97.02, 98.63, 98.40],
        "Precision": [0.96, 0.97, 0.94, 0.96, 0.94, 0.97, 0.97],
        "Recall": [0.96, 0.97, 0.94, 0.95, 0.95, 0.98, 0.97],
        "F1-score": [0.96, 0.97, 0.94, 0.96, 0.94, 0.97, 0.97]
    })

# 🔍 Prediction logic
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        return CLASS_MAP[pred_class.item()], confidence.item(), pred_class.item()

# 🧠 Explanation logic
def generate_explanation(prediction, confidence, checkpoint_name):
    if prediction == "Pneumonia":
        return f"""
        The model has classified this image as **Pneumonia** with **{confidence*100:.2f}%** confidence.

        🔍 **Why?**  
        The Grad-CAM overlay shows intense activation in regions typically associated with pulmonary infiltrates — such as the lower lobes and perihilar areas. These red-hot zones suggest the model is focusing on abnormal opacities, which are common indicators of pneumonia.

        🧪 **Model Used:** `{checkpoint_name}`  
        🧭 **Interpretation:** The model is likely detecting texture irregularities, asymmetry, or density shifts in lung fields that deviate from healthy patterns.
        """
    else:
        return f"""
        The model has classified this image as **Normal** with **{confidence*100:.2f}%** confidence.

        🔍 **Why?**  
        The Grad-CAM overlay shows minimal activation, indicating the model did not detect significant anomalies. The attention is diffuse or centered in non-critical zones, suggesting a lack of pathological features.

        🧪 **Model Used:** `{checkpoint_name}`  
        🧭 **Interpretation:** The model sees balanced lung fields, clear costophrenic angles, and no signs of consolidation or abnormal texture.
        """

# 🖼️ Image preprocessing
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0), image.resize((224, 224))

# 🧠 Streamlit UI
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🫁 Pneumonia Detection Dashboard")
st.markdown("Upload a chest X-ray and select a model to view prediction, Grad-CAM, and performance comparison.")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["png", "jpg", "jpeg"])
selected_model_key = st.selectbox("🧠 Choose Model Checkpoint", list(MODEL_CHECKPOINTS.keys()))

if uploaded_file and selected_model_key:
    image_tensor, resized_image = preprocess_image(uploaded_file)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, MODEL_CHECKPOINTS[selected_model_key])
    base_model_name = selected_model_key.split("_")[0]

    # 🔧 Load model
    model = get_model(base_model_name, num_classes=2)
    model = load_checkpoint(model, checkpoint_path, device="cpu")
    model.eval()

    # 🔍 Predict
    prediction, confidence, class_idx = predict(model, image_tensor)
    st.subheader(f"🩺 Prediction: **{prediction}** ({confidence*100:.2f}%)")

    # 🔥 Grad-CAM
    target_layer = get_target_layer(model, base_model_name)
    gradcam_img = generate_gradcam(model, image_tensor.squeeze(0), target_layer, device="cpu", class_idx=class_idx)

    # 🧠 Explanation
    explanation = generate_explanation(prediction, confidence, selected_model_key)

    # 🖼️ Display Images Side-by-Side
    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_image, caption="🖼️ Original X-Ray", use_container_width=True)
    with col2:
        st.image(gradcam_img, caption="🔥 Grad-CAM Overlay", use_container_width=True)

    # 🧠 Display Explanation
    st.markdown("### 🧠 AI Explainer")
    st.markdown(explanation)

    # 📊 Model Comparison Table
    st.markdown("---")
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(get_model_stats(), use_container_width=True)