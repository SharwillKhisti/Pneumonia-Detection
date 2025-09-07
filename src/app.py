import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

# -------------------------------
# Imports from your modules
# -------------------------------
from models import get_model
from utils import load_checkpoint, generate_gradcam, get_target_layer
from explainers.integrated_gradients import get_integrated_gradients, show_ig_overlay
from explainers.counterfactuals import mask_region, visualize_masked_input
from explainers.natural_explainer import explain_naturally

# -------------------------------
# Constants and Config
# -------------------------------
CHECKPOINT_DIR = os.path.join("..", "checkpoints")
MODEL_CHECKPOINTS = {
    "resnet18_CURLM_phase0": "resnet18_CURLM_phase0.pth",
    "resnet18_CURLM_phase1": "resnet18_CURLM_phase1.pth",
    "resnet18_DIRECT_phase1": "resnet18_DIRECT_phase1.pth",
    "resnet50_phase0": "resnet50_phase0.pth",
    "resnet50_phase1": "resnet50_phase1.pth",
    "densenet121_phase0": "densenet121_phase0.pth",
    "densenet121_phase1": "densenet121_phase1.pth"
}
CLASS_MAP = {0: "Normal", 1: "Pneumonia"}

# -------------------------------
# Utility Functions
# -------------------------------
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

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        return CLASS_MAP[pred_class.item()], confidence.item(), pred_class.item(), probs

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), image.resize((224, 224))

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Pneumonia Detection Suite", layout="wide")
st.title("🫁 Pneumonia Detection Suite")

tab1, tab2 = st.tabs(["📷 Prediction Dashboard", "📊 Threshold Comparison"])

# -------------------------------
# Tab 1: Prediction Dashboard
# -------------------------------
with tab1:
    st.markdown("Upload a chest X-ray and select a model to view prediction, Grad-CAM, Integrated Gradients, Counterfactuals, and performance comparison.")

    uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["png", "jpg", "jpeg"])
    selected_model_key = st.selectbox("🧠 Choose Model Checkpoint", list(MODEL_CHECKPOINTS.keys()))

    if uploaded_file and selected_model_key:
        image_tensor, resized_image = preprocess_image(uploaded_file)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, MODEL_CHECKPOINTS[selected_model_key])
        base_model_name = selected_model_key.split("_")[0]

        model = get_model(base_model_name, num_classes=2)
        model = load_checkpoint(model, checkpoint_path, device="cpu")
        model.eval()

        prediction, confidence, class_idx, probs = predict(model, image_tensor)
        st.subheader(f"🩺 Prediction: **{prediction}** ({confidence*100:.2f}%)")
        st.write("🔍 Raw Probabilities:", {CLASS_MAP[i]: f"{p*100:.2f}%" for i, p in enumerate(probs.squeeze().tolist())})

        target_layer = get_target_layer(model, base_model_name)
        gradcam_img = generate_gradcam(model, image_tensor.squeeze(0), target_layer, device="cpu", class_idx=class_idx)

        ig_attr = get_integrated_gradients(model, image_tensor, class_idx)
        ig_heatmap, ig_overlay = show_ig_overlay(ig_attr, resized_image)

        masked_tensor = mask_region(image_tensor, region="center", size=50)
        masked_pred, masked_conf, _, _ = predict(model, masked_tensor)
        masked_visual = visualize_masked_input(resized_image, region="center", size=50)

        explanation = explain_naturally(prediction, confidence, selected_model_key)

        tabA, tabB, tabC = st.tabs(["Grad-CAM", "Integrated Gradients", "Counterfactual"])

        with tabA:
            col1, col2 = st.columns(2)
            col1.image(resized_image, caption="🖼️ Original X-Ray",width=500)
            col2.image(gradcam_img, caption="🔥 Grad-CAM Overlay", width=500)

        with tabB:
            col1, col2 = st.columns(2)
            col1.image(resized_image, caption="🖼️ Original X-Ray",width=500)
            col2.image(ig_overlay, caption="🧠 Integrated Gradients", width=500)

        with tabC:
            col1, col2 = st.columns(2)
            col1.image(resized_image, caption="🖼️ Original X-Ray",width=500)
            col2.image(masked_visual, caption="🧪 Masked Region", width=500)
            st.markdown(f"### 🧪 Counterfactual Prediction: **{masked_pred}** ({masked_conf*100:.2f}%)")

        st.markdown("### 🧠 AI Explainer")
        st.markdown(explanation)

        st.markdown("### 🧪 What Do These Explainability Methods Mean?")
        with st.expander("ℹ️ Grad-CAM"):
            st.markdown("""
            Grad-CAM (Gradient-weighted Class Activation Mapping) highlights regions in the image that most influenced the model's prediction. 
            It uses gradients flowing into the last convolutional layer to produce a heatmap over the input image.
            """)

        with st.expander("ℹ️ Integrated Gradients"):
            st.markdown("""
            Integrated Gradients attribute the prediction to each input feature by averaging gradients as the input is scaled from a baseline (like a black image) to the actual image. 
            This helps identify which pixels contributed most to the model's decision.
            """)

        with st.expander("ℹ️ Counterfactuals"):
            st.markdown("""
            Counterfactual explanations show how the prediction would change if certain regions of the image were altered or masked. 
            This helps assess the robustness of the model and understand which areas are critical for its decision.
            """)

# Build Youden accuracy comparison table
results = [
    {
        "model": "densenet121",
        "phase": "phase0",
        "metrics": {
            "youden": {
                "threshold": 0.5385,
                "accuracy": 0.9863,
                "precision": 0.9953,
                "recall": 0.9859,
                "f1": 0.9906,
                "roc_auc": 0.9974,
                "pr_auc": 0.9990
            },
            "f1_opt": {
                "threshold": 0.2259,
                "accuracy": 0.9874,
                "precision": 0.9906,
                "recall": 0.9922,
                "f1": 0.9914,
                "roc_auc": 0.9974,
                "pr_auc": 0.9990
            },
            "spec95": {
                "threshold": float("inf"),
                "accuracy": 0.2680,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.9974,
                "pr_auc": 0.9990
            },
            "fixed_0.5": {
                "threshold": 0.5,
                "accuracy": 0.9863,
                "precision": 0.9937,
                "recall": 0.9875,
                "f1": 0.9906,
                "roc_auc": 0.9974,
                "pr_auc": 0.9990
            }
        }
    },
    {
        "model": "densenet121",
        "phase": "phase1",
        "metrics": {
            "youden": {
                "threshold": 0.9355,
                "accuracy": 0.9874,
                "precision": 0.9968,
                "recall": 0.9859,
                "f1": 0.9913,
                "roc_auc": 0.9986,
                "pr_auc": 0.9995
            },
            "f1_opt": {
                "threshold": 0.6729,
                "accuracy": 0.9874,
                "precision": 0.9937,
                "recall": 0.9890,
                "f1": 0.9914,
                "roc_auc": 0.9986,
                "pr_auc": 0.9995
            },
            "spec95": {
                "threshold": float("inf"),
                "accuracy": 0.2680,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.9986,
                "pr_auc": 0.9995
            },
            "fixed_0.5": {
                "threshold": 0.5,
                "accuracy": 0.9840,
                "precision": 0.9890,
                "recall": 0.9890,
                "f1": 0.9890,
                "roc_auc": 0.9986,
                "pr_auc": 0.9995
            }
        }
    },
    {
        "model": "resnet18",
        "phase": "phase0",
        "metrics": {
            "youden": {
                "threshold": 0.6842860579490662,
                "accuracy": 0.9782359679266895,
                "precision": 0.9889589905362776,
                "recall": 0.9812206572769953,
                "f1": 0.9850746268656716,
                "roc_auc": 0.9947701403100464,
                "pr_auc": 0.9979617006731292
            },
            "f1_opt": {
                "threshold": 0.35206830501556396,
                "accuracy": 0.979381443298969,
                "precision": 0.984399375975039,
                "recall": 0.9874804381846636,
                "f1": 0.9859375,
                "roc_auc": 0.9947701403100464,
                "pr_auc": 0.9979617006731292
            },
            "spec95": {
                "threshold":float("inf"),
                "accuracy": 0.26804123711340205,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.9947701403100464,
                "pr_auc": 0.9979617006731292
            },
            "fixed_0.5": {
                "threshold": 0.5,
                "accuracy": 0.979381443298969,
                "precision": 0.9859154929577465,
                "recall": 0.9859154929577465,
                "f1": 0.9859154929577465,
                "roc_auc": 0.9947701403100464,
                "pr_auc": 0.9979617006731292
            }
        }
    },
    {
        "model": "resnet18",
        "phase": "phase1",
        "metrics": {
            "youden": {
                "threshold": 0.7923637628555298,
                "accuracy": 0.9851088201603666,
                "precision": 0.9905956112852664,
                "recall": 0.9890453834115805,
                "f1": 0.9898198903680501,
                "roc_auc": 0.9965156561400692,
                "pr_auc": 0.9986737226807286
            },
            "f1_opt": {
                "threshold": 0.5112547278404236,
                "accuracy": 0.9851088201603666,
                "precision": 0.9890625,
                "recall": 0.9906103286384976,
                "f1": 0.9898358092259578,
                "roc_auc": 0.9965156561400692,
                "pr_auc": 0.9986737226807286
            },
            "spec95": {
                "threshold": float("inf"),
                "accuracy": 0.26804123711340205,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.9965156561400692,
                "pr_auc": 0.9986737226807286
            },
            "fixed_0.5": {
                "threshold": 0.5,
                "accuracy": 0.9851088201603666,
                "precision": 0.9890625,
                "recall": 0.9906103286384976,
                "f1": 0.9898358092259578,
                "roc_auc": 0.9965156561400692,
                "pr_auc": 0.9986737226807286
            }
        }
    },
    {
        "model": "resnet50",
        "phase": "phase0",
        "metrics": {
            "youden": {
                "threshold": 0.8519251346588135,
                "accuracy": 0.97709049255441,
                "precision": 0.9920508744038156,
                "recall": 0.9765258215962441,
                "f1": 0.9842271293375394,
                "roc_auc": 0.9956395543250003,
                "pr_auc": 0.9980779981235952
            },
            "f1_opt": {
                "threshold": 0.42429205775260925,
                "accuracy": 0.9805269186712485,
                "precision": 0.9814241486068112,
                "recall": 0.9921752738654147,
                "f1": 0.9867704280155642,
                "roc_auc": 0.9956395543250003,
                "pr_auc": 0.9980779981235952
            },
            "spec95": {
                "threshold": float("inf"),
                "accuracy": 0.26804123711340205,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.9956395543250003,
                "pr_auc": 0.9980779981235952
            },
            "fixed_0.5": {
                "threshold": 0.5,
                "accuracy": 0.9782359679266895,
                "precision": 0.9813664596273292,
                "recall": 0.9890453834115805,
                "f1": 0.9851909586905689,
                "roc_auc": 0.9956395543250003,
                "pr_auc": 0.9980779981235952
            }
        }
    },
    {
        "model": "resnet50",
        "phase": "phase1",
        "metrics": {
            "youden": {
                "threshold": 0.8946828246116638,
                "accuracy": 0.9713631156930126,
                "precision": 0.9951612903225806,
                "recall": 0.9655712050078247,
                "f1": 0.9801429706115965,
                "roc_auc": 0.9955124861228146,
                "pr_auc": 0.9984273597853457
            },
            "f1_opt": {
                "threshold": 0.28312987089157104,
                "accuracy": 0.9725085910652921,
                "precision": 0.9738058551617874,
                "recall": 0.9890453834115805,
                "f1": 0.9813664596273292,
                "roc_auc": 0.9955124861228146,
                "pr_auc": 0.9984273597853457
            },
            "spec95": {
                "threshold": float("inf"),
                "accuracy": 0.26804123711340205,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.9955124861228146,
                "pr_auc": 0.9984273597853457
            },
            "fixed_0.5": {
                "threshold": 0.5,
                "accuracy": 0.9702176403207331,
                "precision": 0.9826771653543307,
                "recall": 0.9765258215962441,
                "f1": 0.9795918367346939,
                "roc_auc": 0.9955124861228146,
                "pr_auc": 0.9984273597853457
            }
        }
    }

]

# -------------------------------
# Tab 2: Threshold Comparison
# -------------------------------
with tab2:
    st.markdown("This dashboard compares **all models** across multiple thresholds (`youden`, `f1_opt`, `spec95`, `fixed_0.5`).")
    st.markdown("Each table shows detailed metrics, and the best-performing model under the **Youden threshold** is highlighted below.")

    # Simulated results list — replace with actual data loading if needed

    youden_scores = []

    for entry in results:
        model_name = entry["model"]
        phase = entry["phase"]
        metrics = entry["metrics"]

        df = pd.DataFrame([
            {
                "Threshold Type": key,
                "Threshold": f"{val['threshold']:.4f}" if val["threshold"] != float("inf") else "∞",
                "Accuracy": val["accuracy"],
                "Precision": val["precision"],
                "Recall": val["recall"],
                "F1 Score": val["f1"],
                "ROC AUC": val["roc_auc"],
                "PR AUC": val["pr_auc"]
            }
            for key, val in metrics.items()
        ])

        st.markdown(f"#### 📌 Model: `{model_name}` | Phase: `{phase}`")
        st.dataframe(df.style.highlight_max(subset=["F1 Score"], color="lightgreen"), use_container_width=True)

        # Track Youden accuracy for comparison
        youden_scores.append({
            "model": model_name,
            "phase": phase,
            "accuracy": metrics["youden"]["accuracy"]
        })

    # Find best model under Youden threshold
    best_model = max(youden_scores, key=lambda x: x["accuracy"])
    st.markdown("---")
    st.subheader("🏆 Best Model under Youden Threshold")
    st.markdown(f"**Model:** `{best_model['model']}` | **Phase:** `{best_model['phase']}`")
    st.markdown(f"**Accuracy:** `{best_model['accuracy']*100:.2f}%`")

    model_labels = [f"{entry['model']} ({entry['phase']})" for entry in results]
    youden_accuracies = [entry["metrics"]["youden"]["accuracy"] * 100 for entry in results]
    fixed_accuracies = [entry["metrics"]["fixed_0.5"]["accuracy"] * 100 for entry in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    x = range(len(model_labels))

    ax.bar(x, youden_accuracies, width=bar_width, label="Youden", color="mediumseagreen")
    ax.bar([i + bar_width for i in x], fixed_accuracies, width=bar_width, label="Fixed 0.5", color="lightcoral")

    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(model_labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("📊 Accuracy Comparison: Youden vs Fixed 0.5 Threshold")
    ax.legend()
    ax.set_ylim(0, 100)

    st.pyplot(fig)

    with st.expander("📘 Why Use Youden Threshold Instead of Fixed 0.5?"):
        st.markdown("""
        The **Youden Index** identifies the optimal threshold that maximizes the sum of sensitivity and specificity.  
        This is especially important in medical diagnostics where:
        
        - A fixed threshold (like 0.5) may not reflect the true decision boundary for imbalanced datasets.
        - Youden adapts to the model’s confidence distribution, improving diagnostic reliability.
        - It reduces false negatives (missed pneumonia cases) while maintaining precision.

        In contrast, a fixed threshold assumes equal cost for false positives and false negatives—which is rarely true in healthcare.
        """)
        
    # Find best overall model by highest Youden accuracy
    best_overall = max(results, key=lambda x: x["metrics"]["youden"]["accuracy"])
    st.markdown("### 🧠 Verdict: Best Overall Model")
    st.success(f"🏅 **{best_overall['model']} (Phase {best_overall['phase']})** achieved the highest Youden accuracy of **{best_overall['metrics']['youden']['accuracy']*100:.2f}%**.")