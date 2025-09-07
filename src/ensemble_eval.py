import os
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc
)

from data_loader import get_data_loaders
from utils import (
    set_seed, load_checkpoint, plot_confusion_matrix,
    generate_gradcam, visualize_prediction, get_target_layer
)
from models import get_model

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ----------------------------
# Thresholding
# ----------------------------
def find_optimal_thresholds(y_true, y_probs):
    fpr, tpr, roc_thresh = roc_curve(y_true, y_probs)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_probs)

    youden_j = np.argmax(tpr - fpr)
    t_youden = roc_thresh[youden_j]

    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    t_f1 = pr_thresh[np.argmax(f1_scores)]

    spec = 1 - fpr
    idx_spec95 = np.argmax(spec >= 0.95)
    t_spec95 = roc_thresh[idx_spec95] if idx_spec95 < len(roc_thresh) else 0.5

    return {
        "youden": float(t_youden),
        "f1_opt": float(t_f1),
        "spec95": float(t_spec95),
        "fixed_0.5": 0.5
    }

# ----------------------------
# Ensemble Evaluation
# ----------------------------
def evaluate_ensemble(ckpt_paths, device, test_loader, class_names, output_dir):
    # Load all models
    models = []
    for ckpt_path in ckpt_paths:
        arch = "resnet18" if "resnet18" in ckpt_path else (
            "resnet50" if "resnet50" in ckpt_path else "densenet121"
        )
        model = get_model(arch, num_classes=len(class_names))
        model = load_checkpoint(model, ckpt_path, device)
        model.to(device)
        model.eval()
        models.append((arch, model))

    y_true, y_probs, y_pred = [], [], []
    seen_classes = set()

    for img, label in test_loader:
        img, label = img.to(device), label.to(device)

        # Collect probabilities from each model
        probs = []
        with torch.no_grad():
            for arch, model in models:
                out = model(img)
                prob = torch.softmax(out, dim=1)[:, 1].item()
                probs.append(prob)

        # Ensemble probability (average)
        final_prob = np.mean(probs)
        final_pred = int(final_prob >= 0.5)

        y_true.append(label.item())
        y_probs.append(final_prob)
        y_pred.append(final_pred)

        # Save one visualization per class (with Grad-CAM)
        if label.item() not in seen_classes:
            arch, model = models[0]  # pick first model for visualization
            target_layer = get_target_layer(model, arch)

            # ⚡ re-enable gradients only for Grad-CAM
            with torch.enable_grad():
                gradcam_img = generate_gradcam(
                    model, img.squeeze(), target_layer,
                    device, class_idx=final_pred,
                    save_path=os.path.join(
                        output_dir,
                        f"gradcam_true{class_names[label]}_pred{class_names[final_pred]}.png"
                    )
                )

            fig = visualize_prediction(
                img.squeeze(), label.item(), final_pred,
                class_names, gradcam_img,
                title=f"Ensemble | True={class_names[label]} Pred={class_names[final_pred]}"
            )
            fig_path = os.path.join(
                output_dir,
                f"viz_true{class_names[label]}_pred{class_names[final_pred]}.png"
            )
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)

            logging.info(f"🖼️ Viz saved: {fig_path}")
            seen_classes.add(label.item())

    # Metrics
    y_true, y_probs, y_pred = np.array(y_true), np.array(y_probs), np.array(y_pred)
    thresholds = find_optimal_thresholds(y_true, y_probs)

    results = {}
    for tname, tval in thresholds.items():
        preds = (y_probs >= tval).astype(int)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(rec_curve, prec_curve)

        results[tname] = {
            "threshold": tval,
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1,
            "roc_auc": roc_auc, "pr_auc": pr_auc
        }

        cm_path = os.path.join(output_dir, f"confusion_{tname}.png")
        plot_confusion_matrix(y_true, preds, class_names, save_path=cm_path)
        logging.info(f"📊 Confusion matrix saved: {cm_path}")

    return {"model": "ensemble", "metrics": results}

# ----------------------------
# Main
# ----------------------------
def main():
    logging.info("🚀 Starting ensemble evaluation...")
    set_seed(42)

    data_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/DATA"
    ckpt_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/checkpoints"
    output_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/src/outputs_eval"
    os.makedirs(output_dir, exist_ok=True)

    loaders = get_data_loaders(data_dir, batch_size=1, img_size=224, visualize=False)
    test_loader = loaders["test"]
    class_names = loaders["class_names"]

    ckpt_paths = [
        os.path.join(ckpt_dir, "densenet121_phase1.pth"),
        os.path.join(ckpt_dir, "densenet121_phase0.pth"),
        os.path.join(ckpt_dir, "resnet18_CURLM_phase1.pth")
    ]

    results = evaluate_ensemble(
        ckpt_paths, device=torch.device("cpu"),
        test_loader=test_loader, class_names=class_names,
        output_dir=output_dir
    )

    # Save results in outputs_eval with fixed name
    summary_path = os.path.join(output_dir, "ensemble_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"✅ Ensemble metrics saved to {summary_path}")

if __name__ == "__main__":
    main()
