import os
import re
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
# Thresholding utilities
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
# Evaluation
# ----------------------------
def evaluate_model(ckpt_path, device, test_loader, class_names, output_dir):
    fname = os.path.basename(ckpt_path)
    match = re.match(r"(resnet18|resnet50|densenet121).*?(phase\d+)", fname)
    if not match:
        logging.warning(f"⚠️ Could not parse {fname}, skipping...")
        return None
    arch, phase = match.groups()

    model = get_model(arch, num_classes=len(class_names))
    model = load_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    out_dir = os.path.join(output_dir, f"{arch}_{phase}")
    os.makedirs(out_dir, exist_ok=True)

    y_true, y_probs, y_pred = [], [], []
    seen_classes = set()

    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            prob = torch.softmax(out, dim=1)[:, 1].item()
            pred = torch.argmax(out, dim=1).item()

            y_true.append(label.item())
            y_probs.append(prob)
            y_pred.append(pred)                                                                         

            if label.item() not in seen_classes:
                with torch.enable_grad():
                    target_layer = get_target_layer(model, arch)
                    gradcam_img = generate_gradcam(
                        model, img.squeeze(), target_layer,
                        device, class_idx=pred,
                        save_path=os.path.join(out_dir, f"gradcam_true{class_names[label]}_pred{class_names[pred]}.png")
                    )

                # ⬇️ Use visualize_prediction but save manually
                fig = visualize_prediction(
                    img.squeeze(), label.item(), pred,
                    class_names, gradcam_img,
                    title=f"{arch}_{phase} | True={class_names[label]} Pred={class_names[pred]}"
                )
                fig_path = os.path.join(out_dir, f"viz_true{class_names[label]}_pred{class_names[pred]}.png")
                plt.savefig(fig_path, bbox_inches="tight")
                plt.close(fig)

                logging.info(f"🖼️ Viz saved to {fig_path}")
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
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc
        }

        cm_path = os.path.join(out_dir, f"confusion_{tname}.png")
        plot_confusion_matrix(y_true, preds, class_names, save_path=cm_path)

    return {"model": arch, "phase": phase, "metrics": results}

# ----------------------------
# Main
# ----------------------------
def main():
    logging.info("🚀 Starting evaluation pipeline...")
    set_seed(42)

    data_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/DATA"
    ckpt_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/checkpoints"
    output_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/src/outputs_eval"
    os.makedirs(output_dir, exist_ok=True)

    logging.info("📁 Initializing data loaders...")
    loaders = get_data_loaders(data_dir, batch_size=1, img_size=224, visualize=False)
    test_loader = loaders["test"]
    class_names = loaders["class_names"]
    logging.info(f"🧪 Classes: {class_names}")

    summary = []
    for fname in os.listdir(ckpt_dir):
        if fname.endswith(".pth"):
            ckpt_path = os.path.join(ckpt_dir, fname)
            logging.info(f"🚀 Evaluating {os.path.splitext(fname)[0]}")
            res = evaluate_model(ckpt_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                 test_loader=test_loader, class_names=class_names, output_dir=output_dir)
            if res:
                summary.append(res)

    summary_path = os.path.join(output_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    logging.info(f"✅ Saved summary metrics to {summary_path}")

if __name__ == "__main__":
    main()
