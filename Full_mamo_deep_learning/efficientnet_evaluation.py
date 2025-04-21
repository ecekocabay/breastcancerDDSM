import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from common_utils import MammogramDataset
from torchvision import transforms
from model import get_model_and_loader  # assumes get_model(model_name) returns the architecture
from tqdm import tqdm

# CONFIG
BASE_DIR = "/Users/ecekocabay/Desktop/2025SPRING/ CNG492/DDSM"
CSV_PATH = os.path.join(BASE_DIR, "data/full_mammogram_paths.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/full_mamo_deep_learning/efficientnet_full_mammo.pth")  # or efficientnet.pth, vit.pth
MODEL_NAME = "efficientnet"  # "efficientnet" or "vit"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MAIN EVALUATION LOGIC ===
def evaluate():
    print(f"📊 Loading model and data for: {MODEL_NAME}")
    model, _, test_loader = get_model_and_loader(MODEL_NAME, BASE_DIR, batch_size=16)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="🔍 Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n✅ Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["benign", "malignant"]))
    print(f"✅ Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)  # Recommended for macOS
    evaluate()