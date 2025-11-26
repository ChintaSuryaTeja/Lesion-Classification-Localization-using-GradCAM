import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
TRAIN_CSV = "train_balanced_clean.csv"
VAL_CSV   = "val_clean.csv"
CHECKPOINT_PATH = "best_multimodal_effb3.pth"

IMAGE_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-4
NUM_WORKERS = 0  # Windows/Jupyter safe
USE_AMP = True   # Mixed precision

# ============================================================
# DEVICE
# ============================================================
assert torch.cuda.is_available(), "CUDA GPU NOT FOUND!"
device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))

# ============================================================
# SEED FIX
# ============================================================
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

# ============================================================
# DATASET
# ============================================================
class LesionMultimodalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.monet_cols = [c for c in df.columns if c.startswith("MONET_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------
        # IMAGE
        # -------------------
        img_path = row["image_path"].replace("\\", "/")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # -------------------
        # MONET FEATURES
        # -------------------
        monet = torch.tensor(
            row[self.monet_cols].astype("float32").values,
            dtype=torch.float32
        )

        # -------------------
        # METADATA VECTOR
        # -------------------
        meta = torch.tensor(eval(row["metadata_vector"]), dtype=torch.float32)

        # -------------------
        # LABEL
        # -------------------
        label_map = {
            "AKIEC":0,"BCC":1,"BEN_OTH":2,"BKL":3,"DF":4,
            "INF":5,"MAL_OTH":6,"MEL":7,"NV":8,"SCCKA":9,"VASC":10
        }
        label = torch.tensor(label_map[row["target"]], dtype=torch.long)

        return {"image": image, "monet": monet, "meta": meta, "label": label}


# ============================================================
# MODEL
# ============================================================
class MultimodalEfficientNetB3(nn.Module):
    def __init__(self, monet_dim, meta_dim=4, num_classes=11):
        super().__init__()

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        eff = efficientnet_b3(weights=weights)
        in_feats = eff.classifier[1].in_features
        eff.classifier = nn.Identity()
        self.image_backbone = eff

        self.image_proj = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )

        self.monet_mlp = nn.Sequential(
            nn.Linear(monet_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1)
        )

        fusion_in = 512 + 128 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, monet, meta):
        x1 = self.image_backbone(image)
        x1 = self.image_proj(x1)
        x2 = self.monet_mlp(monet)
        x3 = self.meta_mlp(meta)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.fusion(x)


# ============================================================
# LOAD CSV + CLEAN MONET
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

monet_cols = [c for c in train_df.columns if c.startswith("MONET_")]
for c in monet_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    val_df[c]   = pd.to_numeric(val_df[c], errors="coerce")

train_df[monet_cols] = train_df[monet_cols].fillna(0).astype("float32")
val_df[monet_cols]   = val_df[monet_cols].fillna(0).astype("float32")

print("Train size:", len(train_df))
print("Val size:", len(val_df))


# ============================================================
# DATALOADERS
# ============================================================
train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1,0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = LesionMultimodalDataset(train_df, train_tf)
val_ds   = LesionMultimodalDataset(val_df, val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# ============================================================
# MODEL + LOSS + OPTIMIZER
# ============================================================
model = MultimodalEfficientNetB3(monet_dim=len(monet_cols)).to(device)

# Class weights you computed earlier
weights_dict = {
 'AKIEC':1.5721572,'BCC':0.18888328,'BEN_OTH':10.82644628,'BKL':0.87566845,
 'DF':9.16083916,'INF':9.52727272,'MAL_OTH':52.92929292,'MEL':1.05858585,
 'NV':0.63855715,'SCCKA':1.00711128,'VASC':10.13539651
}
class_order = ["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]
class_w = torch.tensor([weights_dict[c] for c in class_order], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


# ============================================================
# TRAINING LOOP
# ============================================================
best_val_acc = 0

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for batch in pbar:
        imgs  = batch["image"].to(device)
        monet = batch["monet"].to(device)
        meta  = batch["meta"].to(device)
        lbls  = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(imgs, monet, meta)
            loss = criterion(logits, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(1)
        train_correct += (preds == lbls).sum().item()
        train_total += lbls.size(0)

        pbar.set_postfix({"loss": loss.item()})

    train_acc = train_correct / train_total


    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

    with torch.no_grad():
        for batch in pbar:
            imgs  = batch["image"].to(device)
            monet = batch["monet"].to(device)
            meta  = batch["meta"].to(device)
            lbls  = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(imgs, monet, meta)

            preds = logits.argmax(1)
            val_correct += (preds == lbls).sum().item()
            val_total += lbls.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc
        }, CHECKPOINT_PATH)

        print(f"🔥 Saved new best model → {CHECKPOINT_PATH}")

print("\nTraining Complete.")
print("Best Validation Accuracy:", best_val_acc)
