import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

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
