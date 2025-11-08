# modelResNet18.py

import torch
import torch.nn as nn
from torchvision import models

class ResNet18Custom(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        # 1️⃣ Load ResNet18 pretrained
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 2️⃣ Ubah input channel jika bukan RGB
        if in_channels != 3:
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Adaptasi bobot channel lama
            with torch.no_grad():
                self.model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # 3️⃣ Ganti fully connected layer untuk jumlah kelas kamu
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        return self.model(x)

# --- Pengujian mandiri ---
if __name__ == "__main__":
    NUM_CLASSES = 2
    IN_CHANNELS = 1

    model = ResNet18Custom(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print(model)

    dummy_input = torch.randn(4, IN_CHANNELS, 224, 224)
    out = model(dummy_input)
    print(f"\nOutput shape: {out.shape}")
    print("Pengujian model 'ResNet18' berhasil.")
