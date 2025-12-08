# models.py
# Hybrid CNN -> Transformer classifier for ECG segments
import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, channels=[32, 64, 128]):
        super().__init__()

        # -----------------------------
        # First block EXPOSED FOR XAI
        # -----------------------------
        self.first_conv_block = nn.Sequential(
            nn.Conv1d(in_ch, channels[0], kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        layers = [self.first_conv_block]
        prev = channels[0]

        # Remaining blocks
        for c in channels[1:]:
            layers += [
                nn.Conv1d(prev, c, kernel_size=7, padding=3, stride=1),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ]
            prev = c

        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        return self.net(x)      # (B, C, T//8)



class CNNTransformerClassifier(nn.Module):
    def __init__(self, num_classes=7, cnn_channels=[32, 64, 128],
                 nhead=8, num_layers=2, dim_feedforward=256):
        super().__init__()

        self.cnn = SmallCNN(in_ch=1, channels=cnn_channels)

        # 1×1 Conv → Transformer embedding dimension 128
        self.post_conv_proj = nn.Conv1d(self.cnn.out_dim,
                                        128,
                                        kernel_size=1)

        # -----------------------------
        # RENAME FOR XAI COMPATIBILITY
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, L)
        z = self.cnn(x)                # (B, C, T')
        z = self.post_conv_proj(z)     # (B, 128, T')
        z = z.permute(0, 2, 1)         # (B, T', 128)
        z = self.transformer_encoder(z)# (B, T', 128)
        z = z.mean(dim=1)              # Global average → (B, 128)
        return self.classifier(z)      # (B, num_classes)
