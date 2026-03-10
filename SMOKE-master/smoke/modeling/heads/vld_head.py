import torch
import torch.nn as nn
import torch.nn.functional as F

class VLDHead(nn.Module):
    def __init__(self, in_channels=256, hidden=128):
        super(VLDHead, self).__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1)
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, 1)
        )

        self.heatmap_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.shared(x)

        heatmap = torch.sigmoid(self.heatmap_head(feat))
        heatmap = torch.clamp(heatmap, 1e-4, 1-1e-4)

        reg = self.reg_head(feat)

        du_dv = reg[:, 0:2]
        dh_dw = torch.exp(reg[:, 2:4])

        regression = torch.cat([du_dv, dh_dw], dim=1)

        return heatmap, regression
