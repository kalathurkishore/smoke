import torch
import torch.nn as nn
import torch.nn.functional as F


class VLDHead(nn.Module):

    def __init__(self, in_channels=256, hidden_channels=128):

        super(VLDHead, self).__init__()

        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Heatmap head
        self.heatmap_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )

        # Regression head
        self.regression_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, 1)
        )

        # Heatmap bias initialization
        self.heatmap_branch[-1].bias.data.fill_(-2.19)

    def forward(self, x):

        features = self.base_conv(x)

        heatmap = self.heatmap_branch(features)
        heatmap = torch.sigmoid(heatmap)
        heatmap = torch.clamp(heatmap, 1e-4, 1-1e-4)

        regression = self.regression_branch(features)

        du_dv = regression[:, 0:2]
        dh_dw = regression[:, 2:4]

        dh_dw = torch.exp(torch.clamp(dh_dw, max=10))

        regression_out = torch.cat([du_dv, dh_dw], dim=1)

        return heatmap, regression_out