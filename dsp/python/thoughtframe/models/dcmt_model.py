import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3,
            stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class DcmtModel(nn.Module):
    """
    Input:  (B, 1, F, T) log-mel
    Output: (B, D) embedding
    """

    def __init__(self, emb_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            DepthwiseBlock(1, 16),
            DepthwiseBlock(16, 32, stride=(2, 2)),
            DepthwiseBlock(32, 64, stride=(2, 2)),
            DepthwiseBlock(64, 128, stride=(2, 1)),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv(x)          # (B, C, F', T')
        x = x.mean(dim=2)         # freq pooling → (B, C, T')
        x = x.transpose(1, 2)     # (B, T', C)

        x, _ = self.attn(x, x, x)
        x = x.mean(dim=1)         # time pooling

        return F.normalize(self.proj(x), dim=-1)
